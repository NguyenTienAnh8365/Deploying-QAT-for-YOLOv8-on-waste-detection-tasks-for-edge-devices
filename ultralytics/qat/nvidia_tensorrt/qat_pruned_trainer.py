# Ultralytics YOLO - QAT Trainer cho Pruned Model
# Adapted từ yolov8-QAT/ultralytics/qat/nvidia_tensorrt/qat_nvidia_trainer.py
#
# Thay đổi chính so với bản gốc:
#   - get_model() load pruned checkpoint (có maskbndict) thay vì tạo DetectionModel
#   - Dùng DetectionModelPruned + quant_module_change_pruned thay vì DetectionModel + quant_module_change
#   - Tất cả kỹ thuật QAT giữ nguyên: BN freeze, Detect FP32, calibration entropy, recalibration
#
# Pipeline: train sparsity → prune → finetune → QAT (file này) → export TensorRT

from copy import copy, deepcopy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch import distributed as dist
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, colorstr, callbacks
from ultralytics.utils.torch_utils import unwrap_model as de_parallel, is_parallel, ModelEMA
from datetime import datetime
import logging
from torch import nn, optim
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
from pytorch_quantization import quant_modules
from .quant_ops_pruned import quant_module_change_pruned
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from tqdm import tqdm
import torch.distributed as dist


# =============================================================================
# Calibration utilities (giữ nguyên từ bản gốc — model-agnostic)
# =============================================================================

def _set_quantizer_state(model, enable_quant: bool, enable_calib: bool):
    """
    Đặt trạng thái đồng nhất cho tất cả TensorQuantizer trong model.

    Logic:
        - quantizer CÓ calibrator  → dùng calib mode hoặc quant mode
        - quantizer KHÔNG calibrator → luôn enable quant (weight quantizers
          đã có _amax từ load_state_dict hoặc từ khởi tạo mặc định)
    """
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue

        if enable_calib:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.enable_quant()
        else:
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                if module._amax is None:
                    LOGGER.warning(
                        f"[QAT-Pruned] TensorQuantizer '{name}' has no _amax "
                        f"and no calibrator — disabling to prevent detach."
                    )
                    module.disable()
                else:
                    module.enable_quant()


def _verify_amax(model, device):
    """Sau calibration: đảm bảo tất cả enabled quantizers đều có _amax hợp lệ."""
    fixed = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        if module._disabled:
            continue
        if module._amax is None:
            LOGGER.warning(
                f"[QAT-Pruned] '{name}': _amax is None after calibration → disabling."
            )
            module.disable()
            fixed += 1
        else:
            module._amax = module._amax.to(device)
    if fixed:
        LOGGER.warning(f"[QAT-Pruned] Disabled {fixed} quantizers with missing _amax.")


def cal_model(model, data_loader, device, num_batch=256):
    """Calibrate quantization scales bằng entropy method (KL divergence)."""

    def collect_stats(model, data_loader, device, num_batch):
        model.eval()
        _set_quantizer_state(model, enable_quant=False, enable_calib=True)
        with torch.no_grad():
            for i, datas in tqdm(
                enumerate(data_loader),
                total=num_batch,
                desc="[QAT-Pruned] Collecting calibration stats"
            ):
                imgs = datas['img'].to(device, non_blocking=True).float() / 255.0
                model(imgs)
                if i >= num_batch:
                    break
        _set_quantizer_state(model, enable_quant=True, enable_calib=False)

    def compute_amax(model, device, **kwargs):
        for name, module in model.named_modules():
            if not isinstance(module, quant_nn.TensorQuantizer):
                continue
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs)
                if module._amax is not None:
                    module._amax = module._amax.to(device)

    collect_stats(model, data_loader, device, num_batch)
    compute_amax(model, device, method="entropy")
    _verify_amax(model, device)


# =============================================================================
# BN freeze hook (giữ nguyên — model-agnostic)
# =============================================================================

def _bn_eval_hook(module, input):
    """Top-level (picklable) BN forward pre-hook — giữ BN ở eval mode."""
    module.eval()


def _register_bn_freeze_hook(model):
    """
    Đăng ký forward pre-hook để giữ BatchNorm ở eval mode trong QAT.

    Lý do dùng hook thay vì monkey-patch model.train():
        - deepcopy (dùng trong ModelEMA) copy cả monkey-patch → EMA bị hỏng
        - Hook chỉ ảnh hưởng forward, không ảnh hưởng model.train() semantics

    BN trong eval mode dùng running_mean/running_var đã calibrate
    thay vì batch statistics bị nhiễu bởi fake quantization noise.

    Note: hook phải là top-level function (không phải closure) để pickle
    được khi save_model() deepcopy EMA.
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))
    ]

    handles = []
    for bn in bn_layers:
        handle = bn.register_forward_pre_hook(_bn_eval_hook)
        handles.append(handle)

    LOGGER.info(f"[QAT-Pruned] Registered BN freeze hooks on {len(bn_layers)} BatchNorm layers.")
    return handles


# =============================================================================
# Skip Detect head quantizers — giữ FP32 cho detection head
# =============================================================================

def _skip_detect_quantizers(model):
    """Disable quantizers trong Detect head (DetectPruned) để giữ FP32 precision.

    Detect head là layer nhạy nhất với quantization vì nó tính bbox + cls scores
    trực tiếp. Giữ FP32 ở đây giúp giảm mAP drop đáng kể.
    """
    model_inner = de_parallel(model) if is_parallel(model) else model
    detect_idx = str(len(model_inner.model) - 1)

    disabled = 0
    for name, module in model_inner.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'model' and parts[1] == detect_idx:
            module.disable()
            disabled += 1

    LOGGER.info(
        f"[QAT-Pruned] Technique 3: disabled {disabled} quantizers in DetectPruned head "
        f"(model.{detect_idx}) → FP32 precision"
    )
    return disabled


# =============================================================================
# TRAINER
# =============================================================================

class QuantizationPrunedTrainer(DetectionTrainer):
    """QAT Trainer cho pruned YOLOv8 model.

    Kế thừa toàn bộ training loop từ DetectionTrainer, chỉ thay đổi:
      - get_model(): build quantized model từ pruned checkpoint
      - _setup_train(): tắt AMP, freeze BN, LR scaling cho QAT
      - validate(): re-calibrate quantization scales trước mỗi validation

    Kỹ thuật QAT áp dụng (giữ nguyên từ bản gốc):
      1. Calibration entropy (KL divergence) — tìm range tối ưu
      2. Dùng train data cho calibration (đa dạng hơn val)
      3. DetectPruned head giữ FP32 (nhạy nhất với quantization)
      4. BN freeze hooks (eval mode trong QAT training)
      5. LR scaling: lr0 /= 100 cho fine-tuning nhẹ
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Pop QAT-specific keys before super().__init__ — get_cfg's
        # check_dict_alignment rejects keys not in DEFAULT_CFG_DICT.
        overrides = overrides or {}
        recalib_every = overrides.pop('recalib_every', 1)
        pruned_checkpoint = overrides.pop('pruned_checkpoint', None)
        super().__init__(cfg, overrides, _callbacks)
        self.qat = 'nvidia'
        self.recalib_every = recalib_every
        self._bn_hook_handles = []
        self.pruned_checkpoint = pruned_checkpoint

    @staticmethod
    def _reset_ckpt_args(args):
        include = {'imgsz', 'data', 'task', 'single_cls'}
        return {k: v for k, v in args.items() if k in include}

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build quantized model từ pruned checkpoint.

        Flow:
          1. Load pruned checkpoint → lấy maskbndict + pruned config
          2. quant_modules.initialize() → patch nn.Conv2d → QuantConv2d
          3. Tạo DetectionModelPruned với QuantConv2d
          4. quant_module_change_pruned() → thêm quantizers cho C2fPruned, BottleneckPruned...
          5. Load pretrained weights (strict=False vì thêm quantizer params)
          6. Calibrate trên train data (entropy method, 512 batches)
          7. Skip DetectPruned head quantizers → giữ FP32

        Args:
            cfg: Không dùng (config lấy từ pruned checkpoint)
            weights: Path tới finetuned weight (nếu khác pruned checkpoint).
                     Nếu None, dùng weights từ pruned checkpoint.
        """
        # test_loader được tạo trong _setup_train sau khi model sẵn sàng;
        # không tạo ở đây vì self.model còn là str yaml path → build_dataset crash.

        # ── Load pruned checkpoint ──────────────────────────────────────────
        pruned_ckpt_path = self.pruned_checkpoint
        if pruned_ckpt_path is None:
            pruned_ckpt_path = weights
        if pruned_ckpt_path is None:
            raise ValueError(
                "[QAT-Pruned] Cần cung cấp --pruned-checkpoint hoặc --pretrained-weight"
            )

        LOGGER.info(f"[QAT-Pruned] Loading pruned checkpoint: {pruned_ckpt_path}")
        pruned_ckpt = torch.load(pruned_ckpt_path, map_location='cpu', weights_only=False)

        # Trích xuất maskbndict và config từ pruned checkpoint
        if isinstance(pruned_ckpt, dict) and 'maskbndict' in pruned_ckpt:
            maskbndict = pruned_ckpt['maskbndict']
            pruned_model_obj = pruned_ckpt['model']
        elif isinstance(pruned_ckpt, dict) and 'model' in pruned_ckpt:
            pruned_model_obj = pruned_ckpt['model']
            if hasattr(pruned_model_obj, 'maskbndict'):
                maskbndict = pruned_model_obj.maskbndict
            else:
                raise ValueError(
                    "[QAT-Pruned] Checkpoint không chứa maskbndict! "
                    "Hãy dùng file pruned.pt gốc (output của prune.py)."
                )
        else:
            raise ValueError(
                "[QAT-Pruned] Format checkpoint không hợp lệ! "
                "Cần dict với keys 'model' và 'maskbndict'."
            )

        pruned_cfg = pruned_model_obj.yaml
        LOGGER.info(
            f"[QAT-Pruned] Pruned config: nc={pruned_cfg['nc']}, "
            f"scale={pruned_cfg.get('scale', 'N/A')}"
        )

        # Lấy state dict từ pruned model hoặc finetuned weight
        # Ultralytics 8.3+ lưu ckpt['model']=None, weights nằm ở ckpt['ema'].
        if weights and weights != pruned_ckpt_path:
            LOGGER.info(f"[QAT-Pruned] Loading finetuned weights: {weights}")
            ft_ckpt = torch.load(weights, map_location='cpu', weights_only=False)
            if isinstance(ft_ckpt, dict):
                model_obj = ft_ckpt.get('model') or ft_ckpt.get('ema')
                if model_obj is None:
                    raise ValueError(
                        f"[QAT-Pruned] Không tìm thấy model/ema trong {weights}. "
                        f"Keys: {list(ft_ckpt.keys())}"
                    )
                original_state_dict = model_obj.float().state_dict()
            else:
                original_state_dict = ft_ckpt.float().state_dict()
        else:
            original_state_dict = pruned_model_obj.float().state_dict()

        # ── Initialize quantization ────────────────────────────────────────
        quant_modules.initialize()

        if RANK in (-1, 0):
            # build_dataset đọc self.model.stride; self.model còn là str yaml
            # → tạm set None để fallback gs=32 (stride lớn nhất của YOLOv8).
            _saved_model = self.model
            self.model = None
            # Technique 2: dùng train data cho calibration (đa dạng hơn val)
            calib_loader = self.get_dataloader(
                self.data["train"], batch_size=16 * 2, rank=-1, mode='val'
            )
            self.model = _saved_model
            LOGGER.info(
                f"[QAT-Pruned] Calibrating on train set "
                f"({len(calib_loader.dataset)} images, 512 batches) — entropy method"
            )

            # Tạo temp quantized pruned model cho calibration
            _temp_model = DetectionModelPruned(
                maskbndict=maskbndict, cfg=deepcopy(pruned_cfg), ch=3, verbose=False
            )
            quant_module_change_pruned(_temp_model)
            # Load weights — strict=False vì quantizer params chưa có trong state dict gốc
            _temp_model.load_state_dict(original_state_dict, strict=False)
            _temp_model.cuda()
            cal_model(_temp_model, calib_loader, 'cuda', num_batch=512)
            torch.save(_temp_model.state_dict(), self.wdir / '_temp_calibration.pt')
            del _temp_model
            torch.cuda.empty_cache()

        if RANK != -1:
            dist.barrier()

        # ── Build final quantized model ────────────────────────────────────
        model = DetectionModelPruned(
            maskbndict=maskbndict, cfg=deepcopy(pruned_cfg), ch=3, verbose=False
        )
        quant_module_change_pruned(model)
        model.load_state_dict(
            torch.load(self.wdir / '_temp_calibration.pt', map_location='cpu', weights_only=False)
        )

        # Technique 3: DetectPruned head giữ FP32
        _skip_detect_quantizers(model)

        return model

    def setup_model(self):
        """Override: model được tạo trong get_model(), không cần setup ở đây."""
        return None

    def _setup_scheduler(self):
        """Cosine annealing: lr từ lr0 → lr0 * 0.01 trong T_max epochs."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * 1.0,
            eta_min=self.args.lr0 * 0.01
        )

    def _setup_train(self, world_size=None):
        """Setup training với QAT-specific config.

        Giữ nguyên logic từ bản gốc:
          - warmup_epochs = 0 (model đã pretrained, không cần warmup)
          - lr0 /= 100 (fine-tuning nhẹ)
          - AMP = False (không tương thích với fake quantization)
          - BN freeze hooks (giữ BN eval mode)
          - Freeze layers theo --freeze parameter
        """
        self.args.warmup_epochs = 0

        # QAT không dùng sparsity regularization; BaseTrainer._do_train đọc self.sr.
        self.sr = 0.0

        # Resolve world_size — new BaseTrainer sets self.world_size internally
        if world_size is None:
            world_size = getattr(self, 'world_size', 1)

        # ── LR scaling nội bộ ────────────────────────────────────────────────
        # lr0 /= 100 → CosineAnnealingLR bắt đầu từ lr0/100
        # Ví dụ: user truyền lr0=1e-3 → scheduler bắt đầu ở 1e-5
        self.args.lr0 /= 100

        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ── Freeze layers ────────────────────────────────────────────────────
        freeze_list = (
            self.args.freeze if isinstance(self.args.freeze, list)
            else range(self.args.freeze) if isinstance(self.args.freeze, int)
            else []
        )

        always_freeze_names = []
        freeze_layer_names = [
            f'model.{x}.' for x in freeze_list
        ] + always_freeze_names
        # Ultralytics 8.3+ _model_train() reads self.freeze_layer_names for BN freeze.
        self.freeze_layer_names = freeze_layer_names

        trainable_count = 0
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            else:
                if not v.requires_grad:
                    v.requires_grad = True
                trainable_count += 1

        LOGGER.info(f"[QAT-Pruned] Trainable parameters: {trainable_count}")

        if trainable_count == 0:
            raise RuntimeError(
                "[QAT-Pruned] No trainable parameters found! "
                "All layers are frozen — training would produce no gradients. "
                "Reduce --freeze value."
            )

        # ── BN freeze hooks ─────────────────────────────────────────────────
        model_inner = de_parallel(self.model) if is_parallel(self.model) else self.model
        self._bn_hook_handles = _register_bn_freeze_hook(model_inner)

        # ── AMP: phải False trong QAT ────────────────────────────────────────
        self.amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[RANK]
            )

        # ── Check imgsz ──────────────────────────────────────────────────────
        gs = max(
            int(self.model.stride.max() if hasattr(self.model, 'stride') else 32),
            32
        )
        self.args.imgsz = check_imgsz(
            self.args.imgsz, stride=gs, floor=gs, max_dim=1
        )

        # ── Batch size ───────────────────────────────────────────────────────
        if self.batch_size == -1 and RANK == -1:
            self.args.batch = self.batch_size = check_train_batch_size(
                self.model, self.args.imgsz, self.amp
            )

        # ── Dataloaders ──────────────────────────────────────────────────────
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=RANK, mode='train'
        )
        val_path = self.data.get("val") or self.data.get("test")
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                val_path, batch_size=batch_size * 2, rank=-1, mode='val'
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # ── Optimizer ────────────────────────────────────────────────────────
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = (
            self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        )
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs))
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks('on_pretrain_routine_end')

    def _freeze_bn_for_qat(self):
        """Deprecated: đã được thay bằng hook trong _setup_train."""
        model = de_parallel(self.model) if is_parallel(self.model) else self.model
        if not self._bn_hook_handles:
            self._bn_hook_handles = _register_bn_freeze_hook(model)

    def _recalibrate(self):
        """Re-calibrate quantization scales sau mỗi epoch."""
        LOGGER.info(
            f"[QAT-Pruned] Re-calibrating at epoch {self.epoch + 1}..."
        )
        model = de_parallel(self.model) if is_parallel(self.model) else self.model

        # Tạm xóa BN hooks để calibration chạy đúng
        for h in self._bn_hook_handles:
            h.remove()
        self._bn_hook_handles = []

        model.eval()
        cal_model(model, self.test_loader, self.device, num_batch=256)

        # Technique 3: re-apply sau mỗi recalibration
        _skip_detect_quantizers(model)

        # Khôi phục BN hooks
        self._bn_hook_handles = _register_bn_freeze_hook(model)
        model.train()

        # Sync _amax sang EMA model
        if self.ema:
            ema_model = self.ema.ema
            model_buffers = dict(model.named_buffers())
            for name, ema_buf in ema_model.named_buffers():
                if '_amax' in name and name in model_buffers:
                    ema_buf.data.copy_(model_buffers[name].data)

        LOGGER.info("[QAT-Pruned] Re-calibration done.")

    def validate(self):
        """Re-calibrate trước validation nếu cần."""
        if self.recalib_every > 0 and RANK in (-1, 0):
            if self.epoch == 0 or (self.epoch + 1) % self.recalib_every == 0:
                self._recalibrate()
        return super().validate()
