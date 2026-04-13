# Ultralytics YOLO 🚀, AGPL-3.0 license
# Fixed version — patches 4 bugs causing "does not require grad" error
#
# BUG 1 (ROOT CAUSE của crash): always_freeze_names = ['.dfl'] + weight
#   quantizers disabled trong calibration nhưng _amax không bao giờ được set
#   → khi enable() lại, quantizer có _amax=None → forward trả về tensor
#   detached khỏi computation graph → loss.grad_fn = None → backward() crash
#
# BUG 2: monkey-patch model.train() bằng attribute assignment
#   → deepcopy trong ModelEMA copy cả monkey-patch → EMA model bị lệch
#   → fix: dùng register_forward_pre_hook thay thế
#
# BUG 3: lr0 /= 100 trong _setup_train nhưng scheduler dùng lr0 sau khi chia
#   → user truyền lr0=1e-5 thực ra chạy 1e-7, gần như không update gì
#   → fix: document rõ, không thay đổi behavior để giữ tương thích
#
# BUG 4: sau _recalibrate(), model.train() cần đảm bảo BN vẫn eval
#   → fix: dùng hook thay vì monkey-patch

from copy import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch import distributed as dist
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, colorstr, callbacks
from ultralytics.utils.torch_utils import de_parallel, is_parallel, ModelEMA
from copy import deepcopy
from datetime import datetime
import logging
from torch import nn, optim
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
from pytorch_quantization import quant_modules
from .quant_ops import quant_module_change
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from tqdm import tqdm
import torch.distributed as dist


# =============================================================================
# FIX BUG 1 — cal_model được viết lại để xử lý quantizers không có calibrator
# =============================================================================

def _set_quantizer_state(model, enable_quant: bool, enable_calib: bool):
    """
    Đặt trạng thái đồng nhất cho tất cả TensorQuantizer trong model.

    Thay thế pattern cũ:
        module.disable()  ← BUG: để _amax=None, enable() sau đó → crash

    Logic mới:
        - quantizer CÓ calibrator  → dùng calib mode hoặc quant mode
        - quantizer KHÔNG calibrator → luôn enable quant (weight quantizers
          đã có _amax từ load_state_dict hoặc từ khởi tạo mặc định)
    """
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue

        if enable_calib:
            # Calibration mode: thu thập activation statistics
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                # Weight quantizer: không có calibrator, giữ enable để
                # _amax không bị None sau khi chuyển về quant mode
                module.enable_quant()
                # Không enable_calib vì không có calibrator
        else:
            # Training / inference mode: fake quantization
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                # Weight quantizer: đảm bảo enable và _amax hợp lệ
                if module._amax is None:
                    # _amax chưa có → disable tạm thời thay vì crash
                    LOGGER.warning(
                        f"[QAT] TensorQuantizer '{name}' has no _amax "
                        f"and no calibrator — disabling to prevent detach."
                    )
                    module.disable()
                else:
                    module.enable_quant()


def _verify_amax(model, device):
    """
    Sau calibration: đảm bảo tất cả enabled quantizers đều có _amax hợp lệ.
    Nếu không → disable để tránh forward trả về tensor detached.
    """
    fixed = 0
    for name, module in model.named_modules():
        if not isinstance(module, quant_nn.TensorQuantizer):
            continue
        if module._disabled:
            continue
        if module._amax is None:
            LOGGER.warning(
                f"[QAT] '{name}': _amax is None after calibration → disabling."
            )
            module.disable()
            fixed += 1
        else:
            module._amax = module._amax.to(device)
    if fixed:
        LOGGER.warning(f"[QAT] Disabled {fixed} quantizers with missing _amax.")


def cal_model(model, data_loader, device, num_batch=256):
    """Calibrate quantization scales using percentile method."""

    def collect_stats(model, data_loader, device, num_batch):
        model.eval()
        _set_quantizer_state(model, enable_quant=False, enable_calib=True)
        with torch.no_grad():
            for i, datas in tqdm(
                enumerate(data_loader),
                total=num_batch,
                desc="[QAT] Collecting calibration stats"
            ):
                imgs = datas['img'].to(device, non_blocking=True).float() / 255.0
                model(imgs)
                if i >= num_batch:
                    break
        # Restore: quant ON, calib OFF
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
    # Entropy: tìm range tối ưu bằng KL divergence, tốt hơn percentile cho class khó
    compute_amax(model, device, method="entropy")

    # FIX BUG 1: kiểm tra và vô hiệu hóa quantizer không có _amax
    _verify_amax(model, device)


# =============================================================================
# FIX BUG 2 — BN freeze dùng hook thay vì monkey-patch
# =============================================================================

def _register_bn_freeze_hook(model):
    """
    Đăng ký forward pre-hook để giữ BatchNorm ở eval mode trong QAT.

    Lý do dùng hook thay vì monkey-patch model.train():
        - deepcopy (dùng trong ModelEMA) copy cả monkey-patch → EMA bị hỏng
        - Hook được copy đúng cách bởi deepcopy
        - Hook chỉ ảnh hưởng forward, không ảnh hưởng model.train() semantics

    Cơ chế:
        Trước mỗi forward call, hook set BN layers về eval mode.
        BN trong eval mode dùng running_mean/running_var đã calibrate
        thay vì batch statistics bị nhiễu bởi fake quantization noise.
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))
    ]

    handles = []

    def _bn_eval_hook(module, input):
        """Pre-hook: set BN về eval trước mỗi forward."""
        module.eval()

    for bn in bn_layers:
        handle = bn.register_forward_pre_hook(_bn_eval_hook)
        handles.append(handle)

    LOGGER.info(f"[QAT] Registered BN freeze hooks on {len(bn_layers)} BatchNorm layers.")
    return handles  # giữ reference để có thể remove nếu cần


# =============================================================================
# TECHNIQUE 3 — Skip Detect head quantizers (giữ FP32 precision)
# =============================================================================

def _skip_detect_quantizers(model):
    """Disable quantizers trong Detect head để giữ precision cao cho detection.

    Detect head là layer nhạy nhất với quantization vì nó tính bbox + cls scores
    trực tiếp. Giữ FP32 ở đây giúp giảm mAP drop đáng kể, đặc biệt class khó.
    """
    model_inner = de_parallel(model) if is_parallel(model) else model
    # Detect head luôn là layer cuối cùng
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
        f"[QAT] Technique 3: disabled {disabled} quantizers in Detect head "
        f"(model.{detect_idx}) → FP32 precision"
    )
    return disabled


# =============================================================================
# TRAINER
# =============================================================================

class QuantizationTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.qat = 'nvidia'
        self.recalib_every = overrides.get('recalib_every', 1) if overrides else 1
        self._bn_hook_handles = []   # giữ reference tới BN hooks

    @staticmethod
    def _reset_ckpt_args(args):
        include = {'imgsz', 'data', 'task', 'single_cls'}
        return {k: v for k, v in args.items() if k in include}

    def get_model(self, cfg=None, weights=None, verbose=True):
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=16 * 2, rank=-1, mode='val'
            )

        quant_modules.initialize()

        if RANK in (-1, 0):
            # Technique 2: dùng train data (đa dạng hơn val) cho calibration
            # mode='val' → không augment, chỉ letterbox → activation distribution sát thực tế
            calib_loader = self.get_dataloader(
                self.trainset, batch_size=16 * 2, rank=-1, mode='val'
            )
            LOGGER.info(
                f"[QAT] Calibrating on train set ({len(calib_loader.dataset)} images, "
                f"512 batches) — entropy method"
            )

            # Calibration trên temp model
            _temp_model = DetectionModel(
                cfg=cfg, ch=3, nc=self.data['nc'], verbose=False
            )
            quant_module_change(_temp_model)
            if weights is not None:
                _temp_model.load_state_dict(
                    torch.load(weights, weights_only=False)['model'].state_dict()
                )
            _temp_model.cuda()
            cal_model(_temp_model, calib_loader, 'cuda', num_batch=512)
            torch.save(_temp_model.state_dict(), self.wdir / '_temp_calibration.pt')
            del _temp_model
            torch.cuda.empty_cache()

        if RANK != -1:
            dist.barrier()

        # Load calibrated weights vào model chính
        model = DetectionModel(cfg=cfg, ch=3, nc=self.data['nc'], verbose=False)
        quant_module_change(model)
        model.load_state_dict(
            torch.load(self.wdir / '_temp_calibration.pt', weights_only=False)
        )

        # Technique 3: Detect head nhạy nhất → giữ ở FP32
        _skip_detect_quantizers(model)

        return model

    def setup_model(self):
        return None

    def _setup_scheduler(self):
        """Cosine annealing: lr từ lr0 → lr0 * 0.01 trong T_max epochs."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * 1.0,
            eta_min=self.args.lr0 * 0.01
        )

    def _setup_train(self, world_size):
        """Setup training với QAT-specific fixes."""
        self.args.warmup_epochs = 0

        # ── LR scaling nội bộ ────────────────────────────────────────────────
        # Script này chia lr0 /= 100 để CosineAnnealingLR bắt đầu từ lr0/100.
        # Ví dụ: user truyền lr0=1e-3 → scheduler bắt đầu ở 1e-5
        # Nếu bạn muốn LR thực tế = lr0 bạn truyền vào,
        # hãy truyền lr0 = target_lr * 100 (ví dụ target 1e-5 → truyền 1e-3)
        # ─────────────────────────────────────────────────────────────────────
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

        # FIX BUG 1: KHÔNG freeze .dfl tự động
        # DFL weights không cần update (fixed lookup table), nhưng việc freeze
        # chúng kết hợp với weight quantizer không có _amax → crash
        # Để an toàn: chỉ freeze những gì user chỉ định
        always_freeze_names = []  # ← đã bỏ '.dfl'
        freeze_layer_names = [
            f'model.{x}.' for x in freeze_list
        ] + always_freeze_names

        trainable_count = 0
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            else:
                if not v.requires_grad:
                    v.requires_grad = True
                trainable_count += 1

        LOGGER.info(f"[QAT] Trainable parameters: {trainable_count}")

        # ── Sanity check: đảm bảo ít nhất 1 param có grad ───────────────────
        if trainable_count == 0:
            raise RuntimeError(
                "[QAT] No trainable parameters found! "
                "All layers are frozen — training would produce no gradients. "
                "Reduce --freeze value."
            )

        # ── FIX BUG 2: BN freeze dùng hook thay vì monkey-patch ─────────────
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
            self.trainset, batch_size=batch_size, rank=RANK, mode='train'
        )
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size * 2, rank=-1, mode='val'
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

    # ── FIX BUG 2: _freeze_bn_for_qat dùng hook (giữ tên để tương thích) ────
    def _freeze_bn_for_qat(self):
        """
        Deprecated: đã được thay bằng hook trong _setup_train.
        Giữ lại để không break nếu code khác gọi method này.
        """
        model = de_parallel(self.model) if is_parallel(self.model) else self.model
        if not self._bn_hook_handles:
            self._bn_hook_handles = _register_bn_freeze_hook(model)

    def _recalibrate(self):
        """Re-calibrate quantization scales sau mỗi epoch."""
        LOGGER.info(
            f"[QAT] Re-calibrating at epoch {self.epoch + 1}..."
        )
        model = de_parallel(self.model) if is_parallel(self.model) else self.model

        # Tạm xóa BN hooks để calibration chạy đúng (cần BN ở train mode)
        for h in self._bn_hook_handles:
            h.remove()
        self._bn_hook_handles = []

        model.eval()
        cal_model(model, self.test_loader, self.device, num_batch=256)

        # Technique 3: re-apply sau mỗi recalibration (cal_model re-enables quantizers)
        _skip_detect_quantizers(model)

        # Khôi phục BN hooks và set model về train mode
        self._bn_hook_handles = _register_bn_freeze_hook(model)
        model.train()
        # Hooks sẽ tự set BN về eval trong forward tiếp theo

        # Sync _amax sang EMA model
        if self.ema:
            ema_model = self.ema.ema
            model_buffers = dict(model.named_buffers())
            for name, ema_buf in ema_model.named_buffers():
                if '_amax' in name and name in model_buffers:
                    ema_buf.data.copy_(model_buffers[name].data)

        LOGGER.info("[QAT] Re-calibration done.")

    def validate(self):
        """Re-calibrate trước validation nếu cần."""
        if self.recalib_every > 0 and RANK in (-1, 0):
            if self.epoch == 0 or (self.epoch + 1) % self.recalib_every == 0:
                self._recalibrate()
        return super().validate()