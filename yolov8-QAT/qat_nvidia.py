from ultralytics.qat.nvidia_tensorrt.qat_nvidia_trainer import QuantizationTrainer
import argparse
import torch
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER


def _reset_ckpt_args(args):
    include = {'imgsz', 'data', 'task', 'single_cls', 'model'}
    return {k: v for k, v in args.items() if k in include}


def train(args):
    LOGGER.info(f"Loading custom weight: {args.pretrained_weight}")
    model_custom, ckpt = attempt_load_one_weight(args.pretrained_weight)

    overrides = _reset_ckpt_args(model_custom.args)

    # ── Core ──────────────────────────────────────────────────────────────
    overrides['model']   = args.model_config
    overrides['data']    = args.data_config
    overrides['epochs']  = args.epochs
    overrides['imgsz']   = args.imgsz
    overrides['batch']   = args.batch
    overrides['device']  = args.device
    overrides['workers'] = args.workers
    overrides['cache']   = args.cache

    # ── Optimizer ─────────────────────────────────────────────────────────
    # LƯU Ý: QuantizationTrainer tự chia lr0 /= 100 trong _setup_train
    # Ví dụ: muốn QAT dùng lr=1e-5 → truyền lr0=1e-3 (1e-3 / 100 = 1e-5)
    overrides['optimizer']     = args.optimizer
    overrides['lr0']           = args.lr0
    overrides['lrf']           = args.lrf
    overrides['momentum']      = args.momentum
    overrides['weight_decay']  = args.weight_decay

    # ── Freeze ────────────────────────────────────────────────────────────
    overrides['freeze'] = args.freeze

    # ── Loss weights ──────────────────────────────────────────────────────
    overrides['box'] = args.box
    overrides['cls'] = args.cls
    overrides['dfl'] = args.dfl

    # ── Augmentation (khuyến nghị tắt hết cho QAT) ────────────────────────
    overrides['augment']    = args.augment
    overrides['mosaic']     = args.mosaic
    overrides['mixup']      = args.mixup
    overrides['copy_paste'] = args.copy_paste
    overrides['hsv_h']      = args.hsv_h
    overrides['hsv_s']      = args.hsv_s
    overrides['hsv_v']      = args.hsv_v
    overrides['flipud']     = args.flipud
    overrides['fliplr']     = args.fliplr

    # ── Control ───────────────────────────────────────────────────────────
    overrides['patience']      = args.patience
    overrides['deterministic'] = args.deterministic
    overrides['project']       = args.project
    overrides['name']          = args.name
    overrides['exist_ok']      = args.exist_ok
    overrides['plots']         = args.plots
    overrides['recalib_every'] = args.recalib_every

    # amp và warmup_epochs bị override cứng trong QuantizationTrainer._setup_train
    # không cần truyền vào đây

    trainer = QuantizationTrainer(cfg=DEFAULT_CFG_DICT.copy(), overrides=overrides)
    trainer.model = trainer.get_model(cfg=args.model_config, weights=args.pretrained_weight)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Core ──────────────────────────────────────────────────────────────
    parser.add_argument("--model-config",      type=str,   default="yolov8s.yaml")
    parser.add_argument("--pretrained-weight", type=str,   required=True)
    parser.add_argument("--data-config",       type=str,   required=True)
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--imgsz",             type=int,   default=640)
    parser.add_argument("--batch",             type=int,   default=16)
    parser.add_argument("--device",            type=int,   default=0)
    parser.add_argument("--workers",           type=int,   default=8)
    parser.add_argument("--cache",             type=str,   default="ram")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # LƯU Ý: lr0 thực tế = lr0 / 100 (QuantizationTrainer chia tự động)
    # Muốn QAT chạy với lr=1e-5 → đặt --lr0 1e-3
    parser.add_argument("--optimizer",         type=str,   default="AdamW")
    parser.add_argument("--lr0",               type=float, default=1e-3)
    parser.add_argument("--lrf",               type=float, default=0.1)
    parser.add_argument("--momentum",          type=float, default=0.90)
    parser.add_argument("--weight-decay",      type=float, default=0.0004)

    # ── Freeze ────────────────────────────────────────────────────────────
    parser.add_argument("--freeze",            type=int,   default=10)

    # ── Loss weights ──────────────────────────────────────────────────────
    parser.add_argument("--box",               type=float, default=0.04)
    parser.add_argument("--cls",               type=float, default=2.5)
    parser.add_argument("--dfl",               type=float, default=1.2)

    # ── Augmentation ──────────────────────────────────────────────────────
    parser.add_argument("--augment",           action="store_true", default=False)
    parser.add_argument("--mosaic",            type=float, default=0.0)
    parser.add_argument("--mixup",             type=float, default=0.0)
    parser.add_argument("--copy-paste",        type=float, default=0.0)
    parser.add_argument("--hsv-h",             type=float, default=0.0)
    parser.add_argument("--hsv-s",             type=float, default=0.0)
    parser.add_argument("--hsv-v",             type=float, default=0.0)
    parser.add_argument("--flipud",            type=float, default=0.0)
    parser.add_argument("--fliplr",            type=float, default=0.0)

    # ── Control ───────────────────────────────────────────────────────────
    parser.add_argument("--patience",          type=int,   default=5)
    parser.add_argument("--deterministic",     action="store_true", default=True)
    parser.add_argument("--project",           type=str,   default="runs/qat")
    parser.add_argument("--name",              type=str,   default="train")
    parser.add_argument("--exist-ok",          action="store_true", default=True)
    parser.add_argument("--plots",             action="store_true", default=True)
    parser.add_argument("--recalib-every",     type=int,   default=1, help="Re-calibrate scales moi N epoch (0=tat)")

    args = parser.parse_args()
    train(args)
