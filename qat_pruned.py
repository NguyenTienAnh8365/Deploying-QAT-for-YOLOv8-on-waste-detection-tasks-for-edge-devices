"""
QAT (Quantization-Aware Training) cho pruned YOLOv8 model.

Pipeline hoàn chỉnh:
  1. train.py --mode sparsity    → Train với sparsity regularization
  2. prune.py                     → Cắt tỉa channels theo BN gamma
  3. train.py --mode finetune    → Finetune pruned model (khôi phục accuracy)
  4. qat_pruned.py               → QAT trên pruned model (FILE NÀY)
  5. qat_pruned_export.py        → Export sang TensorRT-compatible ONNX

Ví dụ chạy:
  # QAT trực tiếp trên pruned model (chưa finetune):
  python qat_pruned.py \\
      --pruned-checkpoint weights/pruned.pt \\
      --data-config data.yaml \\
      --epochs 10 --batch 16

  # QAT trên finetuned pruned model (khuyên dùng):
  python qat_pruned.py \\
      --pruned-checkpoint weights/pruned.pt \\
      --pretrained-weight runs/train-finetune/weights/best.pt \\
      --data-config data.yaml \\
      --epochs 10 --batch 16

Lưu ý LR:
  - QuantizationPrunedTrainer tự chia lr0 /= 100 trong _setup_train
  - Muốn QAT dùng lr=1e-5 → truyền --lr0 1e-3 (vì 1e-3 / 100 = 1e-5)
"""

from ultralytics.qat.nvidia_tensorrt.qat_pruned_trainer import QuantizationPrunedTrainer
import argparse
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER


def train(args):
    LOGGER.info(f"[QAT-Pruned] Pruned checkpoint: {args.pruned_checkpoint}")
    if args.pretrained_weight:
        LOGGER.info(f"[QAT-Pruned] Finetuned weight: {args.pretrained_weight}")
    else:
        LOGGER.info(f"[QAT-Pruned] Using pruned checkpoint weights (no separate finetuned weight)")

    overrides = {}

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

    # ── Pruned-specific ──────────────────────────────────────────────────
    overrides['pruned_checkpoint'] = args.pruned_checkpoint

    trainer = QuantizationPrunedTrainer(cfg=DEFAULT_CFG_DICT.copy(), overrides=overrides)

    # Nếu có finetuned weight riêng, dùng nó; nếu không, dùng pruned checkpoint
    weight_to_load = args.pretrained_weight if args.pretrained_weight else args.pruned_checkpoint
    trainer.model = trainer.get_model(weights=weight_to_load)

    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAT training cho pruned YOLOv8 model"
    )

    # ── Pruned model ─────────────────────────────────────────────────────
    parser.add_argument("--pruned-checkpoint", type=str, required=True,
                        help="Path tới pruned.pt gốc (output của prune.py, chứa maskbndict)")
    parser.add_argument("--pretrained-weight", type=str, default=None,
                        help="Path tới finetuned weight (nếu đã finetune). "
                             "Nếu không truyền, dùng weights từ pruned checkpoint.")

    # ── Core ──────────────────────────────────────────────────────────────
    parser.add_argument("--model-config",      type=str,   default="yolov8s.yaml",
                        help="Model YAML config (dùng cho trainer setup, "
                             "model thực tế build từ pruned checkpoint)")
    parser.add_argument("--data-config",       type=str,   required=True)
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--imgsz",             type=int,   default=640)
    parser.add_argument("--batch",             type=int,   default=16)
    parser.add_argument("--device",            type=int,   default=0)
    parser.add_argument("--workers",           type=int,   default=8)
    parser.add_argument("--cache",             type=str,   default="ram")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # LƯU Ý: lr0 thực tế = lr0 / 100 (trainer chia tự động)
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

    # ── Augmentation (mặc định tắt hết cho QAT) ──────────────────────────
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
    parser.add_argument("--project",           type=str,   default="runs/qat-pruned")
    parser.add_argument("--name",              type=str,   default="train")
    parser.add_argument("--exist-ok",          action="store_true", default=True)
    parser.add_argument("--plots",             action="store_true", default=True)
    parser.add_argument("--recalib-every",     type=int,   default=1,
                        help="Re-calibrate scales mỗi N epoch (0=tắt)")

    args = parser.parse_args()
    train(args)
