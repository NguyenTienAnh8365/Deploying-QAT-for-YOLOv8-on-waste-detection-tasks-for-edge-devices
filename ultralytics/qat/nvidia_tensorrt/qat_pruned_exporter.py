"""
Export pruned + quantized YOLOv8 model to TensorRT-compatible ONNX.

Flow:
  1. Load pruned checkpoint → lấy maskbndict + pruned config
  2. quant_modules.initialize() → patch Conv2d → QuantConv2d
  3. Build DetectionModelPruned với QuantConv2d
  4. quant_module_change_pruned() → thêm quantizers
  5. Load QAT-trained state dict
  6. Export to ONNX (với fake quantization nodes cho TensorRT)

Output: file .tensorrt.onnx → dùng trtexec để convert sang .engine
"""

import torch
from copy import deepcopy
from pathlib import Path
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from .quant_ops_pruned import quant_module_change_pruned


def export_onnx_pruned(pruned_checkpoint, qat_weight, format="onnx", verbose=True):
    """Export pruned + quantized model to TensorRT-compatible ONNX.

    Args:
        pruned_checkpoint: Path tới file pruned.pt gốc (output của prune.py).
                          Chứa maskbndict + pruned model config.
        qat_weight: Path tới QAT-trained weight (best.pt từ QAT training).
        format: Export format (mặc định "onnx").

    Returns:
        model: Quantized pruned model (đã export).
    """
    # Bật fake quantization cho ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # ── Load pruned checkpoint để lấy maskbndict + config ────────────────
    pruned_ckpt = torch.load(pruned_checkpoint, map_location='cpu', weights_only=False)
    if isinstance(pruned_ckpt, dict) and 'maskbndict' in pruned_ckpt:
        maskbndict = pruned_ckpt['maskbndict']
        pruned_model_obj = pruned_ckpt['model']
    else:
        raise ValueError(
            f"Invalid pruned checkpoint format! "
            f"Expected dict with 'model' and 'maskbndict' keys."
        )

    pruned_cfg = pruned_model_obj.yaml

    # ── Build quantized pruned model ────────────────────────────────────
    quant_modules.initialize()

    model = DetectionModelPruned(
        maskbndict=maskbndict, cfg=deepcopy(pruned_cfg), ch=3, verbose=False
    )
    quant_module_change_pruned(model)

    # ── Load QAT-trained weights ────────────────────────────────────────
    ckpt = torch.load(qat_weight, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model'].float().state_dict()
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ── Set export mode cho DetectPruned head ──────────────────────────
    for m in model.modules():
        if isinstance(m, DetectPruned):
            m.export = True
            m.format = format

    # ── Export to ONNX ──────────────────────────────────────────────────
    output_path = Path(qat_weight).with_suffix(".tensorrt.onnx")
    dummy_input = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        verbose=False,
        opset_version=14,
        input_names=["image"],
        output_names=["output"]
    )

    print(f"\n[QAT-Pruned] Exported to: {output_path}")
    print(f"[QAT-Pruned] Convert to TensorRT engine:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.engine --int8")

    return model
