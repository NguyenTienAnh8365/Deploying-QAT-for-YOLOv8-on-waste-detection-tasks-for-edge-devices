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
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from .quant_ops_pruned import quant_module_change_pruned


def _sanitize_quantizer_amax(model, default_amax=1.0, verbose=True):
    """Fix _amax values that are NaN/Inf/≤0 — TensorRT rejects non-positive scales.

    Nguyên nhân: trong QAT calibration, vài quantizer có thể không được kích hoạt
    (đặc biệt ở head branches của pruned model) → _amax ở trạng thái init (0 hoặc NaN).
    """
    fixed, disabled = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        amax = getattr(module, "_amax", None)
        if amax is None:
            continue
        bad = ~torch.isfinite(amax) | (amax <= 0)
        if bad.any():
            fixed += int(bad.sum().item())
            amax.data[bad] = default_amax
            if verbose:
                print(f"[sanitize] {name}: fixed {int(bad.sum().item())}/{amax.numel()} bad amax")
    if verbose:
        print(f"[sanitize] Total quantizer entries fixed: {fixed}, disabled: {disabled}")
    return model


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
    # Ultralytics 8.3+ saves ckpt['model']=None, real weights in ckpt['ema'].
    ckpt = torch.load(qat_weight, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        model_obj = ckpt.get('model') or ckpt.get('ema')
        if model_obj is None:
            raise ValueError(f"No model/ema in {qat_weight}. Keys: {list(ckpt.keys())}")
        state_dict = model_obj.float().state_dict()
    else:
        state_dict = ckpt.float().state_dict() if hasattr(ckpt, 'float') else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _sanitize_quantizer_amax(model, default_amax=1.0, verbose=verbose)

    # ── Set export mode cho DetectPruned head ──────────────────────────
    for m in model.modules():
        if isinstance(m, DetectPruned):
            m.export = True
            m.format = format

    # ── Move model + dummy input to CUDA ────────────────────────────────
    # Quantizer _amax buffers loaded from state_dict có thể nằm trên cuda;
    # phải đồng bộ device để tránh "scale on cuda, weight on cpu".
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    # ── Export to ONNX ──────────────────────────────────────────────────
    output_path = Path(qat_weight).with_suffix(".tensorrt.onnx")

    # Force TorchScript-based exporter (dynamo=False).
    # Torch 2.10 default dynamo exporter uses torch.export which can't trace
    # pytorch_quantization's .item() in _fb_fake_quant (data-dependent symbolic).
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        verbose=False,
        opset_version=13,
        input_names=["image"],
        output_names=["output"],
        dynamo=False,
    )

    print(f"\n[QAT-Pruned] Exported to: {output_path}")
    print(f"[QAT-Pruned] Convert to TensorRT engine:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.engine --int8")

    return model
