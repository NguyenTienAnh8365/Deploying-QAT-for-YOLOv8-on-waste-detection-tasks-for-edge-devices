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
from .qat_pruned_trainer import _skip_detect_quantizers


def _recalibrate_quantizers(model, images_dir, imgsz=640, max_images=200, device='cuda'):
    """Re-run calibration bằng ảnh thật để compute _amax chính xác.

    Dùng khi QAT training không calibrate được hết các quantizer
    (thường xảy ra với detection head — toàn bộ _amax = NaN).
    """
    from pathlib import Path
    import cv2
    import numpy as np
    from pytorch_quantization import calib

    paths = sorted(Path(images_dir).glob('*.jpg'))[:max_images]
    if not paths:
        paths = sorted(Path(images_dir).glob('*.png'))[:max_images]
    if not paths:
        print(f"[calibrate] ⚠️ No images in {images_dir} — skipping re-calibration")
        return model
    print(f"\n[calibrate] === Re-calibrating with {len(paths)} images from {images_dir} ===")

    # Enable calibration, disable quantization
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    model.eval().to(device)
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            scale = imgsz / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh))
            canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            canvas[:nh, :nw] = img
            t = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            model(t)
            if (i + 1) % 50 == 0:
                print(f"[calibrate] {i+1}/{len(paths)}")

    # Compute amax, re-enable quantization
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._calibrator is not None:
                try:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax("percentile", percentile=99.99)
                except Exception as e:
                    print(f"[calibrate] {name}: load_calib_amax failed — {e}")
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    print(f"[calibrate] === DONE ===\n")
    return model


def _sanitize_quantizer_amax(model, default_amax=1.0):
    """Fix _amax values that are NaN/Inf/≤0 — TensorRT rejects non-positive scales.

    Nguyên nhân: trong QAT calibration, vài quantizer có thể không được kích hoạt
    (đặc biệt ở head branches của pruned model) → _amax ở trạng thái init (0 hoặc NaN).
    """
    print(f"\n[sanitize] === START quantizer _amax sanitization ===")
    total_quantizers, amax_missing, fixed_entries, fixed_modules = 0, 0, 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        total_quantizers += 1
        if not hasattr(module, "_amax"):
            amax_missing += 1
            # Quantizer chưa có _amax buffer → tạo mới với default để tránh scale=0 khi export
            module.register_buffer("_amax", torch.tensor(default_amax))
            print(f"[sanitize] {name}: NO _amax → init to {default_amax}")
            fixed_modules += 1
            continue
        amax = module._amax
        bad = ~torch.isfinite(amax) | (amax <= 0)
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            amax.data[bad] = default_amax
            fixed_entries += n_bad
            fixed_modules += 1
            print(f"[sanitize] {name}: fixed {n_bad}/{amax.numel()} bad entries "
                  f"(min={amax.min().item():.4g}, max={amax.max().item():.4g})")
    print(f"[sanitize] Scanned {total_quantizers} quantizers | "
          f"missing _amax: {amax_missing} | "
          f"modules fixed: {fixed_modules} | entries fixed: {fixed_entries}")
    print(f"[sanitize] === DONE ===\n")
    return model


def export_onnx_pruned(pruned_checkpoint, qat_weight, format="onnx", verbose=True,
                       calib_images=None, num_calib_images=200, imgsz=640):
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

    # ── Disable head quantizers (khớp với Technique 3 trong training) ─
    # Trainer cố ý disable head quantizers → head chạy FP32.
    # Export phải làm cùng; nếu không, NaN _amax của head sẽ được emit
    # thành QDQ nodes → TRT reject scale=0/NaN.
    _skip_detect_quantizers(model)

    # ── Move model to device ──────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ── (Optional) Re-calibrate body quantizers bằng ảnh thật ─────────
    if calib_images is not None:
        _recalibrate_quantizers(model, calib_images, imgsz=imgsz,
                                max_images=num_calib_images, device=device)

    # ── Sanitize entries còn sót (dự phòng cho body quantizers) ───────
    _sanitize_quantizer_amax(model, default_amax=1.0)

    # ── Set export mode cho DetectPruned head ──────────────────────────
    for m in model.modules():
        if isinstance(m, DetectPruned):
            m.export = True
            m.format = format

    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)

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
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )

    print(f"\n[QAT-Pruned] Exported to: {output_path}")
    print(f"[QAT-Pruned] Convert to TensorRT engine:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.engine --int8")

    return model
