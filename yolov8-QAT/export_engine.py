"""
Export QAT YOLOv8 model: .pt -> ONNX (with Q/DQ nodes) -> TensorRT INT8 Engine

Usage:
    python export_engine.py \
        --model-config yolov8s.yaml \
        --weight runs/detect/train4/weights/best.pt \
        --nc 3 \
        --imgsz 640 \
        --workspace 4

Requirements:
    pip install pytorch-quantization tensorrt
    (TensorRT phải được cài sẵn, khuyến nghị dùng trên Linux/Jetson)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

from ultralytics.nn.modules import Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel
from ultralytics.qat.nvidia_tensorrt.quant_ops import quant_module_change


def export_onnx(cfg, weights, nc, imgsz=640):
    """Export QAT model sang ONNX với Q/DQ nodes để TensorRT nhận biết INT8."""
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()

    model = DetectionModel(cfg=cfg, ch=3, nc=nc, verbose=False)
    quant_module_change(model)

    # Hỗ trợ cả best.pt (full checkpoint) và _temp_calibration.pt (state dict thuần)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and ('model' in ckpt or 'ema' in ckpt):
        # Ưu tiên EMA model (chính xác hơn regular model)
        src = ckpt.get('ema') or ckpt['model']
        state_dict = src.float().state_dict()
        src_label = 'ema' if ckpt.get('ema') is not None else 'model'
        print(f"[INFO] Loaded checkpoint [{src_label}] from {weights}")
    else:
        state_dict = ckpt
        print(f"[INFO] Loaded state dict from {weights}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cpu()  # _amax buffers được calibrate trên CUDA → phải về CPU trước khi trace

    # Disable Detect head quantizers (giống training) để export đúng
    from pytorch_quantization import calib
    detect_idx = str(len(model.model) - 1)
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            parts = name.split('.')
            if len(parts) >= 2 and parts[0] == 'model' and parts[1] == detect_idx:
                module.disable()
                skipped += 1
    print(f"[INFO] Skipped {skipped} quantizers in Detect head (model.{detect_idx}) → FP32")

    for m in model.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.export = True
            m.format = "onnx"

    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    onnx_path = Path(weights).with_suffix(".tensorrt.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        verbose=False,
        opset_version=17,
        dynamo=False,   # Force old TorchScript exporter — new exporter (PyTorch 2.10+)
                        # không hỗ trợ data-dependent _amax của pytorch_quantization
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )
    print(f"[OK] ONNX saved: {onnx_path}")
    return onnx_path


def export_engine_trtexec(onnx_path, imgsz=640, workspace=4, fp16=False, max_batch=1):
    """Dùng trtexec CLI để build TensorRT INT8 engine từ ONNX có Q/DQ nodes."""
    engine_path = Path(onnx_path).with_suffix(".engine")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--int8",                          # Bật INT8 (Q/DQ nodes cung cấp scale)
        f"--workspace={workspace * 1024}", # MB -> KB
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:{max_batch}x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:{max_batch}x3x{imgsz}x{imgsz}",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[ERROR] trtexec thất bại. Kiểm tra TensorRT đã được cài và thêm vào PATH.")
        sys.exit(1)
    print(f"[OK] TensorRT engine saved: {engine_path}")
    return engine_path


def export_engine_python(onnx_path, imgsz=640, workspace=4, fp16=False, nc=80, names=None, max_batch=1):
    """Dùng Python TensorRT API để build engine (thay thế nếu trtexec không có)."""
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] tensorrt package không tìm thấy. Cài bằng: pip install tensorrt")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ONNX parse error] {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)   # Q/DQ nodes tự cung cấp scale, không cần calib thêm
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      (1, 3, imgsz, imgsz),           # min
                      (max_batch, 3, imgsz, imgsz),   # opt
                      (max_batch, 3, imgsz, imgsz))   # max
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)

    # Ultralytics expects: [4B metadata_len][metadata JSON][engine bytes]
    import json
    if names is None:
        names = {i: str(i) for i in range(nc)}
    metadata = {
        "description": "YOLOv8 QAT INT8",
        "stride": 32,
        "task": "detect",
        "batch": max_batch,
        "imgsz": [imgsz, imgsz],
        "names": names,
        "nc": nc,
    }
    meta_bytes = json.dumps(metadata).encode("utf-8")

    engine_path = Path(onnx_path).with_suffix(".engine")
    with open(engine_path, "wb") as f:
        f.write(len(meta_bytes).to_bytes(4, byteorder="little"))
        f.write(meta_bytes)
        f.write(engine_bytes)
    print(f"[OK] TensorRT engine saved: {engine_path}")
    return engine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export QAT YOLOv8 to TensorRT INT8 Engine")
    parser.add_argument("--model-config", type=str, default="yolov8s.yaml", help="YAML kiến trúc mô hình")
    parser.add_argument("--weight", type=str, required=True, help="Đường dẫn best.pt sau QAT")
    parser.add_argument("--nc", type=int, required=True, help="Số lượng class")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh input")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB)")
    parser.add_argument("--fp16", action="store_true", help="Bật thêm FP16 layer fallback")
    parser.add_argument("--max-batch", type=int, default=1, help="Max batch size cho dynamic engine (default=1)")
    parser.add_argument("--onnx-only", action="store_true", help="Chỉ export ONNX, không build engine")
    parser.add_argument("--use-trtexec", action="store_true", default=False,
                        help="Dùng trtexec CLI (chỉ dùng khi cài TensorRT từ .deb/tar). "
                             "Mặc định False → dùng Python TensorRT API (hoạt động với pip install tensorrt)")
    args = parser.parse_args()

    # Bước 1: Export sang ONNX
    onnx_path = export_onnx(
        cfg=args.model_config,
        weights=args.weight,
        nc=args.nc,
        imgsz=args.imgsz,
    )

    if args.onnx_only:
        print("[INFO] --onnx-only: dừng sau bước export ONNX.")
        sys.exit(0)

    # Bước 2: Build TensorRT Engine
    if args.use_trtexec:
        export_engine_trtexec(onnx_path, args.imgsz, args.workspace, args.fp16, args.max_batch)
    else:
        export_engine_python(onnx_path, args.imgsz, args.workspace, args.fp16,
                             nc=args.nc, names={i: str(i) for i in range(args.nc)},
                             max_batch=args.max_batch)
