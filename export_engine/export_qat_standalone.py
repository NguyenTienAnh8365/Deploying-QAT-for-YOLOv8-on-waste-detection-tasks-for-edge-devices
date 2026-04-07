"""
Export QAT YOLOv8 .pt → ONNX → TensorRT INT8 Engine
Standalone script — không cần source code repo gốc.

Yêu cầu:
    pip install "ultralytics==8.0.228" onnx onnxscript
    pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
    pip install tensorrt

Usage:
    python export_qat_standalone.py \\
        --weight best.pt \\
        --nc 3 \\
        --names plastic,metal,glass \\
        --imgsz 640 \\
        --workspace 4

Lưu ý: Engine build trên GPU nào thì chỉ chạy được trên GPU đó (hoặc cùng kiến trúc).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.nn.modules import _utils
from pytorch_quantization.tensor_quant import QuantDescriptor

from ultralytics.nn.modules import Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel


# ─── Quant ops (inline — không cần repo gốc) ─────────────────────────────────

class QuantAdd(torch.nn.Module, _utils.QuantMixin):
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantC2fChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c

    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)


class QuantConcat(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        return torch.cat((self._input0_quantizer(x[0]), self._input1_quantizer(x[1])), self.dim)


class QuantUpsample(torch.nn.Module):
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())

    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)


def _bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def _concat_quant_forward(self, x):
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)

def _upsample_quant_forward(self, x):
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x, self.size, self.scale_factor, self.mode)

def _c2f_quant_forward(self, x):
    if hasattr(self, "c2fchunkop"):
        y = list(self.c2fchunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def _register_qat_namespace():
    """Đăng ký các class QAT vào đúng namespace ultralytics.qat để torch.load hoạt động.

    best.pt được save với class path ultralytics.qat.nvidia_tensorrt.quant_ops.QuantAdd...
    Khi load trên môi trường mới (không có custom fork), cần map class đó về class local này.
    """
    import sys
    import types

    # Tạo module stubs với __path__ để Python nhận là package (cho phép import submodule)
    for mod_name in [
        'ultralytics.qat',
        'ultralytics.qat.nvidia_tensorrt',
        'ultralytics.qat.nvidia_tensorrt.quant_ops',
        'ultralytics.qat.nvidia_tensorrt.qat_nvidia_trainer',
    ]:
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = []   # đánh dấu là package
            mod.__package__ = mod_name
            sys.modules[mod_name] = mod

    # Gán class local vào đúng namespace
    qat_ops = sys.modules['ultralytics.qat.nvidia_tensorrt.quant_ops']
    qat_ops.QuantAdd       = QuantAdd
    qat_ops.QuantC2fChunk  = QuantC2fChunk
    qat_ops.QuantConcat    = QuantConcat
    qat_ops.QuantUpsample  = QuantUpsample


def quant_module_change(model):
    """Patch model modules để thêm quantizers vào Add/Concat/Upsample/C2f."""
    for name, module in model.named_modules():
        if module.__class__.__name__ == "C2f":
            if not hasattr(module, "c2fchunkop"):
                print(f"  [quant] Add QuantC2fChunk → {name}")
                module.c2fchunkop = QuantC2fChunk(module.c)
            module.__class__.forward = _c2f_quant_forward

        if module.__class__.__name__ == "Bottleneck" and module.add:
            if not hasattr(module, "addop"):
                print(f"  [quant] Add QuantAdd → {name}")
                module.addop = QuantAdd(module.add)
            module.__class__.forward = _bottleneck_quant_forward

        if module.__class__.__name__ == "Concat":
            if not hasattr(module, "concatop"):
                print(f"  [quant] Add QuantConcat → {name}")
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = _concat_quant_forward

        if module.__class__.__name__ == "Upsample":
            if not hasattr(module, "upsampleop"):
                print(f"  [quant] Add QuantUpsample → {name}")
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = _upsample_quant_forward


# ─── Export ONNX ──────────────────────────────────────────────────────────────

def export_onnx(weights, nc, imgsz=640, model_config="yolov8s.yaml"):
    """Load QAT .pt và export sang ONNX với Q/DQ nodes."""
    print("\n[STEP 1] Export ONNX với Q/DQ nodes...")

    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()

    # Đăng ký QAT class namespace trước khi load checkpoint
    _register_qat_namespace()

    # Build model architecture rồi patch quantizers
    model = DetectionModel(cfg=model_config, ch=3, nc=nc, verbose=False)
    quant_module_change(model)

    # Load weights từ checkpoint
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and ('model' in ckpt or 'ema' in ckpt):
        src = ckpt.get('ema') or ckpt['model']
        state_dict = src.float().state_dict()
        label = 'ema' if ckpt.get('ema') is not None else 'model'
        print(f"  Loaded checkpoint [{label}] từ {weights}")
    else:
        state_dict = ckpt
        print(f"  Loaded state dict từ {weights}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cpu()  # _amax buffers cần về CPU trước khi trace

    # Skip Detect head quantizers → giữ FP32 ở output layer
    detect_idx = str(len(model.model) - 1)
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            parts = name.split('.')
            if len(parts) >= 2 and parts[0] == 'model' and parts[1] == detect_idx:
                module.disable()
                skipped += 1
    print(f"  Skipped {skipped} quantizers trong Detect head (model.{detect_idx}) → FP32")

    # Set export mode
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
        dynamo=False,           # BẮT BUỘC False — PyTorch 2.10+ dynamo=True không hỗ trợ Q/DQ
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )
    print(f"  [OK] ONNX saved: {onnx_path}")
    return onnx_path


# ─── Build TensorRT Engine ────────────────────────────────────────────────────

def build_engine(onnx_path, imgsz=640, workspace=4, fp16=False,
                 nc=80, names=None, max_batch=1):
    """Build TensorRT INT8 engine từ ONNX có Q/DQ nodes."""
    print("\n[STEP 2] Build TensorRT INT8 engine...")
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] Chưa cài tensorrt: pip install tensorrt")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [ONNX error] {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)   # Q/DQ nodes cung cấp scale, không cần calibration thêm
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic batch: min=1, opt=max_batch, max=max_batch
    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      (1, 3, imgsz, imgsz),           # min
                      (max_batch, 3, imgsz, imgsz),   # opt
                      (max_batch, 3, imgsz, imgsz))   # max
    config.add_optimization_profile(profile)

    print(f"  Building engine (workspace={workspace}GB, INT8, max_batch={max_batch})...")
    print("  Có thể mất 2-10 phút tùy GPU...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("[ERROR] Build engine thất bại.")
        sys.exit(1)

    # Prepend Ultralytics metadata header: [4B len][JSON][engine bytes]
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

    size_mb = engine_path.stat().st_size / (1024 ** 2)
    print(f"  [OK] Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export QAT YOLOv8 .pt → TensorRT INT8 Engine (standalone)"
    )
    parser.add_argument("--weight",       required=True,            help="Đường dẫn best.pt sau QAT")
    parser.add_argument("--nc",           type=int, required=True,  help="Số lượng class")
    parser.add_argument("--names",        type=str, default=None,
                        help="Tên class cách nhau bởi dấu phẩy. VD: plastic,metal,glass")
    parser.add_argument("--model-config", default="yolov8s.yaml",   help="YAML kiến trúc (default: yolov8s.yaml)")
    parser.add_argument("--imgsz",        type=int, default=640,    help="Kích thước ảnh (default: 640)")
    parser.add_argument("--workspace",    type=int, default=4,      help="TRT workspace GB (default: 4)")
    parser.add_argument("--fp16",         action="store_true",      help="Bật FP16 fallback layer")
    parser.add_argument("--max-batch",    type=int, default=1,      help="Max batch size (default: 1)")
    parser.add_argument("--onnx-only",    action="store_true",      help="Chỉ export ONNX, không build engine")
    args = parser.parse_args()

    # Parse names
    names_dict = None
    if args.names:
        names_list = [n.strip() for n in args.names.split(',')]
        if len(names_list) != args.nc:
            print(f"[ERROR] --names cần đúng {args.nc} phần tử, nhận được {len(names_list)}")
            sys.exit(1)
        names_dict = {i: n for i, n in enumerate(names_list)}

    print("=" * 60)
    print("QAT YOLOv8 Export: .pt → ONNX → TensorRT INT8 Engine")
    print("=" * 60)
    print(f"  weight     : {args.weight}")
    print(f"  nc         : {args.nc}")
    print(f"  names      : {names_dict}")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  workspace  : {args.workspace} GB")
    print(f"  max_batch  : {args.max_batch}")
    print(f"  fp16       : {args.fp16}")

    onnx_path = export_onnx(
        weights=args.weight,
        nc=args.nc,
        imgsz=args.imgsz,
        model_config=args.model_config,
    )

    if not args.onnx_only:
        build_engine(
            onnx_path=onnx_path,
            imgsz=args.imgsz,
            workspace=args.workspace,
            fp16=args.fp16,
            nc=args.nc,
            names=names_dict,
            max_batch=args.max_batch,
        )

    print("\n[DONE] Hoàn tất!")
    print("Tiếp theo: chạy patch_autobackend.py để fix TensorRT 10.x API, sau đó dùng YOLO để inference.")
