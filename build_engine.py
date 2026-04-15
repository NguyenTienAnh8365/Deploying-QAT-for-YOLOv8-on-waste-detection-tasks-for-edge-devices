"""
Build TensorRT engine từ ONNX file (không cần trtexec binary).
Dùng cho QAT model với INT8 + FP16.

Cách chạy trên Colab:
  !pip install tensorrt
  !python build_engine.py \\
      --onnx runs/qat-pruned/train/weights/best.tensorrt.onnx \\
      --engine model.engine

Trên Jetson Orin Nano (SM 8.7) thì TRT đã cài sẵn trong JetPack,
chạy script tương tự hoặc dùng trực tiếp:
  /usr/src/tensorrt/bin/trtexec --onnx=...onnx --saveEngine=...engine --int8 --fp16
"""
import argparse
import json
import tensorrt as trt


def build_engine(onnx_path, engine_path, workspace_gb=2, fp16=True, int8=True,
                 imgsz=640, batch=1, names=None, task="detect", stride=32):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ONNX parse error] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX — xem lỗi phía trên")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")

    # QAT ONNX đã chứa QuantizeLinear/DequantizeLinear nodes, chỉ cần bật INT8 flag
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("[build] INT8 enabled (scales từ QAT QDQ nodes)")

    print(f"[build] TensorRT {trt.__version__} — đang build engine, có thể mất vài phút...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — xem log TRT phía trên")

    data = bytes(serialized)

    # Ultralytics-compatible metadata: 4 bytes little-endian signed length + JSON + raw engine
    meta = {
        "description": "YOLOv8 pruned + QAT INT8",
        "author": "ultralytics",
        "task": task,
        "batch": batch,
        "imgsz": [imgsz, imgsz],
        "names": names if names is not None else {i: str(i) for i in range(80)},
        "stride": stride,
        "args": {"batch": batch, "imgsz": imgsz},
    }
    meta_bytes = json.dumps(meta).encode()

    with open(engine_path, "wb") as f:
        f.write(len(meta_bytes).to_bytes(4, byteorder="little", signed=True))
        f.write(meta_bytes)
        f.write(data)
    print(f"[build] Saved engine → {engine_path} "
          f"({(len(data) + len(meta_bytes) + 4)/1e6:.1f} MB, {len(meta_bytes)} B metadata)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-gb", type=int, default=2)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--nc", type=int, default=3, help="Number of classes (for names metadata)")
    ap.add_argument("--names", type=str, default=None,
                    help='JSON dict of class names, e.g. \'{"0":"car","1":"person"}\'. '
                         "Overrides --nc.")
    ap.add_argument("--data-yaml", type=str, default=None,
                    help="Path to data.yaml — đọc names trực tiếp từ đây. "
                         "Override cả --nc và --names.")
    ap.add_argument("--task", type=str, default="detect")
    ap.add_argument("--stride", type=int, default=32)
    args = ap.parse_args()

    if args.data_yaml:
        import yaml
        with open(args.data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        raw_names = data_cfg["names"]
        if isinstance(raw_names, dict):
            names = {int(k): v for k, v in raw_names.items()}
        else:
            names = {i: n for i, n in enumerate(raw_names)}
        print(f"[build] Loaded {len(names)} class names from {args.data_yaml}: {names}")
    elif args.names:
        names = {int(k): v for k, v in json.loads(args.names).items()}
    else:
        names = {i: str(i) for i in range(args.nc)}

    build_engine(args.onnx, args.engine, args.workspace_gb,
                 fp16=not args.no_fp16, int8=not args.no_int8,
                 imgsz=args.imgsz, batch=args.batch, names=names,
                 task=args.task, stride=args.stride)
