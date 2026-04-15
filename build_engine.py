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
import tensorrt as trt


def build_engine(onnx_path, engine_path, workspace_gb=2, fp16=True, int8=True):
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

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] Saved engine → {engine_path} ({len(serialized)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-gb", type=int, default=2)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()
    build_engine(args.onnx, args.engine, args.workspace_gb,
                 fp16=not args.no_fp16, int8=not args.no_int8)
