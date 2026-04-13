import argparse
from ultralytics.qat.nvidia_tensorrt.qat_exporter import export_onnx_nvidia
from ultralytics.qat.pytorch_native.qat_pytorch_exporter import export_onnx_native


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default = "native", choices=['native', 'nvidia'])
    parser.add_argument("--model-config", type = str, default = "yolov8s.yaml")
    parser.add_argument("--weight", type = str, default = "yolov8s.pt") # Your quantized weight
    parser.add_argument("--nc", type = int, default = 80)
    args = parser.parse_args()


    exporter = export_onnx_native if args.mode == 'native' else export_onnx_nvidia

    model = exporter(args.model_config,
                    args.weight,
                    args.nc)