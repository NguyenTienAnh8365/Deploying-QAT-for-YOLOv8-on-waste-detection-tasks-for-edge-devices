"""
Export pruned + quantized YOLOv8 to TensorRT-compatible ONNX.

Ví dụ:
  python qat_pruned_export.py \\
      --pruned-checkpoint weights/pruned.pt \\
      --weight runs/qat-pruned/train/weights/best.pt

Sau đó convert ONNX → TensorRT engine:
  trtexec --onnx=best.tensorrt.onnx --saveEngine=model.engine --int8
"""

import argparse
from ultralytics.qat.nvidia_tensorrt.qat_pruned_exporter import export_onnx_pruned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pruned + quantized YOLOv8 to TensorRT ONNX"
    )
    parser.add_argument("--pruned-checkpoint", type=str, required=True,
                        help="Path tới pruned.pt gốc (output của prune.py, chứa maskbndict)")
    parser.add_argument("--weight", type=str, required=True,
                        help="Path tới QAT-trained weight (best.pt từ QAT training)")
    args = parser.parse_args()

    model = export_onnx_pruned(
        pruned_checkpoint=args.pruned_checkpoint,
        qat_weight=args.weight,
    )
