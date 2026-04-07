# Deploying-Quantization-Aware-Training-QAT-for-YOLOv8-on-waste-detection-tasks-for-edge-devices.
This work focuses on applying Quantization-Aware Training (QAT) to the YOLOv8 model for waste detection tasks. The proposed pipeline enables efficient deployment on edge devices by reducing model precision to INT8 while maintaining high detection accuracy.

# YOLOv8 QAT (NVIDIA TensorRT Focus)

This repository is a YOLOv8 fork for Quantization-Aware Training (QAT), with an INT8 deployment path for TensorRT.

This README documents QAT only.
Pruning and distillation are intentionally out of scope.

## 1) Scope

- QAT training with NVIDIA `pytorch_quantization`
- Export QAT checkpoint to ONNX with Q/DQ nodes
- Build TensorRT INT8 engine from exported ONNX

Main scripts:
- `qat_nvidia.py`: QAT training entrypoint
- `qat_export.py`: lightweight ONNX export entrypoint (`--mode nvidia`)
- `export_engine.py`: end-to-end export (`.pt -> .onnx -> .engine`)

## 2) Requirements

- Python `>=3.10`
- CUDA GPU environment
- PyTorch + torchvision compatible with your CUDA
- Base deps:

```bash
# PyTorch (CUDA 12.8)
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Ultralytics và ONNX
pip install "ultralytics==8.0.228" onnx onnxsim onnxscript

# NVIDIA pytorch-quantization (bắt buộc đúng version để load QAT weights)
pip install --upgrade setuptools wheel
pip install --no-cache-dir pytorch-quantization==2.1.2 --extra-index-url https://pypi.ngc.nvidia.com

# Downgrade numpy (pytorch-quantization yêu cầu numpy < 2.0)
pip install "numpy<2.0"

# TensorRT
pip install tensorrt==10.0.1 --extra-index-url https://pypi.nvidia.com
```

If you use `trtexec` instead of Python API, install TensorRT system package and ensure `trtexec` is in `PATH`.

## 3) Dataset Format

Use standard Ultralytics YOLO dataset YAML, for example:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class0
  1: class1
```

## 4) QAT Training (NVIDIA)

Minimal command:

```bash
python qat_nvidia.py \
  --model-config yolov8s.yaml \
  --pretrained-weight /path/to/fp32_best.pt \
  --data-config /path/to/data.yaml \
  --epochs 10 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Recommended QAT-style settings (already defaulted in script):
- Keep strong augmentations disabled (`mosaic=0`, `mixup=0`, etc.)
- Keep AMP disabled (trainer enforces this)
- Recalibrate quantizers every epoch (`--recalib-every 1`)

Important learning-rate note:
- Inside trainer, `lr0` is divided by `100`.
- Effective LR = `--lr0 / 100`.
- Example: target LR `1e-5` -> pass `--lr0 1e-3`.

Outputs are saved under:
- `runs/qat/<name>/weights/last.pt`
- `runs/qat/<name>/weights/best.pt`

## 5) Export

### A. ONNX only (Q/DQ graph)

```bash
python qat_export.py \
  --mode nvidia \
  --model-config yolov8s.yaml \
  --weight runs/qat/train/weights/best.pt \
  --nc 3
```

### B. ONNX + TensorRT INT8 engine (recommended)

```bash
python export_engine.py \
  --model-config yolov8s.yaml \
  --weight runs/qat/train/weights/best.pt \
  --nc 3 \
  --names plastic,metal,glass \
  --imgsz 640 \
  --workspace 4 \
  --max-batch 1
```

Useful flags:
- `--onnx-only`: stop after ONNX export
- `--fp16`: allow FP16 fallback for unsupported layers
- `--use-trtexec`: build engine with TensorRT CLI instead of Python API

Generated files:
- `best.tensorrt.onnx`
- `best.tensorrt.engine`

## 6) QAT Implementation Notes

This QAT trainer includes practical stability/accuracy adjustments:
- Configurable calibration method (max, entropy, percentile)
- Optional selective quantization (skip sensitive layers)
- BatchNorm freeze during QAT fine-tuning
- Safe handling for quantizers without calibrator
- Optional periodic re-calibration

These behaviors are implemented in:
- `ultralytics/qat/nvidia_tensorrt/qat_nvidia_trainer.py`
- `ultralytics/qat/nvidia_tensorrt/quant_ops.py`
- `ultralytics/qat/nvidia_tensorrt/qat_exporter.py`

## 7) References

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018
- Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference," arXiv 2018
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," arXiv 2013
- NVIDIA pytorch-quantization: https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization
- NVIDIA TensorRT: https://github.com/NVIDIA/TensorRT
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- PyTorch Quantization Tutorial: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
