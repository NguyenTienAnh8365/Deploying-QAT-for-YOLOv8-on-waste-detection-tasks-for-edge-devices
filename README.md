# Pruned + QAT YOLOv8 for Waste Detection on Edge Devices

Pipeline tối ưu hóa YOLOv8s cho bài toán phát hiện rác thải (**PAPER / PLASTIC / GLASS**) chạy thời gian thực trên **Jetson Orin Nano Super Developer Kit**. Kết hợp **Network Slimming (BN-gamma pruning, ratio 30%)** với **Quantization-Aware Training (QAT INT8)** và export sang **TensorRT engine** qua QDQ ONNX.

> Mục tiêu: giảm latency / tăng FPS trên edge device mà vẫn giữ được mAP gần với mô hình FP32 gốc.
>
> Toàn bộ pipeline reproducible qua notebook [`run_pruning_qat.ipynb`](run_pruning_qat.ipynb) — chạy được end-to-end trên Colab hoặc workstation có CUDA.

---

## Kết quả

Benchmark trên **NVIDIA RTX PRO 6000 Blackwell Server Edition** (dùng để đối chứng pipeline — số liệu thực tế trên Jetson Orin Nano sẽ khác, nhưng **tỉ lệ speedup và Δ mAP sẽ tương đương**):

| Model                                         |   mAP50   | mAP50-95 | Latency (ms) |   FPS    | Speedup (vs FP32 baseline) |
|-----------------------------------------------|----------:|---------:|-------------:|---------:|---------------------------:|
| YOLOv8s FP32 — original (phase_C baseline)    | 0.8901    | 0.7837   | 3.21         |  311.29  | 1.00x                      |
| YOLOv8s FP32 — pruned + finetune              | **0.8878**| 0.7826   | 3.53         |  283.33  | 0.91x                      |
| **TensorRT INT8 — pruned + QAT**              | **0.8724**| 0.7661   | **1.82**     | **550.74** | **1.77x**                |

- Input: `1×3×640×640`, batch = 1, imgsz = 640.
- Δ mAP50: **−1.73%** so với baseline FP32 original, **−1.54%** so với FP32 pruned+finetune.
- Engine: `--int8 --fp16`, workspace 2 GB, metadata JSON embedded.
- Engine size: **59 MB** (so với ~22 MB FP32 original `.pt`).
- Training set: **16 883 ảnh**, calibration entropy method với 512 batches.
- Per-image test (80 ảnh test set), speedup trung bình: **1.24x** (các ảnh nhỏ bottleneck ở host preprocessing).

---

## Hardware đích: Jetson Orin Nano Super Developer Kit

| Thông số          | Giá trị                              |
|-------------------|---------------------------------------|
| GPU               | Ampere 1024 CUDA cores + 32 Tensor cores, **SM 8.7** |
| AI performance    | 67 TOPS (INT8)                        |
| RAM               | 8 GB LPDDR5                           |
| JetPack           | **6.1+** (L4T 36.x)                   |
| TensorRT          | 10.3                                  |
| CUDA              | 12.6                                  |

> ONNX là **framework/hardware-agnostic** — file `.tensorrt.onnx` export trên Colab có thể copy sang Jetson rồi build engine tại đó (engine phải build trên chính GPU đích vì gắn với kernel tuning + SM).

---

## Cấu trúc dự án

```
.
├── data.yaml                          # Dataset config (3 class: PAPER, PLASTIC, GLASS)
├── train-normal.py                    # Train YOLOv8s thường (baseline)
├── train-sparsity.py                  # Bước 1: Sparsity training (sr=1e-2)
├── prune.py                           # Bước 2: BN-gamma pruning (ratio=0.05)
├── finetune.py                        # Bước 3: Finetune pruned model
├── qat_pruned.py                      # Bước 4: QAT trên pruned model
├── qat_pruned_export.py               # Bước 5: Export QAT → TRT-compatible ONNX
├── build_engine.py                    # Bước 6: Build TensorRT engine + embed metadata
├── val.py                             # Đo mAP & latency mô hình PyTorch
├── vis-bn-weight.py                   # Visualize phân bố BN gamma (chọn ratio prune)
├── run_pruning_qat.ipynb              # Notebook chạy toàn pipeline
├── weights/
│   ├── original.pt                    # YOLOv8s đã train thường
│   └── pruned_05.pt                   # Pruned checkpoint (chứa maskbndict + model)
├── runs/
│   ├── train-sparsity2/weights/       # best.pt / last.pt sau sparsity training
│   ├── train-finetune/weights/        # best.pt / last.pt sau finetune
│   └── qat-pruned/train/weights/
│       ├── best.pt                    # QAT-trained (fake-quant)
│       ├── best.tensorrt.onnx         # QDQ ONNX cho TensorRT
│       └── best.engine                # TRT INT8 engine (đã embed ultralytics metadata)
└── ultralytics/                       # Ultralytics fork (thêm pruned + QAT modules)
    ├── nn/
    │   ├── modules/block_pruned.py    # C2fPruned, BottleneckPruned, SPPFPruned
    │   ├── modules/head_pruned.py     # DetectPruned
    │   └── tasks_pruned.py            # DetectionModelPruned + parse_model_pruned
    └── qat/nvidia_tensorrt/
        ├── qat_pruned_trainer.py      # QuantizationPrunedTrainer
        ├── qat_pruned_exporter.py     # export_onnx_pruned (+ sanitize _amax)
        └── quant_ops_pruned.py        # QuantC2fSplitPruned / QuantAdd / QuantConcat / QuantUpsample
```

---

## Pipeline 6 bước

### Bước 1 — Sparsity training

Train với hệ số regularization `sr` để đẩy nhiều BN γ về gần 0, phân hóa kênh quan trọng / không quan trọng.

```bash
python train-sparsity.py
# → runs/train-sparsity2/weights/last.pt
```

Tham số thực tế dùng trong notebook `run_pruning_qat.ipynb`:

```python
model.train(
    epochs=80,
    batch=96,
    sr=1.5e-2,            # sparsity regularization coefficient
    lr0=1e-2,
    warmup_epochs=3.0,
    data="data.yaml",
    project=".",
    name="runs/train-sparsity2",
)
```

Thời gian train thực tế: **1h29' / 80 epoch** trên RTX PRO 6000 Blackwell.

### Bước 2 — Pruning (BN-gamma channel pruning)

Cắt tỉa channels có `|γ|` nhỏ hơn ngưỡng **(ratio = 0.3, tức loại bỏ 30% kênh ít quan trọng nhất)**, xây lại network bằng `DetectionModelPruned` với `maskbndict`.

```bash
python prune.py \
    --weights runs/train-sparsity2/weights/last.pt \
    --cfg ultralytics/cfg/models/v8/yolov8.yaml \
    --model-size s \
    --prune-ratio 0.3 \
    --save-dir weights
# → weights/pruned_05.pt  (chứa {'model': DetectionModelPruned, 'maskbndict': {...}})
```

> Lưu ý: file output luôn được `prune.py` lưu dưới tên `pruned_05.pt` (hardcoded trong `step10` của script), **không** phản ánh `prune-ratio`. Có thể rename sau hoặc sửa script.

Ví dụ output channel sau pruning (yolov8s, ratio 0.3):

| Layer              | Channels gốc | Sau pruning |
|--------------------|-------------:|------------:|
| Backbone C2f #2    | 64           | 58          |
| Backbone C2f #4    | 128          | 118         |
| Backbone C2f #6    | 256          | 207         |
| Backbone C2f #8    | 512          | 289         |
| SPPF #9            | 512          | 297         |
| Head C2f #12       | 256          | 178         |
| Head C2f #15       | 128          | 84          |
| Head C2f #18       | 256          | 140         |
| Head C2f #21       | 512          | 322         |

Có thể dùng `vis-bn-weight.py` để xem histogram BN γ trước khi chọn `--prune-ratio`.

### Bước 3 — Finetune pruned model

Khôi phục mAP sau pruning. Tham số thực tế dùng trong notebook:

```python
model.train(
    epochs=100,
    batch=96,
    sr=0.0,
    finetune=True,
    lr0=1e-3,
    data="data.yaml",
    project=".",
    name="runs/train-finetune",
)
# → runs/train-finetune/weights/best.pt
```

Sau finetune, mAP50 hồi phục về **0.8878** (so với 0.8901 của mô hình FP32 gốc) — drop chỉ ~0.26% với 30% channels cắt đi.

### Bước 4 — QAT (Quantization-Aware Training)

Thêm fake-quant nodes vào pruned model và tiếp tục train ngắn (~10 epoch) với LR rất nhỏ (lr0 thực = `args.lr0 / 100`).

Kiến trúc đã được chuẩn bị cho QAT:
- `Conv2d` → `QuantConv2d` (qua `quant_modules.initialize()`)
- `C2fPruned.forward` được monkey-patch để chèn `QuantC2fSplitPruned` trước split (sections bất đối xứng).
- `BottleneckPruned` có residual → chèn `QuantAdd`.
- `Concat` / `Upsample` → `QuantConcat` / `QuantUpsample`.
- **Detection head giữ FP32** (Technique 3): head quantizers bị disable bởi `_skip_detect_quantizers` để tránh sụt mAP ở regression/classification branch.

Lệnh thực tế dùng trong notebook:

```bash
python qat_pruned.py \
    --pruned-checkpoint weights/pruned_05.pt \
    --pretrained-weight runs/train-finetune/weights/best.pt \
    --data-config data.yaml \
    --epochs 10 --imgsz 640 --batch 16 --device 0 --workers 12 --cache ram \
    --lr0 1e-3 --freeze 10 --optimizer AdamW --patience 5 --recalib-every 1 \
    --project runs/qat-pruned --name train --exist-ok --plots
# → runs/qat-pruned/train/weights/best.pt
```

Trainer tự động **chia `lr0 / 100`**, tức `lr0=1e-3` thực tế chạy ở `1e-5`. Calibration dùng **entropy method** trên ~512 batches train (~16k ảnh). Thời gian QAT thực tế: **~12 phút / 10 epoch** trên RTX PRO 6000 Blackwell.

### Bước 5 — Export TensorRT-compatible ONNX

Load pruned checkpoint, build lại quantized model, nạp QAT weights, disable head quantizers (khớp với training), sanitize `_amax` (dự phòng cho body quantizers), export với `opset=13`, `dynamo=False` (Torch 2.10 default dynamo exporter không trace được `.item()` trong `_fb_fake_quant`).

```bash
# Cài onnxscript trước (dependency của torch.onnx.export ở opset 13)
pip install onnxscript -q

python qat_pruned_export.py \
    --pruned-checkpoint weights/pruned_05.pt \
    --weight runs/qat-pruned/train/weights/best.pt
# → runs/qat-pruned/train/weights/best.tensorrt.onnx
```

ONNX output: `input_names=["images"]`, `output_names=["output0"]`, chứa `QuantizeLinear`/`DequantizeLinear` nodes ở body (head giữ FP32).

### Bước 6 — Build TensorRT engine

Script Python dùng trực tiếp TRT API (không cần `trtexec` binary).

```bash
# Trên máy train (Colab / Workstation):
pip install tensorrt
python build_engine.py \
    --onnx runs/qat-pruned/train/weights/best.tensorrt.onnx \
    --engine runs/qat-pruned/train/weights/best.engine \
    --data-yaml data.yaml \
    --workspace-gb 2 \
    --imgsz 640 --batch 1

# Trên Jetson Orin Nano (dùng trtexec có sẵn trong JetPack):
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.tensorrt.onnx \
    --saveEngine=best.engine \
    --int8 --fp16 \
    --memPoolSize=workspace:2048M
```

Output thực tế: `best.engine` ≈ **59 MB** (224 B ultralytics metadata + raw engine).

Engine file được embed **ultralytics metadata** theo format chuẩn (4 bytes little-endian length + JSON header + raw engine bytes):

```json
{
  "description": "YOLOv8 pruned + QAT INT8",
  "task": "detect",
  "batch": 1,
  "imgsz": [640, 640],
  "names": {"0": "PAPER", "1": "PLASTIC", "2": "GLASS"},
  "stride": 32,
  "args": {"batch": 1, "imgsz": 640}
}
```

Metadata này giúp `ultralytics.YOLO("best.engine")` load engine đúng cách và predict / val ra nhãn class người đọc được.

---

## Đánh giá & benchmark

### Đo mAP + latency mô hình PyTorch

```bash
python val.py   # chỉnh `weight = "weights/..."` trong file
```

### So sánh PyTorch .pt ↔ TRT .engine

Script benchmark trong notebook (cell cuối):

```python
from ultralytics import YOLO
import torch, time, numpy as np

IMGSZ, DEVICE = 640, 0

def benchmark(path, label):
    print(f'\n--- Testing {label} ---')
    model = YOLO(path)
    # Ultralytics validator auto-fuses BN+Conv ở .pt → phá QAT pattern. Disable:
    if path.endswith('.pt'):
        model.model.fuse = lambda verbose=True: model.model
    r = model.val(data='data.yaml', imgsz=IMGSZ, batch=1, device=DEVICE,
                  plots=False, verbose=False)
    for _ in range(10):
        model.predict(torch.zeros((1, 3, IMGSZ, IMGSZ)), device=DEVICE, verbose=False)
    lat = []
    for _ in range(100):
        s = time.perf_counter()
        model.predict(torch.zeros((1, 3, IMGSZ, IMGSZ)), device=DEVICE, verbose=False)
        lat.append((time.perf_counter() - s) * 1000)
    print(f'mAP50={r.box.map50:.4f}, mAP50-95={r.box.map:.4f}, '
          f'lat={np.mean(lat):.2f} ms, FPS={1000/np.mean(lat):.2f}')
    return {'mAP50': r.box.map50, 'latency': np.mean(lat)}

stats_orig = benchmark('weights/original.pt',                  'Original FP32')
stats_pt   = benchmark('runs/train-finetune/weights/best.pt',  'Pruned + finetune FP32')
stats_trt  = benchmark('runs/qat-pruned/train/weights/best.engine', 'TensorRT INT8 Engine')

print(f'\nSpeedup: TRT INT8 is {stats_orig["latency"] / stats_trt["latency"]:.2f}x '
      f'faster than Original FP32')
```

---

## Triển khai trên Jetson Orin Nano

```bash
# 1. Flash JetPack 6.1+
# 2. Cài ultralytics (phiên bản compatible — tránh code dynamo của Torch 2.10)
pip install ultralytics==8.3.x

# 3. Copy ONNX từ máy train sang Jetson (không cần copy engine — sẽ build lại)
scp best.tensorrt.onnx jetson@<ip>:~/

# 4. Build engine trên Jetson (gắn với SM 8.7)
python build_engine.py \
    --onnx best.tensorrt.onnx \
    --engine best.engine \
    --data-yaml data.yaml

# 5. Inference
from ultralytics import YOLO
model = YOLO('best.engine')
results = model.predict('image.jpg', imgsz=640)
```

> Engine TRT **không portable** giữa các GPU khác SM — phải build trên GPU đích. ONNX thì portable.

---

## Các chi tiết kỹ thuật đáng lưu ý

- **Head giữ FP32**: `_skip_detect_quantizers` disable quantizers trong `DetectPruned`. Nếu quantize head → `_amax` ở các branch regression có thể NaN / ≤ 0 → TRT reject (`Scale coefficients must all be positive`). Đã áp dụng ở cả trainer lẫn exporter.
- **`_amax` sanitize**: `_sanitize_quantizer_amax` quét toàn bộ `TensorQuantizer`, thay mọi entry NaN / Inf / ≤ 0 bằng `1.0` để engine build không lỗi — fallback dự phòng sau khi đã disable head.
- **Split sections bất đối xứng**: Sau pruning, `C2fPruned` có 2 nhánh channels không đều (vd `[80, 40]`), nên dùng `torch.split(sections)` thay vì `torch.chunk(2)` → `QuantC2fSplitPruned` nhận `split_sections` từ module pruned.
- **Validator fuse trap**: `model.val()` của Ultralytics tự fuse BN vào Conv — điều này **phá quantization pattern** của QAT checkpoint (`.pt`). Workaround: monkey-patch `model.model.fuse = lambda v=True: model.model`. Không ảnh hưởng engine (đã static).
- **Input name phải là `"images"`** (số nhiều) để `ultralytics.YOLO('*.engine')` autobackend nhận diện.
- **TRT 10.x API**: `build_serialized_network` trả `IHostMemory` không có `len()` — phải `bytes(serialized)` trước khi lấy size.
- **Torch 2.10 + pytorch-quantization**: bắt buộc `dynamo=False` khi export ONNX, vì `torch.export` không trace được `.item()` trong `_fb_fake_quant` (data-dependent symbolic).

---

## Dataset

File `data.yaml`:

```yaml
train: /content/dataset/train/images
val:   /content/dataset/val/images
test:  /content/dataset/test/images
nc: 3
names:
  - PAPER
  - PLASTIC
  - GLASS
```

Thay đường dẫn phù hợp với môi trường thực (Colab / Jetson / local).

---

## Requirements

Môi trường đã test (Colab / Workstation, khớp với `run_pruning_qat.ipynb`):

| Package               | Version                          |
|-----------------------|-----------------------------------|
| Python                | 3.12.13                           |
| PyTorch               | 2.10.0+cu128                      |
| Ultralytics (fork)    | 8.3.231 (đã patch pruned + QAT)   |
| pytorch-quantization  | **2.1.2** (pin cứng)              |
| numpy                 | **< 2.0** (pin cứng — tương thích pytorch-quantization) |
| onnxscript            | latest (cần cho export opset 13)  |
| TensorRT              | ≥ 10.0 (Colab), **10.3** (Jetson JetPack 6.1) |
| CUDA                  | 12.6 / 12.8                       |

Cài từ đầu (Colab):

```bash
# Ultralytics fork (editable — có modules pruned + QAT)
pip install -e .

# NVIDIA pytorch-quantization (từ pypi.nvidia.com)
pip install nvidia-pyindex -q
pip install --no-cache-dir pytorch-quantization==2.1.2 \
    --extra-index-url https://pypi.nvidia.com -q
pip install "numpy<2.0" -q

# TensorRT + ONNX tooling
pip install tensorrt onnxscript -q
```

Trên Jetson Orin Nano thì TensorRT + CUDA có sẵn trong JetPack — chỉ cần cài thêm `ultralytics` và (tùy chọn) `pytorch-quantization` nếu muốn tái export.

---

## Tham khảo

- YOLOv8 — Ultralytics: https://github.com/ultralytics/ultralytics
- Network Slimming (Liu et al., 2017): https://arxiv.org/abs/1708.06519
- NVIDIA pytorch-quantization: https://github.com/NVIDIA/TensorRT/tree/release/10.0/tools/pytorch-quantization
- TensorRT QDQ workflow: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html

## License

AGPL-3.0 (kế thừa từ Ultralytics).
