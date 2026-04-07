# Hướng dẫn Export QAT YOLOv8 → TensorRT INT8 Engine

> **Lưu ý quan trọng:** TensorRT engine được tối ưu riêng cho từng GPU.
> File `best.pt` có thể chuyển giữa các máy, nhưng engine phải **build lại trên GPU đích**.

---

## Cấu trúc thư mục

```
export_engine/
├── check_point_qat/
│   ├── best.pt                   ← Weights YOLOv8s sau QAT (portable)
│   └── best.tensorrt.engine      ← Engine đã build sẵn (chỉ dùng trên A100)
├── check_point_yolov8s/
│   └── phase_C.pt                ← Weights FP32 baseline (trước QAT)
├── export_qat_standalone.py      ← Script export (tự chứa, không cần repo gốc)
├── patch_autobackend.py          ← Patch ultralytics cho TensorRT 10.x
└── readme.md
```

---

## Yêu cầu hệ thống

- Python 3.10+
- CUDA 12.x + cuDNN tương ứng
- GPU NVIDIA (CUDA-capable, compute capability ≥ 7.0)

---

## Bước 1 — Cài đặt

```bash
# PyTorch (CUDA 12.8 — đúng theo môi trường đã test)
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

---

## Bước 2 — Export Engine

Script sẽ tự động thực hiện 2 bước: export ONNX → build TensorRT engine.

```bash
python export_qat_standalone.py \
    --weight check_point_qat/best.pt \
    --nc 3 \
    --names plastic,metal,glass \
    --imgsz 640 \
    --workspace 4
```

### Tham số

| Tham số | Kiểu | Mô tả | Mặc định |
|---------|------|-------|----------|
| `--weight` | str | Đường dẫn file `best.pt` sau QAT | *(bắt buộc)* |
| `--nc` | int | Số lượng class | *(bắt buộc)* |
| `--names` | str | Tên class cách nhau bởi dấu phẩy. VD: `plastic,metal,glass` | Tự đánh số `0,1,2...` |
| `--model-config` | str | YAML kiến trúc YOLOv8 | `yolov8s.yaml` |
| `--imgsz` | int | Kích thước ảnh vuông (pixel) | `640` |
| `--workspace` | int | TensorRT workspace (GB) | `4` |
| `--max-batch` | int | Max batch size cho dynamic engine | `1` |
| `--fp16` | flag | Bật thêm FP16 fallback cho các layer không hỗ trợ INT8 | `False` |
| `--onnx-only` | flag | Chỉ export ONNX, không build engine | `False` |

### Output

Output được tạo cùng thư mục với file `--weight`:

```
check_point_qat/best.tensorrt.onnx    ← ONNX với Q/DQ nodes (bước trung gian)
check_point_qat/best.tensorrt.engine  ← TensorRT INT8 engine (dùng để deploy)
```

> Build engine thường mất **2–10 phút** tùy GPU và workspace.

### Ví dụ nâng cao

```bash
# Export ONNX trước, kiểm tra rồi mới build engine
python export_qat_standalone.py --weight best.pt --nc 3 --names plastic,metal,glass --onnx-only

# Build engine với dynamic batch=8 và FP16 fallback
python export_qat_standalone.py --weight best.pt --nc 3 --names plastic,metal,glass --max-batch 8 --fp16
```

---

## Bước 3 — Patch ultralytics cho TensorRT 10.x

TensorRT 10.x đã xóa hoàn toàn API cũ (`num_bindings`, `get_binding_name`...).
Script này tự tìm file `autobackend.py` trong gói ultralytics đã cài và vá tự động.

```bash
python patch_autobackend.py
```

Script sẽ:
1. Tự động tìm `ultralytics/nn/autobackend.py`
2. Tạo backup tại `autobackend.py.bak`
3. Áp dụng patch TRT 10.x
4. Kiểm tra kết quả

**Output khi thành công:**
```
[OK] Patch thành công: .../ultralytics/nn/autobackend.py
[Kiểm tra patch]
  [✓] version-based API detection
  [✓] TRT 10.x get_tensor_name loop
[DONE] Restart Python/kernel để áp dụng.
```

Sau khi patch, **restart Python / Jupyter kernel** để áp dụng.

> **Nếu ultralytics ở vị trí không chuẩn**, truyền path thủ công:
> ```bash
> python patch_autobackend.py /path/to/ultralytics/nn/autobackend.py
> ```

> **Nếu dùng TensorRT 8.x**, bỏ qua bước này.

---

## Bước 4 — Inference

### Ảnh / Video

```python
from ultralytics import YOLO

model = YOLO("check_point_qat/best.tensorrt.engine", task="detect")

# Ảnh đơn
results = model("image.jpg", imgsz=640, conf=0.25)

# Thư mục ảnh
results = model("images/", imgsz=640, conf=0.25, save=True)

# Video
results = model("video.mp4", imgsz=640, conf=0.25, save=True)
```

### Đánh giá mAP

```python
from ultralytics import YOLO

model = YOLO("check_point_qat/best.tensorrt.engine", task="detect")
metrics = model.val(data="dataset.yaml", imgsz=640, device=0)

print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")
```

### Batch inference

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Engine build với max-batch=1 → phải loop từng ảnh
for img in images:
    result = model(img, verbose=False)

# Engine build với max-batch >= N → có thể chạy batch trực tiếp
results = model(images, verbose=False)
```

---

## Kết quả tham khảo

**Model:** YOLOv8s | **Dataset:** 3 class (plastic, metal, glass) | **imgsz:** 640 | **batch:** 1

| Metric | FP32 | INT8 QAT | Delta |
|--------|------|----------|-------|
| mAP50 | 0.8903 | 0.8857 | −0.52% |
| mAP50-95 | 0.7865 | 0.7876 | **+0.14%** |
| Latency | 9.21 ms | 3.96 ms | **2.33× faster** |
| FPS | 108.6 | 252.7 | **+132.7%** |
| Size | 22.5 MB | 19.1 MB | −15.1% |

*Đo trên NVIDIA A100-SXM4-80GB, TensorRT 10.16.0.72.*

---

## Môi trường đã test

| Package | Version | Ghi chú |
|---------|---------|---------|
| Python | 3.12.12 | |
| PyTorch | 2.10.0+cu128 | |
| torchvision | 0.25.0+cu128 | |
| CUDA | 12.8 | |
| TensorRT | 10.16.0.72 | Cài `==10.0.1`, Colab dùng bản pre-installed |
| ultralytics | 8.0.228 | Custom fork (cài bằng `pip install -e .`) |
| pytorch-quantization | 2.1.2 | |
| numpy | 1.26.4 | Phải `< 2.0` |
| onnx | latest | Không pin version |
| GPU | NVIDIA A100-SXM4-80GB | Google Colab |

---

## Engine portability

| Trường hợp | Engine có dùng được? |
|------------|----------------------|
| Cùng GPU, cùng máy | Có |
| Cùng dòng GPU (VD: 2× RTX 3090) | Thường có, khuyến nghị build lại |
| GPU khác dòng (VD: RTX 3090 → RTX 4090) | **Không — build lại** |
| GPU khác kiến trúc (VD: A100 → Jetson Orin) | **Không — build lại** |
| TensorRT version khác | **Không — build lại** |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pytorch_quantization'`
```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

### `ModuleNotFoundError: No module named 'onnxscript'`
```bash
pip install onnxscript
```

### `AttributeError: 'ICudaEngine' object has no attribute 'num_bindings'`
Đang dùng TensorRT 10.x nhưng chưa patch ultralytics:
```bash
python patch_autobackend.py
# Sau đó restart Python / kernel
```

### `AssertionError: input size torch.Size([N, ...]) not equal to max model size (1, ...)`
Engine được build với `--max-batch 1`. Phải inference từng ảnh hoặc build lại:
```bash
python export_qat_standalone.py --weight best.pt --nc 3 --max-batch 8
```

### `UnicodeDecodeError: 'utf-8' codec can't decode` khi load engine
Engine cũ không có metadata header. Build lại bằng `export_qat_standalone.py`.

### `[WARN] Không tìm thấy pattern để patch` khi chạy patch_autobackend.py
File autobackend.py đã bị sửa trước đó hoặc dùng version ultralytics khác.
Truyền path thủ công và kiểm tra file:
```bash
python patch_autobackend.py /path/to/ultralytics/nn/autobackend.py
```
