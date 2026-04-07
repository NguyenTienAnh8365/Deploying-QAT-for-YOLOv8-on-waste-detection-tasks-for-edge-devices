# Deploy QAT YOLOv8 trên Jetson Nano

> Jetson Nano không chạy được `export_qat_standalone.py` do Python 3.6 + CUDA 10.2 quá cũ.
> Giải pháp: **export ONNX trên Colab/PC, build engine trên Jetson Nano.**

---

## Thông tin thiết bị

| Thành phần | Version |
|-----------|---------|
| Hệ điều hành | Ubuntu 18.04.6 LTS |
| Python | 3.6.9 |
| CUDA | 10.2 |
| cuDNN | 8.2.1.32 |
| TensorRT | 8.2.1.8 |
| OpenCV | 4.1.1 |
| JetPack | 4.6.1 |
| L4T | 32.7.1 |
| GPU | Maxwell (sm_53), 128 CUDA cores |

---

## Vấn đề tương thích

| Yêu cầu export | Jetson Nano có | Kết quả |
|-----------------|----------------|---------|
| Python 3.8+ | 3.6.9 | Không chạy được export script |
| PyTorch 2.x | Max ~1.10 | Không cài được pytorch-quantization 2.1.2 |
| TensorRT 10.x | 8.2.1.8 | API khác (8.x dùng `num_bindings`) |
| ultralytics 8.0.228 | Không cài được | Python 3.6 không hỗ trợ |

**Giải pháp:** File ONNX (có Q/DQ nodes) là portable — chỉ cần copy sang Jetson rồi build engine tại chỗ.

---

## Bước 1 — Chuẩn bị trên Colab/PC

Export ONNX đã hoàn tất trên Colab. File cần copy sang Jetson:

```
check_point_qat/best.tensorrt.onnx    ← File ONNX với Q/DQ nodes (portable)
```

> **Lưu ý:** Nếu chưa export ONNX, chạy trên Colab:
> ```bash
> python export_qat_standalone.py \
>     --weight best.pt --nc 3 --names plastic,metal,glass --onnx-only
> ```

---

## Bước 2 — Copy ONNX sang Jetson Nano

```bash
# Qua SCP
scp best.tensorrt.onnx jetson@<IP_JETSON>:/home/jetson/model/

# Hoặc qua USB/SD card
cp best.tensorrt.onnx /media/jetson/usb/
```

---

## Bước 3 — Build engine trên Jetson Nano

### INT8 engine (dùng Q/DQ nodes từ QAT)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=/home/jetson/model/best.tensorrt.onnx \
    --saveEngine=/home/jetson/model/best_int8.engine \
    --int8 \
    --workspace=1024 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640
```

### FP16 engine (khuyến nghị cho Jetson Nano)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=/home/jetson/model/best.tensorrt.onnx \
    --saveEngine=/home/jetson/model/best_fp16.engine \
    --fp16 \
    --workspace=1024 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640
```

> Build engine trên Jetson Nano có thể mất **15–30 phút** do phần cứng yếu.

---

## INT8 vs FP16 trên Jetson Nano

Jetson Nano dùng GPU **Maxwell (sm_53)** — **không có INT8 Tensor Cores**:

| Jetson | GPU | INT8 Tensor Cores | INT8 speedup so với FP16 |
|--------|-----|-------------------|--------------------------|
| **Nano** | Maxwell (sm_53) | **Không** | **Không đáng kể hoặc chậm hơn** |
| TX2 | Pascal (sm_62) | Không | Không đáng kể |
| Xavier NX | Volta (sm_72) | Có | 2–3x |
| AGX Orin | Ampere (sm_87) | Có | 3–5x |

**Khuyến nghị:** Trên Jetson Nano, build cả INT8 và FP16, benchmark rồi chọn cái nhanh hơn.
FP16 thường nhanh hơn hoặc bằng INT8 trên Jetson Nano vì không có hardware INT8 chuyên dụng.

---

## Bước 4 — Nếu trtexec báo lỗi opset

ONNX được export với opset 17, TensorRT 8.2 có thể không hỗ trợ đầy đủ.

### Cách 1: Downgrade opset trên Colab trước khi copy

```python
import onnx
from onnx import version_converter

model = onnx.load("best.tensorrt.onnx")
model_converted = version_converter.convert_version(model, 13)
onnx.save(model_converted, "best.tensorrt.opset13.onnx")
```

### Cách 2: Re-export ONNX với opset thấp hơn trên Colab

Sửa `opset_version=13` trong `export_qat_standalone.py` rồi export lại.

---

## Bước 5 — Inference trên Jetson Nano

Jetson Nano không cài được ultralytics (Python 3.6), dùng TensorRT Python API trực tiếp:

### Cài dependencies

```bash
# pycuda (nếu chưa có)
pip3 install pycuda
```

### Script inference

```python
#!/usr/bin/env python3
"""Inference YOLOv8 TensorRT engine trên Jetson Nano."""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# ─── Config ───────────────────────────────────────────
ENGINE_PATH = "/home/jetson/model/best_fp16.engine"
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMGSZ       = 640
NAMES       = {0: "plastic", 1: "metal", 2: "glass"}
COLORS      = [(0,255,0), (255,0,0), (0,0,255)]


# ─── Load engine ─────────────────────────────────────
def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


# ─── Allocate buffers ─────────────────────────────────
def alloc_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(i):
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape})
    return inputs, outputs, bindings, stream


# ─── Preprocess ───────────────────────────────────────
def preprocess(img_bgr, imgsz=640):
    """Resize + normalize + NCHW."""
    h0, w0 = img_bgr.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    new_h, new_w = int(h0 * scale), int(w0 * scale)
    img_resized = cv2.resize(img_bgr, (new_w, new_h))

    # Letterbox padding
    pad_h = (imgsz - new_h) // 2
    pad_w = (imgsz - new_w) // 2
    img_padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

    # BGR → RGB, HWC → CHW, normalize
    img = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)  # [1, 3, 640, 640]
    return img, scale, pad_w, pad_h


# ─── Postprocess (YOLOv8 output format) ──────────────
def postprocess(output, conf_thresh, iou_thresh, scale, pad_w, pad_h):
    """Parse YOLOv8 output [1, nc+4, 8400] → boxes."""
    # output shape: [1, 4+nc, 8400]
    output = output[0]  # [4+nc, 8400]
    output = output.T   # [8400, 4+nc]

    boxes_xywh = output[:, :4]
    scores_all = output[:, 4:]

    max_scores = scores_all.max(axis=1)
    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores_all = scores_all[mask]
    max_scores = max_scores[mask]
    class_ids = scores_all.argmax(axis=1)

    if len(boxes_xywh) == 0:
        return [], [], []

    # xywh → xyxy
    x, y, w, h = boxes_xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Remove padding + rescale
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), max_scores.tolist(), conf_thresh, iou_thresh
    )
    if len(indices) == 0:
        return [], [], []
    indices = indices.flatten()

    return boxes_xyxy[indices], max_scores[indices], class_ids[indices]


# ─── Draw ─────────────────────────────────────────────
def draw_boxes(img, boxes, scores, class_ids, names, colors):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[cls_id % len(colors)]
        label = f"{names[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


# ─── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    engine, context = load_engine(ENGINE_PATH)
    inputs, outputs, bindings, stream = alloc_buffers(engine)
    print(f"[OK] Engine loaded: {ENGINE_PATH}")
    print(f"  Input : {inputs[0]['shape']}")
    print(f"  Output: {outputs[0]['shape']}")

    # Inference trên ảnh
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Khong doc duoc anh: {img_path}")
        sys.exit(1)

    img_input, scale, pad_w, pad_h = preprocess(img_bgr, IMGSZ)

    # Copy input → GPU → inference → copy output
    np.copyto(inputs[0]["host"], img_input.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    output = outputs[0]["host"].reshape(outputs[0]["shape"])
    boxes, scores, class_ids = postprocess(
        output, CONF_THRESH, IOU_THRESH, scale, pad_w, pad_h
    )

    print(f"  Detected: {len(boxes)} objects")
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"    {NAMES[cls_id]}: {score:.2f} @ [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # Draw + save
    img_out = draw_boxes(img_bgr.copy(), boxes, scores, class_ids, NAMES, COLORS)
    out_path = "result.jpg"
    cv2.imwrite(out_path, img_out)
    print(f"  Saved: {out_path}")

    # Benchmark
    print("\n[Benchmark] Warming up...")
    for _ in range(10):
        np.copyto(inputs[0]["host"], img_input.ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()

    runs = 100
    cuda.Context.synchronize()
    t0 = time.time()
    for _ in range(runs):
        np.copyto(inputs[0]["host"], img_input.ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()
    elapsed = (time.time() - t0) / runs * 1000

    print(f"  Latency : {elapsed:.1f} ms/img")
    print(f"  FPS     : {1000/elapsed:.1f}")
```

---

## Bước 6 — Inference video / camera

```python
#!/usr/bin/env python3
"""Inference video real-time trên Jetson Nano."""

import cv2
import time

# Import tất cả functions từ script inference ở bước 5
# (copy vào cùng file hoặc import)

ENGINE_PATH = "/home/jetson/model/best_fp16.engine"
IMGSZ = 640
NAMES = {0: "plastic", 1: "metal", 2: "glass"}
COLORS = [(0,255,0), (255,0,0), (0,0,255)]

engine, context = load_engine(ENGINE_PATH)
inputs, outputs, bindings, stream = alloc_buffers(engine)

# Mở camera (CSI camera trên Jetson Nano)
# cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)

# Hoặc USB camera
cap = cv2.VideoCapture(0)

# Hoặc video file
# cap = cv2.VideoCapture("video.mp4")

fps_avg = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    img_input, scale, pad_w, pad_h = preprocess(frame, IMGSZ)
    np.copyto(inputs[0]["host"], img_input.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    output = outputs[0]["host"].reshape(outputs[0]["shape"])
    boxes, scores, class_ids = postprocess(output, 0.25, 0.45, scale, pad_w, pad_h)

    elapsed = time.time() - t0
    fps = 1.0 / elapsed
    fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg else fps

    frame = draw_boxes(frame, boxes, scores, class_ids, NAMES, COLORS)
    cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 INT8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Hiệu năng dự kiến trên Jetson Nano

| Precision | Latency (ước tính) | FPS (ước tính) |
|-----------|-------------------|----------------|
| FP32 | ~200–300 ms | ~3–5 |
| **FP16** | **~80–120 ms** | **~8–12** |
| INT8 | ~80–150 ms | ~7–12 |

> Jetson Nano (128 CUDA cores, 472 GFLOPS FP16) yếu hơn T4 (~50x).
> FP16 thường nhanh hơn hoặc bằng INT8 trên Nano vì Maxwell không có INT8 Tensor Cores.

---

## Tổng kết pipeline

```
[Colab/PC]                              [Jetson Nano]
best.pt ──→ export_qat_standalone.py
         └──→ best.tensorrt.onnx ──SCP──→ trtexec --int8/--fp16
                                              └──→ best.engine
                                              └──→ infer_jetson.py (Python 3.6)
```

| Bước | Chạy ở đâu | Cần gì |
|------|-----------|--------|
| Export ONNX | Colab/PC | PyTorch, pytorch-quantization |
| Build engine | **Jetson Nano** | trtexec (có sẵn JetPack) |
| Inference | **Jetson Nano** | TensorRT + pycuda |
