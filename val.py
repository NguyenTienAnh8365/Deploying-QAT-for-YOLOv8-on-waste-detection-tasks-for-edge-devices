from ultralytics import YOLO
import torch
import time

# weight = "weights/best.pt"
# weight = "weights/orignal.pt"
weight = "weights/pruned_05.pt"
model = YOLO(weight)

device = 'cuda'
model.model.to(device)

dummy_img = torch.randn(1, 3, 640, 640).to(device)
with torch.no_grad():
    _ = model.model(dummy_img)

n_runs = 50  
latencies = []

for _ in range(n_runs):
    img = torch.randn(1, 3, 640, 640).to(device)
    start_time = time.time()
    with torch.no_grad():
        _ = model.model(img)
    latency = time.time() - start_time
    latencies.append(latency)

avg_latency = sum(latencies)/len(latencies)
print(f"Average Latency per image: {avg_latency*1000:.2f} ms")
print(f"Approx FPS: {1/avg_latency:.2f}")

metrics = model.val(data="data.yaml", batch=32, device=device, split="test")

print("=== Evaluation Metrics ===")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP@50:     {metrics.box.map50:.4f}")
print(f"mAP@50-95:  {metrics.box.map:.4f}")