from ultralytics import YOLO

weight = "weights/yolov8s.pt"

model = YOLO(weight)
metrics = model.val(data="dataset/data.yaml")