from ultralytics import YOLO

model = YOLO("weights/yolov8n.pt")

model.train(
    data="datasets/data.yaml",
    epochs=100,
    imgsz=640,
    batch=128,
    lr0=0.0009,
    lrf=0.05,
    patience=20,
    project=".",
    name="runs/train-norm",
    pretrained=True,
    freeze=0
)