from ultralytics import YOLO

model = YOLO("weights/yolov8s.pt")

model.train(
    sr=0,
    data="ultralytics/cfg/datasets/coco.yaml",
    epochs=200,
    project='.',
    name='runs/train-normal',
    batch=48,
    device=0
)