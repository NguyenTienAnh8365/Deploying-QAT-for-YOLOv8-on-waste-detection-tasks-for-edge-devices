from ultralytics import YOLO


model = YOLO("runs/train-norm/weights/best.pt")

model.train(
    data="data.yaml",
    epochs=50,
    patience=20,

    imgsz=640,
    batch=48,
    workers=4,

    lr0=5e-4,         
    lrf=0.1,         
    warmup_epochs=3,

    sr=5e-3,           

    project=".",
    name="runs/train-sparsity",
    device=0           
)