from ultralytics import YOLO

model = YOLO("weights/original.pt")

model.train(
    sr=8e-2,
    lr0=1.3e-3,
    data="data.yaml",
    epochs=100,
    patience=20,
    project='.',
    name='runs/train-sparsity',
    batch=64,
    device=0
)