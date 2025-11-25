from ultralytics import YOLO


model = YOLO("runs/train-norm/weights/best.pt")

model.train(
    sr=1e-2,
    lr0=1e-3,
    data="datasets/data.yaml",
    epochs=50,
    patience=50,
    project='.',
    name='runs/train-sparsity',
    batch=48,
    device="cpu"
)