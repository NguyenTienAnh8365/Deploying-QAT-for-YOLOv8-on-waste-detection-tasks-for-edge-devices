from ultralytics import YOLO

model = YOLO("weights/orignal.pt")

model.train(
    sr=1e-2,
    lr0=1e-3,
    data="data.yaml",
    epochs=50,
    patience=50,
    project='.',
    name='runs/train-sparsity',
    batch=48
)