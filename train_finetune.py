from ultralytics import YOLO

weight = "weights/pruned.pt"

model = YOLO(weight)

model.train(
    data="datasets/data.yaml",
    epochs=200,
    finetune=True,
    device=0
)