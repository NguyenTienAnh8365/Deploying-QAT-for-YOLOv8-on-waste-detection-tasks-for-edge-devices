# create_gamma_ref.py
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # model pretrain COCO
gamma_dict = {}
for name, m in model.model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        gamma_dict[name] = m.weight.data.abs().clone().cpu()

torch.save(gamma_dict, "gamma_coco_ref.pt")
print("Saved gamma_coco_ref.pt")