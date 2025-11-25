import argparse
from ultralytics import YOLO

MODE_CONFIGS = {
    "standard": {
        "weights": "yolov8n.pt",
        "epochs": 200,
        "batch": 32,
        "project": ".",
        "name": "train-normal",
        "resume": False
    },
    "sparsity": {
        "weights": "runs/train-normal/weights/best.pt",
        "epochs": 50,
        "batch": 32,
        "sr": 5e-4,
        "project": ".",
        "name": "train-sparsity",
        "resume": False
    },
    "finetune": {
        "weights": "weights/pruned.pt",
        "epochs": 150,
        "batch": 32,
        "project": ".",
        "name": "train-finetune",
        "resume": False
    }
}


def merge_configs(defaults, overrides):
    new = defaults.copy()
    for k, v in overrides.items():
        if v is not None:
            new[k] = v
    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=MODE_CONFIGS.keys(), required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--sr", type=float, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    cfg = MODE_CONFIGS[args.mode]
    cfg = merge_configs(cfg, vars(args))

    model = YOLO(cfg["weights"])
    model.train(
        data=args.data,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        sr=cfg["sr"],
        project=cfg["project"],
        name=cfg["name"],
        resume=cfg["resume"]
    )
