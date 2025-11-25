import argparse
from ultralytics import YOLO

def train_model(weights, data, epochs, batch, sr, project, name, imgsz=640, lr0=0.001, lrf=0.05, patience=20, warmup_epochs=3, device=0, resume=False):
    """
    Train YOLO model with arbitrary parameters.
    """
    model = YOLO(weights)  # Load checkpoint or pretrained model

    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        warmup_epochs=warmup_epochs,
        sr=sr,
        project=project,
        name=name,
        device=device,
        resume=resume
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Unified Training Script")
    parser.add_argument("--weights", type=str, default="", help="Path to weights or checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sr", type=float, default=0.0, help="Sparsity factor")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="train-run")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    train_model(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        sr=args.sr,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        resume=args.resume
    )
