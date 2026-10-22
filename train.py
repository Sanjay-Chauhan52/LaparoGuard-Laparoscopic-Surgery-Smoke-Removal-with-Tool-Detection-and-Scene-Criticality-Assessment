"""
train.py — YOLOv8 Training Script for Laparoscopic Instrument Detection
"""

from ultralytics import YOLO
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Laparoscopic Dataset")
    parser.add_argument("--model",   type=str,   default="yolov8n.pt",    help="Pretrained model weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--data",    type=str,   default="data.yaml",     help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int,   default=100,             help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int,   default=640,             help="Input image size")
    parser.add_argument("--batch",   type=int,   default=16,              help="Batch size (-1 for AutoBatch)")
    parser.add_argument("--device",  type=str,   default="",             help="Device: '' for auto, 'cpu', '0', '0,1' etc.")
    parser.add_argument("--workers", type=int,   default=8,               help="Number of dataloader workers")
    parser.add_argument("--project", type=str,   default="runs/train",    help="Output project directory")
    parser.add_argument("--name",    type=str,   default="exp",           help="Experiment name")
    parser.add_argument("--resume",  action="store_true",                 help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Laparoscopic Detection — Training")
    print(f"{'='*50}")
    print(f"  Model   : {args.model}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img Size: {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"{'='*50}\n")

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # ---------- Augmentation ----------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        # ---------- Optimisation ----------
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # ---------- Misc ----------
        plots=True,
        save=True,
        val=True,
        patience=20,
    )

    print("\nTraining complete!")
    print(f"Results saved to: {os.path.join(args.project, args.name)}")
    return results


if __name__ == "__main__":
    main()
