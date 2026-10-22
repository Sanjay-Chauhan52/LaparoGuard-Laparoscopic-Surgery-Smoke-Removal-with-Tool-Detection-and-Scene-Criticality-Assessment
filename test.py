"""
test.py — YOLOv8 Evaluation & Inference Script for Laparoscopic Instrument Detection
"""

from ultralytics import YOLO
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Test / Evaluate YOLOv8 on Laparoscopic Dataset")
    parser.add_argument("--weights",    type=str,   default="runs/train/exp/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--data",       type=str,   default="data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--source",     type=str,   default=None,
                        help="Inference source: image path, folder, video, or webcam index. "
                             "Leave empty to run validation on the test split.")
    parser.add_argument("--imgsz",      type=int,   default=640,
                        help="Input image size")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--device",     type=str,   default="",
                        help="Device: '' auto, 'cpu', '0', etc.")
    parser.add_argument("--project",    type=str,   default="runs/test",
                        help="Output project directory")
    parser.add_argument("--name",       type=str,   default="exp",
                        help="Experiment name")
    parser.add_argument("--save",       action="store_true", default=True,
                        help="Save result images/videos")
    parser.add_argument("--save-txt",   action="store_true",
                        help="Save predictions as .txt label files")
    parser.add_argument("--save-conf",  action="store_true",
                        help="Include confidence scores in saved .txt labels")
    return parser.parse_args()


def validate(model, args):
    """Run YOLO validation on the test split."""
    print("\n[Mode] Validation on test split")
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        save_json=True,
    )
    print("\n--- Metrics ---")
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    return metrics


def infer(model, args):
    """Run inference on a custom source."""
    print(f"\n[Mode] Inference on source: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        stream=False,
    )
    print(f"\nInference complete. Results saved to: {os.path.join(args.project, args.name)}")
    return results


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Laparoscopic Detection — Testing")
    print(f"{'='*50}")
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print(f"{'='*50}\n")

    # Load trained model
    model = YOLO(args.weights)

    if args.source is None:
        # No custom source → run validation on the test split
        validate(model, args)
    else:
        # Custom source provided → run inference
        infer(model, args)


if __name__ == "__main__":
    main()
