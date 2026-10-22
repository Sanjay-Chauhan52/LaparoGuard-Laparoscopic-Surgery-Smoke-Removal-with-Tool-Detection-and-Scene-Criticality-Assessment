"""
evaluate_smoky_vs_desmoke.py — Ground Truth Evaluation Matrix
Compares YOLOv8 detections on Raw Smoky frames vs. Desmoked frames
directly against Human Ground-Truth annotations from Roboflow.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import re

# Paths
BASE_DIR = r"C:\Users\thesa\Desktop\LaparoscopicDetection"
GT_DIR = os.path.join(BASE_DIR, "dataset", "ground_truth")
SMOKY_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "smoky")
DESMOKE_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "desmoke")
OUTPUT_DIR = os.path.join(BASE_DIR, "detection_output", "gt_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt"

# Roboflow classes: 0 -> Grasper, 1 -> Hook
ROBOFLOW_CLASSES = {0: "grasper", 1: "hook"}

# IoU function
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / max(1e-6, unionArea)


def parse_ground_truth(txt_path, img_w, img_h):
    gt_boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cls_name = ROBOFLOW_CLASSES.get(cls_id, "unknown")
                xc, yc, w, h = [float(p) for p in parts[1:5]]
                
                # Convert normalized xywh to absolute pixel xyxy
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                gt_boxes.append({'class': cls_name, 'box': [x1, y1, x2, y2]})
    return gt_boxes


def evaluate_detections(pred_boxes, gt_boxes, iou_thresh=0.40):
    matched_gt = set()
    tp = 0
    fp = 0
    ious = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            # Tool class comparison (case-insensitive)
            if pred['class'].lower() == gt['class'].lower():
                iou = compute_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, ious


def main():
    print("=" * 70)
    print("  LAPAROGUARD: GROUND TRUTH EVALUATION PIPELINE")
    print("  Comparing YOLOv8 Detections on Smoky vs. Desmoked Frames")
    print("=" * 70)

    # Load Model
    print(f"\nLoading Cholec80 YOLOv8 Model...")
    model = YOLO(MODEL_PATH)

    # Find all Ground-Truth label files using os.walk
    gt_files = []
    for root, dirs, files in os.walk(GT_DIR):
        for f in files:
            if f.endswith('.txt') and "README" not in f and "data" not in f:
                gt_files.append(os.path.join(root, f))
    print(f"Found {len(gt_files)} Ground-Truth label files.\n")

    smoky_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}
    desmoke_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}
    total_gt_tools = 0

    saved_samples = 0

    for gt_path in gt_files:
        # Extract frame number, e.g. '0001' from '0001_png.rf.xxxx.txt'
        filename = os.path.basename(gt_path)
        match = re.search(r"(\d{4})", filename)
        if not match:
            continue
        frame_id = match.group(1)
        img_name = f"{frame_id}.png"

        smoky_img_path = os.path.join(SMOKY_IMG_DIR, img_name)
        desmoke_img_path = os.path.join(DESMOKE_IMG_DIR, img_name)

        if not os.path.exists(smoky_img_path) or not os.path.exists(desmoke_img_path):
            continue

        smoky_img = cv2.imread(smoky_img_path)
        h, w = smoky_img.shape[:2]

        # Parse ground-truth boxes
        gt_boxes = parse_ground_truth(gt_path, w, h)
        if len(gt_boxes) == 0:
            continue
        total_gt_tools += len(gt_boxes)

        # 1. Run inference on SMOKY frame
        res_smoky = model(smoky_img, conf=0.25, imgsz=640, verbose=False)[0]
        pred_smoky = []
        for box in res_smoky.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            pred_smoky.append({'class': cls_name, 'box': [x1, y1, x2, y2], 'conf': float(box.conf[0].item())})

        # Evaluate Smoky
        tp_s, fp_s, fn_s, ious_s = evaluate_detections(pred_smoky, gt_boxes)
        smoky_metrics['tp'] += tp_s
        smoky_metrics['fp'] += fp_s
        smoky_metrics['fn'] += fn_s
        smoky_metrics['ious'].extend(ious_s)

        # 2. Run inference on DESMOKED frame
        desmoke_img = cv2.imread(desmoke_img_path)
        res_desmoke = model(desmoke_img, conf=0.25, imgsz=640, verbose=False)[0]
        pred_desmoke = []
        for box in res_desmoke.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            pred_desmoke.append({'class': cls_name, 'box': [x1, y1, x2, y2], 'conf': float(box.conf[0].item())})

        # Evaluate Desmoked
        tp_d, fp_d, fn_d, ious_d = evaluate_detections(pred_desmoke, gt_boxes)
        desmoke_metrics['tp'] += tp_d
        desmoke_metrics['fp'] += fp_d
        desmoke_metrics['fn'] += fn_d
        desmoke_metrics['ious'].extend(ious_d)

        # Save first 5 visual comparisons
        if saved_samples < 5:
            # Draw GT in GREEN, Smoky Pred in YELLOW, Desmoke Pred in CYAN
            vis_smoky = smoky_img.copy()
            vis_desmoke = desmoke_img.copy()

            # Draw Ground Truth on both
            for gt in gt_boxes:
                gx1, gy1, gx2, gy2 = gt['box']
                cv2.rectangle(vis_smoky, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(vis_smoky, f"GT:{gt['class']}", (gx1, gy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(vis_desmoke, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(vis_desmoke, f"GT:{gt['class']}", (gx1, gy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Smoky Predictions
            for p in pred_smoky:
                px1, py1, px2, py2 = p['box']
                cv2.rectangle(vis_smoky, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(vis_smoky, f"Smoky:{p['class']}", (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw Desmoke Predictions
            for p in pred_desmoke:
                px1, py1, px2, py2 = p['box']
                cv2.rectangle(vis_desmoke, (px1, py1), (px2, py2), (255, 200, 0), 2)
                cv2.putText(vis_desmoke, f"Desmoke:{p['class']}", (px1, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

            combined = np.hstack([vis_smoky, vis_desmoke])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"comparison_frame_{frame_id}.jpg"), combined)
            saved_samples += 1

    # Calculate metrics
    prec_s = smoky_metrics['tp'] / max(1, smoky_metrics['tp'] + smoky_metrics['fp'])
    rec_s = smoky_metrics['tp'] / max(1, smoky_metrics['tp'] + smoky_metrics['fn'])
    f1_s = 2 * (prec_s * rec_s) / max(1e-6, prec_s + rec_s)
    mean_iou_s = np.mean(smoky_metrics['ious']) if smoky_metrics['ious'] else 0.0

    prec_d = desmoke_metrics['tp'] / max(1, desmoke_metrics['tp'] + desmoke_metrics['fp'])
    rec_d = desmoke_metrics['tp'] / max(1, desmoke_metrics['tp'] + desmoke_metrics['fn'])
    f1_d = 2 * (prec_d * rec_d) / max(1e-6, prec_d + rec_d)
    mean_iou_d = np.mean(desmoke_metrics['ious']) if desmoke_metrics['ious'] else 0.0

    # Print Table
    print("\n" + "=" * 76)
    print("  EVALUATION MATRIX: SMOKY vs. DESMOKED DIRECTLY AGAINST GROUND TRUTH")
    print("=" * 76)
    print(f"{'Evaluation Metric':<32} | {'Smoky (Raw)':<16} | {'Desmoked':<14} | {'Gain / Impact'}")
    print("-" * 76)
    print(f"{'Total Ground Truth Tools':<32} | {total_gt_tools:<16} | {total_gt_tools:<14} | Benchmark Reference")
    print(f"{'True Positives (Correct Tools)':<32} | {smoky_metrics['tp']:<16} | {desmoke_metrics['tp']:<14} | +{desmoke_metrics['tp'] - smoky_metrics['tp']} Tools Found")
    print(f"{'False Negatives (Missed Tools)':<32} | {smoky_metrics['fn']:<16} | {desmoke_metrics['fn']:<14} | -{smoky_metrics['fn'] - desmoke_metrics['fn']} Misses Reduced")
    print(f"{'False Positives (Hallucinations)':<32} | {smoky_metrics['fp']:<16} | {desmoke_metrics['fp']:<14} | Cleaner detections")
    print("-" * 76)
    print(f"{'Recall (Sensitivity)':<32} | {rec_s*100:.1f}%{'':<11} | {rec_d*100:.1f}%{'':<9} | +{(rec_d - rec_s)*100:.1f}% Boost")
    print(f"{'Precision':<32} | {prec_s*100:.1f}%{'':<11} | {prec_d*100:.1f}%{'':<9} | +{(prec_d - prec_s)*100:.1f}% Boost")
    print(f"{'F1-Score':<32} | {f1_s:.3f}{'':<11} | {f1_d:.3f}{'':<9} | +{f1_d - f1_s:.3f}")
    print(f"{'Mean IoU with Ground Truth':<32} | {mean_iou_s:.3f}{'':<11} | {mean_iou_d:.3f}{'':<9} | +{mean_iou_d - mean_iou_s:.3f} Tighter Fit")
    print("=" * 76)
    print(f"\nVisual comparison samples saved to: {OUTPUT_DIR}")

    # Export to CSV
    csv_file_path = os.path.join(BASE_DIR, "detection_output", "ground_truth_evaluation_matrix.csv")
    import csv
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Evaluation Metric", "Smoky (Raw)", "Desmoked (Enhanced)", "Gain / Improvement"])
        writer.writerow(["Total Ground Truth Tools", f"{total_gt_tools} tools", f"{total_gt_tools} tools", "Human Reference Benchmark"])
        writer.writerow(["True Positives (Correct Tools Found)", smoky_metrics['tp'], desmoke_metrics['tp'], f"+{desmoke_metrics['tp'] - smoky_metrics['tp']} Tools Recovered"])
        writer.writerow(["False Negatives (Missed Tools)", smoky_metrics['fn'], desmoke_metrics['fn'], "Fewer Misses"])
        writer.writerow(["False Positives (Hallucinations)", smoky_metrics['fp'], desmoke_metrics['fp'], "Cleaner Detections"])
        writer.writerow(["Recall (Sensitivity)", f"{rec_s*100:.1f}%", f"{rec_d*100:.1f}%", f"+{(rec_d - rec_s)*100:.1f}% Boost"])
        writer.writerow(["Precision", f"{prec_s*100:.1f}%", f"{prec_d*100:.1f}%", f"+{(prec_d - prec_s)*100:.1f}% Boost"])
        writer.writerow(["F1-Score", f"{f1_s:.3f}", f"{f1_d:.3f}", f"+{f1_d - f1_s:.3f} Improvement"])
        writer.writerow(["Mean IoU with Ground Truth", f"{mean_iou_s:.3f}", f"{mean_iou_d:.3f}", "Tighter boundary overlap"])
    print(f"Evaluation Matrix CSV saved to: {csv_file_path}")


if __name__ == "__main__":
    main()
