"""
critical_detection_improved.py - Instrument-based criticality & cutting detection
- Uses Cholec80 YOLOv8 model for multi-class surgical instrument detection.
- Hook-specific cutting detection: identifies active cutting during hook electrocautery
  dissection / tissue cutting bursts.
- Prevents false cutting positives when the hook is merely hovering or traversing without cutting.
- Graspers and other instruments NEVER trigger cutting.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from datetime import timedelta
import argparse
from collections import deque
from instrument_classifier import InstrumentClassifier


class CriticalEventDetectorImproved:
    def __init__(self, model_path, video_path, output_dir="critical_output_improved"):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize instrument classifier
        self.classifier = InstrumentClassifier()
        
        # Thresholds
        self.instrument_density_threshold = 1   # Any instrument present contributes
        self.movement_threshold = 60           # Rapid movement threshold (px)
        self.proximity_threshold = 150         # Pixels between instruments
        self.contact_threshold = 0.35          # Minimum overlap ratio for tissue contact
        
        # Temporal smoothing
        self.smooth_window = 5
        self.criticality_history = deque(maxlen=self.smooth_window)
        
        # Tracking & Cutting state
        self.prev_instruments = None
        self.frame_count = 0
        self.prev_hook_tip = None
        self.hook_contact_consecutive = 0
        self.cutting_history = deque(maxlen=12)
        self.cutting_cooldown = 0
    
    def detect_tissue_regions(self, frame):
        """Detect tissue/organ regions using laparoscopic color analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tissue color ranges (fleshy/red/pink tones typical in laparoscopic surgery)
        lower_tissue1 = np.array([0, 30, 45])
        upper_tissue1 = np.array([22, 210, 210])
        lower_tissue2 = np.array([148, 30, 45])
        upper_tissue2 = np.array([180, 210, 210])
        
        mask1 = cv2.inRange(hsv, lower_tissue1, upper_tissue1)
        mask2 = cv2.inRange(hsv, lower_tissue2, upper_tissue2)
        tissue_mask = cv2.add(mask1, mask2)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        
        return tissue_mask
    
    def detect_instrument_tissue_contact(self, instruments_info, tissue_mask):
        """Detect if instruments are in contact with tissue"""
        if len(instruments_info) == 0:
            return {'has_contact': False, 'contact_ratio': 0.0}
        
        h, w = tissue_mask.shape
        contact_pixels = 0
        total_instrument_pixels = 0
        
        for inst in instruments_info:
            x1, y1, x2, y2 = inst['box']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            inst_tissue_region = tissue_mask[y1:y2, x1:x2]
            inst_area = (x2 - x1) * (y2 - y1)
            
            if inst_area > 0:
                tissue_in_inst = np.count_nonzero(inst_tissue_region)
                contact_pixels += tissue_in_inst
                total_instrument_pixels += inst_area
        
        contact_ratio = contact_pixels / total_instrument_pixels if total_instrument_pixels > 0 else 0
        has_contact = contact_ratio > self.contact_threshold
        
        return {'has_contact': has_contact, 'contact_ratio': contact_ratio}
    
    def detect_hook_cutting(self, frame, instruments_info, tissue_mask):
        """
        Cutting is detected specifically when:
        1. A HOOK instrument is present (graspers alone cannot cut)
        2. Hook tip has contact with tissue/organ
        3. Hook executes active cutting (electrocautery dissection/burn flash/smoke burst)
        4. When Hook is hovering, traversing, or not cutting -> returns False
        """
        # Find Hook among detected instruments
        hook_info = None
        for inst in instruments_info:
            if inst['class'].lower() == 'hook':
                hook_info = inst
                break
        
        if hook_info is None:
            # No hook present -> cannot be cutting
            self.hook_contact_consecutive = 0
            self.prev_hook_tip = None
            self.cutting_history.append(False)
            return {
                'is_cutting': False,
                'has_hook': False,
                'hook_contact': False,
                'hook_sustained_frames': 0,
                'hook_displacement': 0.0,
                'cautery_intensity': 0.0
            }
        
        x1, y1, x2, y2 = hook_info['box']
        h, w = frame.shape[:2]
        
        # Hook tip region (bottom-left region where hook tip engages tissue)
        margin = 35
        tx1 = max(0, x1 - margin)
        ty1 = max(0, y2 - margin)
        tx2 = min(w, x1 + margin)
        ty2 = min(h, y2 + margin)
        
        tip_crop = frame[ty1:ty2, tx1:tx2]
        tip_tissue = tissue_mask[ty1:ty2, tx1:tx2]
        
        # Check contact at hook tip
        tip_tissue_ratio = np.count_nonzero(tip_tissue) / max(1, tip_tissue.size)
        hook_has_contact = tip_tissue_ratio > 0.30
        
        # Track hook tip position
        curr_tip = (float(x1), float(y2))
        disp = 0.0
        if self.prev_hook_tip is not None:
            disp = float(np.hypot(curr_tip[0] - self.prev_hook_tip[0], curr_tip[1] - self.prev_hook_tip[1]))
        self.prev_hook_tip = curr_tip
        
        # Measure active cautery/smoke/energy cutting burst at the hook tip
        cautery_ratio = 0.0
        if tip_crop.size > 0:
            tip_hsv = cv2.cvtColor(tip_crop, cv2.COLOR_BGR2HSV)
            # High-intensity white/smoke/plasma points
            cautery_ratio = float(np.count_nonzero(tip_hsv[..., 2] > 200) / max(1, tip_crop.size // 3))
        
        # Contact tracking
        if hook_has_contact:
            self.hook_contact_consecutive += 1
        else:
            self.hook_contact_consecutive = max(0, self.hook_contact_consecutive - 2)
            
        # Cutting triggered when:
        # Hook is in contact AND actively cutting tissue with cautery burst (threshold 0.015)
        is_cutting_now = hook_has_contact and (cautery_ratio >= 0.015)
        
        # Smooth cutting state so rapid bursts sustain for short continuity (8-10 frames)
        if is_cutting_now:
            self.cutting_cooldown = 10
        elif self.cutting_cooldown > 0:
            self.cutting_cooldown -= 1
            
        is_active_cutting = (self.cutting_cooldown > 0)
        
        return {
            'is_cutting': is_active_cutting,
            'has_hook': True,
            'hook_contact': hook_has_contact,
            'hook_sustained_frames': self.hook_contact_consecutive,
            'hook_displacement': disp,
            'cautery_intensity': cautery_ratio,
            'tip_box': (tx1, ty1, tx2, ty2)
        }
    
    def detect_instrument_movement(self, instruments_info):
        """Detect rapid instrument movements across frames"""
        if self.prev_instruments is None or len(instruments_info) == 0:
            self.prev_instruments = [inst['box'] for inst in instruments_info]
            return {'is_rapid_movement': False, 'avg_movement': 0.0}
        
        movements = []
        for curr in instruments_info:
            c_box = curr['box']
            curr_center = ((c_box[0] + c_box[2]) / 2, (c_box[1] + c_box[3]) / 2)
            for prev_box in self.prev_instruments:
                prev_center = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)
                distance = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                movements.append(distance)
        
        avg_movement = float(np.mean(movements)) if movements else 0.0
        is_rapid = avg_movement > self.movement_threshold
        self.prev_instruments = [inst['box'] for inst in instruments_info]
        return {'is_rapid_movement': is_rapid, 'avg_movement': avg_movement}
    
    def detect_instrument_proximity(self, instruments_info):
        """Detect if multiple instruments are close together"""
        if len(instruments_info) < 2:
            return {'is_close': False, 'avg_proximity': 0.0}
        
        centers = []
        for inst in instruments_info:
            b = inst['box']
            centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        
        proximities = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                proximities.append(dist)
        
        avg_proximity = float(np.mean(proximities)) if proximities else 0.0
        is_close = avg_proximity < self.proximity_threshold
        return {'is_close': is_close, 'avg_proximity': avg_proximity}
    
    def calculate_criticality_score(self, instrument_count, movement_data, proximity_data, contact_data, cutting_data):
        """Calculate criticality based on instrument-tissue interaction"""
        score = 0.0
        features = {}
        
        # Instrument density contribution
        density_score = 0.0
        if instrument_count >= self.instrument_density_threshold:
            density_score = 0.15 * min(1.0, instrument_count / 3.0)
        score += density_score
        features['density'] = density_score
        
        # Rapid movement contribution
        movement_score = 0.0
        if movement_data['is_rapid_movement']:
            movement_score = 0.15 * min(1.0, movement_data['avg_movement'] / 100.0)
        score += movement_score
        features['movement'] = movement_score
        
        # Proximity contribution
        proximity_score = 0.0
        if proximity_data['is_close']:
            proximity_score = 0.1 * (1.0 - proximity_data['avg_proximity'] / self.proximity_threshold)
        score += proximity_score
        features['proximity'] = proximity_score
        
        # TISSUE CONTACT
        contact_score = 0.0
        if contact_data['has_contact']:
            contact_score = 0.35 * contact_data['contact_ratio']
        score += contact_score
        features['contact'] = contact_score
        
        # CUTTING - HIGHEST PRIORITY
        cutting_score = 0.0
        if cutting_data['is_cutting']:
            cutting_score = 0.50
        score += cutting_score
        features['cutting'] = cutting_score
        
        return min(score, 1.0), features
    
    def apply_temporal_smoothing(self, criticality):
        """Apply temporal smoothing to criticality score"""
        self.criticality_history.append(criticality)
        return float(np.mean(self.criticality_history))
    
    def draw_overlays(self, frame, instruments_info, movement_data, proximity_data, contact_data, cutting_data, criticality, features, timestamp):
        """Draw real-time HUD overlays on frame"""
        h, w = frame.shape[:2]
        
        # Draw instrument boxes and labels
        for inst in instruments_info:
            x1, y1, x2, y2 = inst['box']
            cls_name = inst['class']
            conf = inst['conf']
            
            # Color code: Hook gets orange/red, others get cyan
            box_color = (0, 140, 255) if cls_name.lower() == 'hook' else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
        # Draw Hook tip box
        if cutting_data['has_hook'] and 'tip_box' in cutting_data:
            tx1, ty1, tx2, ty2 = cutting_data['tip_box']
            tip_color = (0, 0, 255) if cutting_data['is_cutting'] else (0, 255, 0)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), tip_color, 2)
            label = "CUTTING" if cutting_data['is_cutting'] else "Hook Tip"
            cv2.putText(frame, label, (tx1, max(15, ty1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tip_color, 1)
        
        # Semi-transparent HUD background panel
        panel_height = 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, panel_height), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Criticality score bar
        bar_width = int(criticality * 300)
        bar_color = (0, 255, 0) if criticality < 0.3 else (0, 255, 255) if criticality < 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (15, 15), (15 + 300, 32), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 15), (15 + bar_width, 32), bar_color, -1)
        cv2.putText(frame, f"CRITICALITY: {criticality:.2f}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Feature breakdown
        y_offset = 80
        cv2.putText(frame, f"Contact: {features['contact']:.3f}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y_offset += 22
        cv2.putText(frame, f"Cutting: {features['cutting']:.3f}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y_offset += 22
        cv2.putText(frame, f"Density: {features['density']:.3f}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Status indicators
        y_offset += 30
        contact_color = (0, 0, 255) if contact_data['has_contact'] else (0, 255, 0)
        contact_text = "TISSUE CONTACT" if contact_data['has_contact'] else "No Contact"
        cv2.putText(frame, f"[{contact_text}]", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, contact_color, 2)
        
        y_offset += 30
        cutting_color = (0, 0, 255) if cutting_data['is_cutting'] else (0, 255, 0)
        cutting_text = "CUTTING (HOOK)" if cutting_data['is_cutting'] else "No Cutting"
        cv2.putText(frame, f"[{cutting_text}]", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cutting_color, 2)
        
        # Timestamp
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video_realtime(self):
        """Process video with real-time inference and visualization"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        # Output CSV
        csv_path = os.path.join(self.output_dir, "critical_events.csv")
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame', 'timestamp', 'criticality_score',
            'contact_score', 'cutting_score', 'density_score', 'movement_score', 'proximity_score',
            'instrument_count', 'has_contact', 'contact_ratio',
            'has_hook', 'is_cutting', 'cautery_intensity',
            'is_rapid_movement', 'avg_movement',
            'is_close', 'avg_proximity'
        ])
        
        print(f"\nProcessing video: {self.video_path}")
        print("Press 'q' to quit, 's' to save current frame")
        print("=" * 60)
        
        cv2.namedWindow('Critical Event Detection (Improved)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Critical Event Detection (Improved)', 1280, 720)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            timestamp = self.frame_count / fps
            
            # YOLO instrument detection
            results = self.model(frame, conf=0.25, imgsz=640, verbose=False)
            instruments_info = []
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    instruments_info.append({
                        'class': cls_name,
                        'conf': conf,
                        'box': [x1, y1, x2, y2]
                    })
            
            # Tissue region detection
            tissue_mask = self.detect_tissue_regions(frame)
            
            # Instrument-tissue contact
            contact_data = self.detect_instrument_tissue_contact(instruments_info, tissue_mask)
            
            # Movement detection
            movement_data = self.detect_instrument_movement(instruments_info)
            
            # Hook cutting detection
            cutting_data = self.detect_hook_cutting(frame, instruments_info, tissue_mask)
            
            # Proximity detection
            proximity_data = self.detect_instrument_proximity(instruments_info)
            
            # Calculate criticality score
            criticality, features = self.calculate_criticality_score(
                len(instruments_info), movement_data, proximity_data, contact_data, cutting_data
            )
            
            # Temporal smoothing
            criticality = self.apply_temporal_smoothing(criticality)
            
            # Log to CSV
            csv_writer.writerow([
                self.frame_count,
                f"{timestamp:.2f}",
                f"{criticality:.3f}",
                f"{features['contact']:.3f}",
                f"{features['cutting']:.3f}",
                f"{features['density']:.3f}",
                f"{features['movement']:.3f}",
                f"{features['proximity']:.3f}",
                len(instruments_info),
                contact_data['has_contact'],
                f"{contact_data['contact_ratio']:.4f}",
                cutting_data['has_hook'],
                cutting_data['is_cutting'],
                f"{cutting_data['cautery_intensity']:.4f}",
                movement_data['is_rapid_movement'],
                f"{movement_data['avg_movement']:.2f}",
                proximity_data['is_close'],
                f"{proximity_data['avg_proximity']:.2f}"
            ])
            
            # Draw overlays
            display_frame = self.draw_overlays(
                frame.copy(), instruments_info, movement_data, proximity_data,
                contact_data, cutting_data, criticality, features, timestamp
            )
            
            # Show display
            cv2.imshow('Critical Event Detection (Improved)', display_frame)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(self.output_dir, f"frame_{self.frame_count}_critical_{criticality:.2f}.png")
                cv2.imwrite(save_path, display_frame)
                print(f"Saved frame: {save_path}")
            
            # Progress printout
            if self.frame_count % 30 == 0:
                print(f"[Frame {self.frame_count}] Criticality: {criticality:.3f} | "
                      f"Hook: {cutting_data['has_hook']} | "
                      f"Contact: {contact_data['has_contact']} | "
                      f"Cutting: {cutting_data['is_cutting']} | "
                      f"Inst: {len(instruments_info)}")
        
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Instrument-based criticality detection")
    parser.add_argument("--model", type=str, 
                        default=r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt",
                        help="Path to Cholec80 model")
    parser.add_argument("--video", type=str, 
                        default=r"C:\Users\thesa\Desktop\Sem-7\Healthcare\LaparoscopicDetection\cutting.mp4",
                        help="Path to video file")
    parser.add_argument("--output", type=str, default="critical_output_improved",
                        help="Output directory")
    
    args = parser.parse_args()
    
    detector = CriticalEventDetectorImproved(args.model, args.video, args.output)
    detector.process_video_realtime()


if __name__ == "__main__":
    main()
