"""
critical_detection.py — Detect critical events in laparoscopic surgery videos
- Bleeding/complications detection
- High-risk surgical phase identification
- Poor visibility (smoke) flagging
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from datetime import timedelta
import argparse


class CriticalEventDetector:
    def __init__(self, model_path, video_path, output_dir="critical_output"):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Detection thresholds
        self.smoke_threshold = 0.4  # Below this contrast = smoky
        self.bleeding_threshold = 0.15  # Red pixel ratio threshold
        self.instrument_density_threshold = 3  # High-risk if >3 instruments
        self.movement_threshold = 50  # Pixel movement threshold
        
        # Tracking
        self.prev_instruments = None
        self.frame_count = 0
        
    def detect_smoke(self, frame):
        """Detect poor visibility using contrast and brightness metrics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Contrast metric (standard deviation)
        contrast = np.std(gray) / 255.0
        
        # Brightness metric
        brightness = np.mean(gray) / 255.0
        
        # Laplacian variance for blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian) / 10000.0
        
        # Combined smoke score (lower = more smoke)
        smoke_score = (contrast * 0.4 + brightness * 0.3 + blur_score * 0.3)
        
        is_smoky = smoke_score < self.smoke_threshold
        
        return {
            'is_smoky': is_smoky,
            'smoke_score': smoke_score,
            'contrast': contrast,
            'brightness': brightness,
            'blur_score': blur_score
        }
    
    def detect_bleeding(self, frame):
        """Detect bleeding using HSV color analysis for red regions"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color ranges (two ranges for HSV wrap-around)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)
        
        # Calculate red pixel ratio
        total_pixels = frame.shape[0] * frame.shape[1]
        red_pixels = np.count_nonzero(red_mask)
        red_ratio = red_pixels / total_pixels
        
        is_bleeding = red_ratio > self.bleeding_threshold
        
        return {
            'is_bleeding': is_bleeding,
            'red_ratio': red_ratio,
            'red_pixels': red_pixels
        }
    
    def detect_instrument_movement(self, instruments):
        """Detect rapid instrument movements indicating emergency"""
        if self.prev_instruments is None or len(instruments) == 0:
            self.prev_instruments = instruments
            return {'is_rapid_movement': False, 'avg_movement': 0.0}
        
        movements = []
        for curr in instruments:
            for prev in self.prev_instruments:
                # Calculate center point movement
                curr_center = ((curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2)
                prev_center = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
                
                distance = np.sqrt(
                    (curr_center[0] - prev_center[0])**2 + 
                    (curr_center[1] - prev_center[1])**2
                )
                movements.append(distance)
        
        avg_movement = np.mean(movements) if movements else 0
        is_rapid = avg_movement > self.movement_threshold
        
        self.prev_instruments = instruments
        
        return {
            'is_rapid_movement': is_rapid,
            'avg_movement': avg_movement
        }
    
    def calculate_criticality_score(self, smoke_data, bleeding_data, instrument_count, movement_data):
        """Calculate overall criticality score (0-1)"""
        score = 0.0
        
        # Smoke contributes to criticality (poor visibility)
        if smoke_data['is_smoky']:
            score += 0.3 * (1 - smoke_data['smoke_score'])
        
        # Bleeding is highly critical
        if bleeding_data['is_bleeding']:
            score += 0.4 * (bleeding_data['red_ratio'] / self.bleeding_threshold)
        
        # High instrument density indicates complex maneuver
        if instrument_count > self.instrument_density_threshold:
            score += 0.2 * (instrument_count / 10)
        
        # Rapid movement indicates emergency response
        if movement_data['is_rapid_movement']:
            score += 0.3 * (movement_data['avg_movement'] / 100)
        
        return min(score, 1.0)
    
    def process_video(self):
        """Process video and detect critical events"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output CSV
        csv_path = os.path.join(self.output_dir, "critical_events.csv")
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame', 'timestamp', 'criticality_score',
            'is_smoky', 'smoke_score',
            'is_bleeding', 'red_ratio',
            'instrument_count', 'is_rapid_movement', 'movement_score'
        ])
        
        # Summary tracking
        critical_frames = []
        bleeding_frames = []
        smoky_frames = []
        
        print(f"\nProcessing video: {self.video_path}")
        print(f"FPS: {fps:.2f} | Total frames: {total_frames}")
        print("=" * 60)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            timestamp = self.frame_count / fps
            
            # YOLO instrument detection
            results = self.model(frame, conf=0.30, imgsz=640, verbose=False)
            instruments = []
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    instruments.append([x1, y1, x2, y2])
            
            # Smoke detection
            smoke_data = self.detect_smoke(frame)
            
            # Bleeding detection
            bleeding_data = self.detect_bleeding(frame)
            
            # Movement detection
            movement_data = self.detect_instrument_movement(instruments)
            
            # Calculate criticality
            criticality = self.calculate_criticality_score(
                smoke_data, bleeding_data, len(instruments), movement_data
            )
            
            # Log to CSV
            csv_writer.writerow([
                self.frame_count,
                f"{timestamp:.2f}",
                f"{criticality:.3f}",
                smoke_data['is_smoky'],
                f"{smoke_data['smoke_score']:.3f}",
                bleeding_data['is_bleeding'],
                f"{bleeding_data['red_ratio']:.4f}",
                len(instruments),
                movement_data['is_rapid_movement'],
                f"{movement_data['avg_movement']:.2f}"
            ])
            
            # Track critical events
            if criticality > 0.5:
                critical_frames.append((self.frame_count, timestamp, criticality))
            
            if bleeding_data['is_bleeding']:
                bleeding_frames.append((self.frame_count, timestamp, bleeding_data['red_ratio']))
            
            if smoke_data['is_smoky']:
                smoky_frames.append((self.frame_count, timestamp, smoke_data['smoke_score']))
            
            # Progress
            if self.frame_count % 30 == 0:
                print(f"[{self.frame_count}/{total_frames}] Frame {self.frame_count} - "
                      f"Criticality: {criticality:.3f} | "
                      f"Smoky: {smoke_data['is_smoky']} | "
                      f"Bleeding: {bleeding_data['is_bleeding']} | "
                      f"Instruments: {len(instruments)}")
        
        cap.release()
        csv_file.close()
        
        # Generate summary report
        self.generate_summary(critical_frames, bleeding_frames, smoky_frames, total_frames, fps)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")
    
    def generate_summary(self, critical_frames, bleeding_frames, smoky_frames, total_frames, fps):
        """Generate summary report of critical events"""
        summary_path = os.path.join(self.output_dir, "critical_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("LAPAROSCOPIC SURGERY CRITICAL EVENT ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Duration: {total_frames/fps:.2f} seconds\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("CRITICALITY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Critical frames (score > 0.5): {len(critical_frames)} ({len(critical_frames)/total_frames*100:.1f}%)\n")
            f.write(f"Bleeding detected: {len(bleeding_frames)} frames ({len(bleeding_frames)/total_frames*100:.1f}%)\n")
            f.write(f"Poor visibility (smoke): {len(smoky_frames)} frames ({len(smoky_frames)/total_frames*100:.1f}%)\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("HIGH CRITICALITY MOMENTS (Score > 0.5)\n")
            f.write("-" * 70 + "\n")
            if critical_frames:
                for frame, timestamp, score in critical_frames[:20]:  # Top 20
                    f.write(f"Frame {frame:5d} | {str(timedelta(seconds=timestamp))} | Score: {score:.3f}\n")
                if len(critical_frames) > 20:
                    f.write(f"... and {len(critical_frames) - 20} more\n")
            else:
                f.write("No high criticality moments detected.\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("BLEEDING EVENTS\n")
            f.write("-" * 70 + "\n")
            if bleeding_frames:
                for frame, timestamp, red_ratio in bleeding_frames[:20]:
                    f.write(f"Frame {frame:5d} | {str(timedelta(seconds=timestamp))} | Red ratio: {red_ratio:.4f}\n")
                if len(bleeding_frames) > 20:
                    f.write(f"... and {len(bleeding_frames) - 20} more\n")
            else:
                f.write("No bleeding events detected.\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("POOR VISIBILITY (SMOKE) EVENTS\n")
            f.write("-" * 70 + "\n")
            if smoky_frames:
                for frame, timestamp, smoke_score in smoky_frames[:20]:
                    f.write(f"Frame {frame:5d} | {str(timedelta(seconds=timestamp))} | Smoke score: {smoke_score:.3f}\n")
                if len(smoky_frames) > 20:
                    f.write(f"... and {len(smoky_frames) - 20} more\n")
            else:
                f.write("No poor visibility events detected.\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"\nSummary report: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect critical events in laparoscopic surgery videos")
    parser.add_argument("--model", type=str, 
                        default=r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt",
                        help="Path to YOLO model")
    parser.add_argument("--video", type=str, 
                        default=r"C:\Users\thesa\Desktop\LaparoscopicDetection\surgery.mp4",
                        help="Path to video file")
    parser.add_argument("--output", type=str, default="critical_output",
                        help="Output directory")
    
    args = parser.parse_args()
    
    detector = CriticalEventDetector(args.model, args.video, args.output)
    detector.process_video()


if __name__ == "__main__":
    main()
