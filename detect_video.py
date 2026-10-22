from ultralytics import YOLO
import os

# Path to your downloaded Cholec80 model
MODEL_PATH = r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt"

# Your video
VIDEO_PATH = r"C:\Users\thesa\Desktop\Sem-7\Healthcare\LaparoscopicDetection\surgery.mp4"

# Load model
model = YOLO(MODEL_PATH)

# Run detection and save annotated video
results = model.predict(
    source=VIDEO_PATH,
    conf=0.30,
    imgsz=640,
    save=True,
    verbose=True
)

print("\nVideo detection completed.")