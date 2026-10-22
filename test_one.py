from ultralytics import YOLO
import os

# Path to downloaded model
model_path = r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt"

# Load model
model = YOLO(model_path)

# Your smoky image
image_path = r"C:\Users\thesa\Desktop\LaparoscopicDetection\dataset\images\smoky\0001.png"

# Check image exists
if not os.path.exists(image_path):
    print("Image not found:", image_path)
    exit()

# Run detection
results = model(
    source=image_path,
    conf=0.30,
    imgsz=640
)

# Create output folder
os.makedirs("test_output", exist_ok=True)

for result in results:

    # Save image with bounding boxes
    output_path = "test_output/-0001_detected.png"
    result.save(filename=output_path)

    print("\nDetected instruments:")

    if len(result.boxes) == 0:
        print("No instruments detected.")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"Instrument: {class_name} | "
            f"Confidence: {confidence:.3f} | "
            f"Box: ({x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f})"
        )

    print("\nSaved to:", output_path)