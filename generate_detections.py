

from ultralytics import YOLO
import os
import csv

# =========================================================
# MODEL
# =========================================================

MODEL_PATH = r"C:\Users\thesa\.cache\huggingface\hub\models--cesaraha--yolov8s-surgical-instrument-detection-cholec80\snapshots\c1f30aa178e1348bbfc1bb5a3a89971c51a977dc\yolov8s_cholec80.pt"

model = YOLO(MODEL_PATH)

# =========================================================
# FOLDERS
# =========================================================

BASE_DIR = r"C:\Users\thesa\Desktop\LaparoscopicDetection"

SMOKY_DIR = os.path.join(BASE_DIR, "dataset", "images", "smoky")
DESMOKE_DIR = os.path.join(BASE_DIR, "dataset", "images", "desmoke")

OUTPUT_DIR = os.path.join(BASE_DIR, "detection_output")

# Create output directories
for folder in [
    OUTPUT_DIR,
    os.path.join(OUTPUT_DIR, "smoky_annotated"),
    os.path.join(OUTPUT_DIR, "desmoke_annotated"),
    os.path.join(OUTPUT_DIR, "smoky_labels"),
    os.path.join(OUTPUT_DIR, "desmoke_labels"),
]:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================

CONFIDENCE = 0.30
IMAGE_SIZE = 640

# =========================================================
# PROCESS FUNCTION
# =========================================================

def process_folder(image_dir, type_name):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {type_name.upper()}")
    print("=" * 60)

    annotated_dir = os.path.join(
        OUTPUT_DIR,
        f"{type_name}_annotated"
    )

    labels_dir = os.path.join(
        OUTPUT_DIR,
        f"{type_name}_labels"
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{type_name}_detections.csv"
    )

    # CSV columns
    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "image",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    # -----------------------------------------------------
    # Process 0001.png -> 0660.png
    # -----------------------------------------------------

    for i in range(1, 661):

        filename = f"{i:04d}.png"

        image_path = os.path.join(
            image_dir,
            filename
        )

        # -----------------------------------------------
        # Check image
        # -----------------------------------------------

        if not os.path.exists(image_path):
            print(f"[WARNING] Missing: {filename}")
            continue

        print(
            f"[{i:03d}/660] Processing {filename}"
        )

        # -----------------------------------------------
        # YOLO inference
        # -----------------------------------------------

        results = model(
            source=image_path,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        # -----------------------------------------------
        # Image dimensions
        # -----------------------------------------------

        image_height, image_width = result.orig_shape

        # -----------------------------------------------
        # YOLO label file
        # -----------------------------------------------

        label_path = os.path.join(
            labels_dir,
            filename.replace(".png", ".txt")
        )

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as label_file:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # ---------------------------------------
                # Convert to YOLO format
                # ---------------------------------------

                x_center = (
                    (x1 + x2) / 2
                ) / image_width

                y_center = (
                    (y1 + y2) / 2
                ) / image_height

                width = (
                    x2 - x1
                ) / image_width

                height = (
                    y2 - y1
                ) / image_height

                # Keep values safely within range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                label_file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

                # ---------------------------------------
                # Save information to CSV
                # ---------------------------------------

                class_name = model.names[class_id]

                csv_writer.writerow([
                    filename,
                    class_name,
                    round(confidence, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ])

        # -----------------------------------------------
        # Save annotated image
        # -----------------------------------------------

        annotated_path = os.path.join(
            annotated_dir,
            filename
        )

        result.save(
            filename=annotated_path
        )

    csv_file.close()

    print("\nCompleted:", type_name)
    print("Labels:", labels_dir)
    print("Annotated images:", annotated_dir)
    print("CSV:", csv_path)


# =========================================================
# RUN
# =========================================================

process_folder(
    SMOKY_DIR,
    "smoky"
)

process_folder(
    DESMOKE_DIR,
    "desmoke"
)

print("\n" + "=" * 60)
print("ALL 1320 IMAGES PROCESSED")
print("=" * 60)