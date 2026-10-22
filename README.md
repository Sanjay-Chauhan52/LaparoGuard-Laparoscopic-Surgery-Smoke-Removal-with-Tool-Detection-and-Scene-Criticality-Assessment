# LaparoGuard: Laparoscopic Surgery Smoke Removal with Tool Detection and Scene Criticality Assessment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Cholec80-orange.svg)](http://ciss.u-strasbg.fr/cholec80/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  1. What is this Project?

**LaparoGuard** is a computer vision and deep learning system built for **laparoscopic surgery** (minimally invasive keyhole surgery, like gallbladder removal). 

During laparoscopic surgery, doctors insert a small camera and thin instruments through tiny incisions. When using electric tools to cut or burn tissue, a lot of **surgical smoke** is produced inside the patient's abdomen. This causes two major problems:
1. **Poor Visibility**: The surgeon cannot see blood vessels and tissue clearly.
2. **AI Detection Failure**: Computer vision models like YOLO fail to detect surgical instruments when they are covered by smoke.

### What LaparoGuard does:
- **Smoke Analysis & Desmoking Comparison**: Evaluates how smoke degrades surgical video quality and compares instrument detection on raw smoky frames versus cleaned (desmoked) frames.
- **Multi-Class Surgical Tool Detection**: Uses YOLOv8 (trained on the benchmark **Cholec80** dataset) to detect 7 types of surgical instruments in real-time.
- **Hook-Specific Cutting Detection**: Distinguishes between harmless tools (like graspers) and cutting tools (like the electric hook). It detects cutting only when the hook tip is in active contact with tissue and firing an electric cautery burst.
- **Scene Criticality Assessment**: Calculates an overall **Criticality Score (from 0.0 to 1.0)** based on tool movement, crowding, tissue contact, and active cutting, showing live alerts on a Heads-Up Display (HUD) and saving telemetry data to a CSV file.

---

##  2. Project Objectives

1. **Enhance Surgical Vision**: Analyze how surgical smoke reduces image contrast and sharpness, and prove that desmoking (smoke removal) significantly boosts tool detection accuracy.
2. **Real-Time Instrument Tracking**: Detect 7 essential laparoscopic tools at 30+ frames per second so the system can run live during a surgery.
3. **Accurate Cutting Detection**: Stop false cutting alerts. Graspers should never trigger cutting; only a hook touching tissue with an electric energy burst triggers a cutting event.
4. **Intraoperative Risk Scoring**: Combine tool contact, cutting, tool density, and speed into a single safety index (0.0 to 1.0) to warn surgeons during dangerous moments.
5. **Post-Surgery Auditing**: Automatically log all events into a CSV file to review surgical performance and train junior doctors.

---

##  3. Key Features

- **7-Class Tool Detection**:
  - `Hook`: L-shaped electric hook used for cutting and dissection.
  - `Grasper`: Tool for holding and pulling tissue (cannot cut).
  - `Bipolar`: Forceps that use electric current to stop bleeding.
  - `Clipper`: Applies small titanium clips to seal vessels.
  - `Scissors`: Micro-scissors for physical cutting.
  - `Irrigator`: Suction tube used to spray water and remove fluid.
  - `Bag`: Specimen pouch used to remove organs safely.
- **Smoke Removal & Quality Testing**:
  - Compares 660 smoky frames against 660 desmoked frames to prove how smoke removal improves detection confidence and recall.
  - Computes a mathematical **Visibility Score** based on local contrast, Laplacian sharpness, and edge density.
- **Smart Hook Cutting Detection**:
  - Looks specifically at the hook tip region ($35 \times 35$ pixels).
  - Checks if the tip is touching segmented tissue.
  - Checks for the bright white flash and smoke puff produced by electrocautery.
  - Smooths out the detection so brief smoke pauses still count as a continuous cutting stroke.
- **Tissue Contact Detection**: Uses HSV color masks to detect liver, gallbladder, and visceral tissue and calculates the exact overlap with instruments.
- **Movement & Proximity Tracking**: Detects sudden fast hand movements and alerts when two or more tools get too close to each other ($< 150$ pixels).
- **Live HUD & CSV Logger**: Shows a color-coded criticality bar, tool names, and timestamps on the video, and exports every frame's metrics into `critical_events.csv`.

---

##  4. System Architecture

The pipeline takes in a surgical video and processes it step-by-step:

```
                           Laparoscopic Video Input
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Preprocessing & Resizing  │
                        │    (640x640, BGR & HSV)   │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│     YOLOv8s Detector      │                   │   Tissue Color Mask       │
│ (Cholec80 7 Tool Classes) │                   │ (HSV Fleshy Tissue Range) │
└─────────────┬─────────────┘                   └─────────────┬─────────────┘
              │                                               │
              ├───────────────────────┬───────────────────────┘
              ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   Kinematics & Tracking   │   │    Tool-Tissue Contact    │
│  - Movement speed (px)    │   │  - Overlap ratio with     │
│  - Distance between tools │   │    segmented organ mask   │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │   Hook Cutting Checker    │
                │  - Is tool a Hook?        │
                │  - Is tip touching organ? │
                │  - Is cautery firing?     │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │ Criticality Score Engine  │
                │  - Weighted Sum Formula   │
                │  - 5-Frame Smoothing      │
                └─────────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  Live Video HUD Overlay   │   │ Telemetry CSV File Export │
│ (Green / Yellow / Red bar)│   │   (critical_events.csv)   │
└───────────────────────────┘   └───────────────────────────┘
```

---

##  5. How the Formulas & Scoring Work

### A. Criticality Score Formula
The total criticality score is a number between **0.0 (safe)** and **1.0 (high risk)**. It is calculated by adding 5 separate components:

$$\text{Criticality Score} = S_{\text{cutting}} + S_{\text{contact}} + S_{\text{density}} + S_{\text{movement}} + S_{\text{proximity}}$$

$$\text{Final Score} = \min(1.0, \; \text{Smoothed 5-frame average})$$

| Factor | Maximum Points | Formula | Why it matters |
| :--- | :---: | :--- | :--- |
| **Hook Cutting** | **0.50** | $+0.50$ when hook cuts tissue | Cutting tissue with electric heat is the most critical action in surgery. |
| **Tissue Contact** | **0.35** | $0.35 \times \text{Contact Ratio}$ | Tools pressing against organs have a higher chance of accidental injury. |
| **Tool Density** | **0.15** | $0.15 \times \min(1.0, \frac{\text{Tool Count}}{3})$ | Having multiple tools crowded in a small space increases danger. |
| **Rapid Movement** | **0.15** | $0.15 \times \min(1.0, \frac{\text{Speed}}{100})$ | Sudden fast motions may mean the surgeon lost grip or moved abruptly. |
| **Tool Proximity** | **0.10** | $0.10 \times (1.0 - \frac{\text{Distance}}{150})$ | Tools coming too close to each other can collide or pinch tissue. |

#### Criticality Risk Levels:
- **0.00 to 0.30 (Green - Normal)**: Tools are moving freely, no cutting, minimal contact.
- **0.30 to 0.60 (Yellow - Moderate Risk)**: Tools are touching tissue or moving close together.
- **0.60 to 1.00 (Red - High Risk)**: Active cutting with the hook, deep tissue contact, or multiple crowded tools.

---

### B. Hook Cutting Detection Rules
To prevent false cutting alarms, cutting is **ONLY** marked `True` when all three conditions are satisfied:
1. **Instrument is a Hook**: Graspers, clippers, and scissors give an automatic `0.0`.
2. **Hook Tip is Touching Tissue**: The tip crop ($35 \times 35$ px) must overlap with the tissue mask by at least $30\%$.
3. **Electric Cautery Burst**: The tip area must contain at least $1.5\%$ high-brightness pixels ($V > 200$ in HSV), which shows the electrical spark and smoke burst.

A 10-frame cooldown keeps the cutting state active during a normal cut, even if smoke briefly flickers for a split second.

---

### C. Smoke & Visibility Formula ([diagnostic_detection.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/diagnostic_detection.py))
To measure how much smoke is blocking the camera, the system calculates a **Visibility Score**:

$$\text{Visibility} = 0.35(\text{Local Contrast}) + 0.20(\text{Global Contrast}) + 0.20(\text{Sharpness}) + 0.15(\text{Edge Density}) + 0.10(\text{Brightness})$$

- **If Visibility $< 0.18$**: The video is flagged as **`SMOKE DETECTED: TRUE`**.

---

##  6. About the Datasets

### 1. The Cholec80 Benchmark Dataset
- **What it is**: 80 full videos of laparoscopic gallbladder removals recorded at Strasbourg University Hospital (France).
- **Labels**: Frame-by-frame annotations for 7 surgical tools, surgery phases, and actions.
- **Model Used**: We use the pretrained `yolov8s_cholec80.pt` model, fine-tuned specifically on this surgical dataset.

### 2. Smoky vs. Desmoke Comparative Dataset
- Located under `dataset/images/smoky/` and `dataset/images/desmoke/`.
- Contains **660 pairs of frames** (smoky original vs. smoke-removed).
- Used by `generate_detections.py` to compare tool detection before and after desmoking.

### 3. Custom Training Directory Structure
If you want to train YOLOv8 on your own annotated dataset, format it like this:

```
dataset/
├── images/
│   ├── train/      # Training image frames (.jpg, .png)
│   ├── val/        # Validation image frames
│   └── test/       # Test evaluation image frames
└── labels/
    ├── train/      # YOLO format label files (.txt)
    ├── val/        # YOLO format label files
    └── test/       # YOLO format label files
```
Each `.txt` label line has: `<class_id> <x_center> <y_center> <width> <height>` (all values between 0 and 1). Configured in [data.yaml](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/data.yaml).

---

##  7. Project File Structure

Here is how the files in this repository work:

- [critical_detection_improved.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/critical_detection_improved.py) : **Main script**. Runs real-time tool detection, hook cutting detection, criticality scoring, HUD display, and CSV export.
- [generate_detections.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/generate_detections.py) : Runs YOLOv8 on all 660 smoky frames and 660 desmoked frames to compare detection performance.
- [diagnostic_detection.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/diagnostic_detection.py) : Visualizes smoke levels, contrast, sharpness, and visibility scores.
- [download_model.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/download_model.py) : Automatically downloads the pretrained Cholec80 YOLOv8 model from Hugging Face.
- [instrument_classifier.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/instrument_classifier.py) : Helper module for classifying instrument shapes and types.
- [train.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/train.py) : Script for training or fine-tuning YOLOv8 on a custom dataset.
- [test.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/test.py) : Runs validation and calculates test metrics (mAP@0.5, Precision, Recall).
- [data.yaml](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/data.yaml) : Dataset configuration file for YOLO training.
- [requirements.txt](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/requirements.txt) : Python libraries required to run this project.
- `cutting.mp4` : Demo video showing active hook cutting and dissection.
- `surgery.mp4` : Demo video of a laparoscopic procedure.
- `Desmoking Video.mp4` : Video showing smoke removal in action.
- `Tool_detection_video/` : Output folder containing the annotated detection video (`surgery.mp4`).
- `detection_output/` : Output folder with annotated frames, label files, and CSV comparison results for smoky vs. desmoked runs.

---

##  8. Installation & Setup

### Requirements
- Windows 10/11 or Linux
- Python 3.10 or 3.11
- Webcam or sample video file (included)

### Step 1: Open PowerShell in Project Folder
```powershell
cd C:\Users\thesa\Desktop\LaparoscopicDetection
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create venv (if not already created)
python -m venv venv

# Activate venv on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 4: Download the Cholec80 Model Weights
```powershell
python download_model.py
```
This fetches `yolov8s_cholec80.pt` and caches it on your machine.

---

##  9. How to Run the Project

### A. Run Real-Time Critical Event & Tool Detection (Main System)
Run the improved detection pipeline on a surgical video:
```powershell
python critical_detection_improved.py --video "cutting.mp4"
```

#### Optional Arguments:
```powershell
python critical_detection_improved.py --video "surgery.mp4" --output "critical_output_improved"
```

#### Keyboard Shortcuts while video is playing:
- **`q`** : Quit the video and finish saving the CSV log.
- **`s`** : Save the current video frame with all HUD boxes, score bars, and metrics as an image.

---

### B. Run Smoky vs. Desmoke Detection Comparison
To run YOLOv8 across all 1,320 images in the smoky and desmoke folders and generate side-by-side detection CSV files:
```powershell
python generate_detections.py
```
Results will be saved in `detection_output/smoky_annotated/`, `detection_output/desmoke_annotated/`, and CSV files.

---

### C. Run Smoke Diagnostics
To inspect smoke levels, image contrast, and sharpness:
```powershell
python diagnostic_detection.py
```

---

### D. Train on a Custom Dataset
To train YOLOv8 on your own dataset defined in `data.yaml`:
```powershell
python train.py --data data.yaml --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640
```

---

### E. Test & Evaluate Model
To evaluate your trained model on the test split:
```powershell
python test.py --weights runs/train/exp/weights/best.pt --data data.yaml
```

---

##  10. Understanding the Output CSV File

When you run `critical_detection_improved.py`, it saves a complete log of every frame in `critical_output_improved/critical_events.csv`. Here is what each column means:

| Column | Meaning |
| :--- | :--- |
| `frame` | Current video frame number (1, 2, 3...) |
| `timestamp` | Time in seconds from the start of the video |
| `criticality_score` | Final smoothed safety risk score (0.00 to 1.00) |
| `contact_score` | Points contributed by tool-tissue contact |
| `cutting_score` | Points contributed by active hook cutting (+0.50 when cutting) |
| `density_score` | Points contributed by having multiple tools present |
| `movement_score` | Points contributed by fast tool speed |
| `proximity_score` | Points contributed by tools being very close together |
| `instrument_count` | Number of surgical tools visible in the frame |
| `has_contact` | `True` if any tool touches tissue; `False` otherwise |
| `contact_ratio` | Percentage of tool bounding box touching organ tissue |
| `has_hook` | `True` if the L-hook instrument is detected |
| `is_cutting` | `True` if active hook electrocautery cutting is occurring |
| `cautery_intensity` | Intensity of the white spark / smoke burst at the hook tip |
| `is_rapid_movement` | `True` if tool moved faster than 60 pixels between frames |
| `avg_movement` | Average tool movement speed in pixels |
| `is_close` | `True` if tools are closer than 150 pixels from each other |
| `avg_proximity` | Distance in pixels between instruments |

---

##  11. Faculty Presentation Guide (Simple Answers to Tough Questions)

When presenting to your professors and project evaluators, here are simple, clear explanations you can give:

**Q1: Why did you pick YOLOv8 for this medical project?**
> *"Laparoscopic surgery needs instant, real-time feedback (over 30 frames per second) so warnings appear without any delay. YOLOv8 is fast, accurate, and lightweight enough to run smoothly on standard GPUs while detecting multiple small surgical tools."*

**Q2: What is the connection between Smoke Removal and Tool Detection in your project?**
> *"During surgery, electric cutting tools produce heavy smoke. This smoke makes the video hazy and causes YOLO to miss tools. In our project, we show that removing smoke (desmoking) restores contrast and edges, which significantly improves tool detection accuracy and confidence."*

**Q3: How do you stop graspers from being wrongly detected as cutting?**
> *"In laparoscopic cholecystectomy, only the hook tool uses electric current to cut tissue. Graspers only pull and hold tissue. We programmed a strict rule: only the `Hook` class is allowed to trigger cutting. Furthermore, the hook must be touching tissue and firing an electric spark/smoke burst."*

**Q4: How does the Criticality Score help doctors?**
> *"The score turns complex visual details into a simple 0 to 1 risk indicator. A high score flags intense cutting or crowded tool movement. This can be used for live warnings during surgery and to automatically bookmark critical moments for post-surgery review and student training."*

---

##  12. Future Work

- **Automatic Bleeding Detection**: Detect sudden red blood pooling in real-time.
- **Surgery Phase Detection**: Identify whether the surgery is in dissection, clipping, or gallbladder retrieval phase.
- **3D Depth Estimation**: Calculate how close tools are to vital organs like the bile duct in millimeters.

---

##  13. References

- **Cholec80 Dataset**: Twinanda et al., *"EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos"*, IEEE TMI, 2017.
- **Ultralytics YOLOv8**: Modern real-time object detection architecture.

---

## 14. License

This project is licensed under the **MIT License** — see the [LICENSE](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/LICENSE) file for details.


