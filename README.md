# LaparoGuard: AI-Powered Laparoscopic Surgical Tool Detection & Critical Phase Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8s-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Cholec80-orange.svg)](http://ciss.u-strasbg.fr/cholec80/)

---

## 📌 1. Project Overview

**LaparoGuard** is an intelligent, real-time Computer Vision system engineered for **Minimally Invasive Surgery (MIS)**, specifically laparoscopic cholecystectomy (gallbladder removal). Operating on surgical video streams, LaparoGuard detects and tracks specialized surgical instruments, monitors tissue-tool interactions, identifies high-risk surgical events (such as active electrosurgical cutting/dissection), and synthesizes these spatial-temporal features into a unified **Surgical Criticality Index (0.0 to 1.0)**.

During laparoscopy, surgical teams operate through small incisions using rigid long-shaft instruments guided by a 2D/3D camera. High-risk complications (such as inadvertent bile duct transection, vascular laceration, or thermal visceral burn) typically occur during active electrosurgical dissection when instruments maneuver in close contact with delicate anatomical tissue. 

LaparoGuard serves as an **AI-assisted surgical co-pilot** that:
- Identifies instruments frame-by-frame with bounding boxes and classification tags.
- Eliminates false cutting alerts by differentiating between non-cutting instruments (e.g., atraumatic graspers) and active electrosurgical tools (L-hook electrocautery).
- Segments visceral tissue regions using laparoscopic HSV color profiling.
- Quantifies tool-tissue contact area, tool velocity, and instrument crowding.
- Displays an intuitive Heads-Up Display (HUD) overlay and logs frame-level surgical telemetry to CSV for post-operative audit and training.

---

## 🎯 2. Objectives

### Clinical Objectives
1. **Intraoperative Risk Awareness**: Alert surgical personnel in real-time when instruments perform high-intensity dissection or operate in crowded, high-speed scenarios.
2. **Post-Operative Auditing & Video Indexing**: Automatically bookmark timestamps where critical dissections, thermal energy applications, or dense multi-tool manipulations took place.
3. **Surgical Skill Assessment & Training**: Provide objective quantitative metrics (motion smoothness, tissue contact duration, instrument economy) to evaluate residents and surgical fellows.

### Technical & Engineering Objectives
1. **Real-time Latency**: Achieve high-throughput inference (>30 FPS) suitable for live operating room video feeds using lightweight deep learning backbones (YOLOv8s).
2. **Context-Aware Cutting Detection**: Replace naive frame-difference motion triggers with hook-specific spatial localization, tissue boundary contact verification, and electrocautery plasma/smoke flash discrimination.
3. **Standardized Benchmarking**: Leverage the clinical gold-standard **Cholec80** surgical benchmark dataset.

---

## 🚀 3. Key Features

- **7-Class Surgical Instrument Detection**: Accurately recognizes and tracks specialized laparoscopic tools:
  - `Hook` (L-hook electrocautery)
  - `Grasper` (Atraumatic grasping forceps)
  - `Bipolar` (Bipolar electrocautery forceps)
  - `Clipper` (Titanium clip applier)
  - `Scissors` (Micro-laparoscopic scissors)
  - `Irrigator` (Suction & irrigation cannula)
  - `Bag` (Endoscopic specimen retrieval pouch)
- **Specialized Hook-Specific Cutting Detection**:
  - Distinguishes passive tool traversal from active tissue resection.
  - Monitors the hook's active electrode tip for tissue penetration.
  - Employs dynamic brightness/cautery burst thresholding to identify electrocautery activation and tissue vaporisation/smoke bursts.
  - Prevents erroneous cutting triggers when only graspers or retractors are in view.
- **Dynamic Tissue Segmentation**: Employs morphological processing and dual-spectrum HSV color masks tailored for internal anatomy, liver bed, visceral peritoneum, and fat pads.
- **Spatial Kinematics & Proximity Analysis**: Computes inter-instrument Euclidean distances and instrument tip velocities to detect hazardous crowding and sudden, uncontrolled motions.
- **Dynamic Criticality Engine**: Combines 5 distinct biomechanical signals into a unified risk metric with temporal moving-average smoothing ($W = 5$) and hysteresis state cooldown.
- **Surgeon HUD Overlay & Telemetry Logging**: Live graphical overlays displaying tool labels, tip interaction zones, color-coded criticality gauge, and automated CSV telemetry export.

---

## 🧠 4. System Architecture

The pipeline processes video frames sequentially through parallel vision modules before synthesizing observations in the Criticality Assessment Engine:

```
                                  Input Laparoscopic Video / Camera Feed
                                                    │
                                                    ▼
                             ┌──────────────────────────────────────────────┐
                             │          Preprocessing & Resizing            │
                             │         (640x640, BGR-to-RGB / HSV)          │
                             └──────────────────────┬───────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
    ┌─────────────────────────────┐                                   ┌─────────────────────────────┐
    │     YOLOv8s Deep Model      │                                   │   Laparoscopic Color Engine  │
    │  Cholec80 Multi-Class Tool  │                                   │     Tissue Segmentation     │
    │    Detection & Bounding     │                                   │       (HSV Dual-Mask)       │
    └──────────────┬──────────────┘                                   └──────────────┬──────────────┘
                   │                                                                 │
                   ├────────────────────────────────┬────────────────────────────────┘
                   ▼                                ▼
    ┌─────────────────────────────┐  ┌─────────────────────────────┐
    │   Instrument Kinematics     │  │   Tool-Tissue Contact Area  │
    │   - Euclidean Velocity      │  │   - Bounding Box Overlap    │
    │   - Multi-Tool Proximity    │  │   - Area Ratio Calculation  │
    └──────────────┬──────────────┘  └──────────────┬──────────────┘
                   │                                │
                   └────────────────┬───────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Hook Cutting Discriminator│
                     │   - Hook Tip Cropping       │
                     │   - Active Contact Check    │
                     │   - Cautery Energy Analysis │
                     │   - Hysteresis Cooldown     │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Criticality Scoring Engine  │
                     │   - Weighted Formula        │
                     │   - Temporal Smoothing      │
                     └──────────────┬──────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
    ┌─────────────────────────────┐   ┌─────────────────────────────┐
    │  Heads-Up Display Overlay   │   │  Frame Telemetry CSV Export │
    │ (Real-time Video Rendering) │   │   (critical_events.csv)     │
    └─────────────────────────────┘   └─────────────────────────────┘
```

---

## 📐 5. Mathematical Formulation & Scoring Engine

The **Surgical Criticality Score** ($C \in [0.0, 1.0]$) quantifies the instantaneous surgical hazard. It is calculated as the sum of five physiological and kinematic sub-scores, clamped at $1.0$:

$$C_{\text{raw}} = S_{\text{cutting}} + S_{\text{contact}} + S_{\text{density}} + S_{\text{movement}} + S_{\text{proximity}}$$

$$C_{\text{final}}(t) = \min\left(1.0, \; \frac{1}{W} \sum_{k=0}^{W-1} C_{\text{raw}}(t-k)\right)$$

Where $W = 5$ frames provides temporal moving-average smoothing to prevent frame-to-frame flicker.

### Sub-Score Breakdown

| Feature | Notation | Maximum Weight | Calculation Formula | Clinical Rationale |
| :--- | :---: | :---: | :--- | :--- |
| **Hook Cutting** | $S_{\text{cutting}}$ | **0.50** | $0.50 \times \mathbb{I}_{\{\text{is\_cutting}\}}$ | Electrosurgical dissection represents the highest risk of thermal injury and organ perforation. |
| **Tissue Contact** | $S_{\text{contact}}$ | **0.35** | $0.35 \times \text{ContactRatio}$ | Direct tool-tissue physical interaction precedes dissection or traction. |
| **Tool Density** | $S_{\text{density}}$ | **0.15** | $0.15 \times \min\left(1.0, \frac{N_{\text{inst}}}{3.0}\right)$ | Multiple instruments operating in a confined cavity elevate collision and blind-spot risk. |
| **Rapid Movement** | $S_{\text{movement}}$ | **0.15** | $0.15 \times \min\left(1.0, \frac{\bar{v}}{100.0}\right)$ | High-velocity tool displacements indicate urgent adjustments or loss of fine motor control. |
| **Tool Proximity** | $S_{\text{proximity}}$ | **0.10** | $0.10 \times \left(1.0 - \frac{\bar{d}}{d_{\text{thresh}}}\right)$ | Tools operating in extreme close proximity ($< 150 \text{ px}$) risk tool clash. |

---

### Hook-Specific Cutting Detection Algorithm

A critical innovation in [critical_detection_improved.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/critical_detection_improved.py) is the strict **Electrocautery Hook Dissection Discriminator**:

1. **Instrument Identity Gate**:
   $$\text{HasHook} = \exists \, i \in \text{Instruments} \text{ such that } \text{Class}(i) = \text{'Hook'}$$
   *If false, cutting is immediately assigned 0.0 regardless of frame activity.*

2. **Tip Region of Interest (ROI)**:
   Extracts a targeted sub-window $(x_1 \pm 35\text{px}, y_2 \pm 35\text{px})$ localized around the distal operative tip of the hook:
   $$\text{TipTissueRatio} = \frac{\sum_{(x,y) \in \text{ROI}} \text{TissueMask}(x,y)}{\text{Area}(\text{ROI})} > 0.30$$

3. **Active Cautery Energy & Smoke Flash**:
   Active electrocautery causes rapid tissue boiling, localized plasma ionization, and white smoke wisps. This is captured by analyzing high-intensity pixels in the HSV Value (V) channel:
   $$\text{CauteryRatio} = \frac{\sum_{(x,y) \in \text{ROI}} \mathbb{I}_{\{V(x,y) > 200\}}}{\text{Area}(\text{ROI})} \ge 0.015$$

4. **Hysteresis Smoothing**:
   $$\text{is\_cutting}(t) = \begin{cases} 
   \text{True}, & \text{if } \text{TipContact} \land (\text{CauteryRatio} \ge 0.015) \\
   \text{Sustained (Cooldown)}, & \text{cooldown} \in [1, 10] \\
   \text{False}, & \text{otherwise}
   \end{cases}$$

This ensures that smoke bursts and micro-pauses during a single cutting stroke are recorded as a continuous, stable cutting event rather than oscillating signals.

---

## 📊 6. Dataset Information

### The Cholec80 Benchmark
The detection model utilizes weights trained on the clinical gold-standard **Cholec80 Dataset**:
- **Origin**: University Hospital of Strasbourg (IRCAD / IHU Strasbourg).
- **Procedures**: 80 complete laparoscopic cholecystectomy procedures performed by 21 different surgeons.
- **Resolution & Frame Rate**: Recorded at 25 FPS, $854 \times 480$ or $1920 \times 1080$ resolution.
- **Annotations**: Frame-by-frame labeling of 7 surgical instruments, surgical phase transitions, and action triplets.

### Custom Dataset Structure (YOLO Format)
For retraining or fine-tuning on custom surgical recordings, the repository is structured as follows:

```
dataset/
├── images/
│   ├── train/      # Training frame images (.jpg, .png)
│   ├── val/        # Validation frame images
│   └── test/       # Test evaluation frame images
└── labels/
    ├── train/      # YOLO format annotations (.txt)
    ├── val/        # YOLO format annotations
    └── test/       # YOLO format annotations
```

Each `.txt` label file contains normalized annotations formatted as:
```text
<class-id> <x_center> <y_center> <width> <height>
```
Example:
```text
4 0.5241 0.6120 0.1450 0.2310
```
(Where class `4` corresponds to `Hook`).

Dataset configuration is declared in [data.yaml](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/data.yaml).

---

## 📂 7. Project Directory Structure

```
LaparoscopicDetection/
├── critical_detection_improved.py  # Primary production inference & criticality pipeline
├── critical_detection.py           # Baseline criticality detection script
├── instrument_classifier.py        # Shape-based and feature-based instrument classifier
├── download_model.py               # Hugging Face downloader for Cholec80 YOLOv8 weights
├── train.py                        # YOLOv8 training and augmentation script
├── test.py                         # Evaluation, validation, and inference script
├── data.yaml                       # YOLO dataset specification configuration
├── requirements.txt                # Python package dependencies
├── cutting.mp4                     # Sample surgical cut / dissection video
├── surgery.mp4                     # Extended procedure recording
├── critical_output_improved/       # Output directory for telemetry CSV & screenshots
│   └── critical_events.csv         # Frame-by-frame telemetry log
└── venv/                           # Isolated Python virtual environment
```

---

## ⚙️ 8. Installation & Environment Setup

### Prerequisites
- Windows 10/11 or Linux
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 11.8+ (Recommended for real-time >30 FPS inference; CPU execution is fully supported)

### Step 1: Clone or Open Workspace
```powershell
cd C:\Users\thesa\Desktop\LaparoscopicDetection
```

### Step 2: Set Up Virtual Environment
```powershell
# Create virtual environment (if not already present)
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Cholec80 Model
Run [download_model.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/download_model.py) to fetch the Cholec80 YOLOv8s checkpoint from Hugging Face:
```powershell
python download_model.py
```
The checkpoint will be saved to your local Hugging Face cache.

---

## 💻 9. How to Run the Project

### A. Run Real-Time Surgical Criticality Detection (Recommended)

To run the flagship pipeline on a surgical video with live Heads-Up Display and CSV telemetry logging:

```powershell
python critical_detection_improved.py --video "cutting.mp4"
```

#### Custom CLI Arguments:
```powershell
python critical_detection_improved.py `
    --model "C:\path\to\yolov8s_cholec80.pt" `
    --video "surgery.mp4" `
    --output "critical_output_improved"
```

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--video` | `str` | `cutting.mp4` | Path to the laparoscopic input video. |
| `--model` | `str` | *(HuggingFace Cache)* | Path to the trained YOLOv8 surgical model weights. |
| `--output` | `str` | `critical_output_improved` | Destination folder for `critical_events.csv` and frame captures. |

#### Interactive Keyboard Controls:
- **`q`** : Exit video stream and finalize telemetry export.
- **`s`** : Save a high-resolution screenshot of the current frame with all HUD overlays and score bars to the output folder.

---

### B. Train YOLOv8 on Custom Surgical Dataset

To train or fine-tune YOLOv8 on your custom annotated dataset:

```powershell
python train.py --data data.yaml --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640
```

Training parameters include:
- Optimizer: `AdamW` (warmup epochs = 3, cosine learning rate scheduler)
- Surgical data augmentations: HSV color jitter, scale shifting, horizontal flips, and Mosaic augmentation.
- Best model checkpoints will be stored under `runs/train/exp/weights/best.pt`.

---

### C. Evaluate Model on Test Set

To run formal test split validation and compute mAP, Precision, and Recall:

```powershell
python test.py --weights runs/train/exp/weights/best.pt --data data.yaml
```

To run batch inference on a folder of images or video files:

```powershell
python test.py --weights runs/train/exp/weights/best.pt --source "path/to/test_videos/"
```

---

## 📈 10. Output Telemetry & CSV Interpretation

Every run of [critical_detection_improved.py](file:///C:/Users/thesa/Desktop/LaparoscopicDetection/critical_detection_improved.py) produces a comprehensive, synchronized telemetry log: `critical_output_improved/critical_events.csv`.

### CSV Columns Description

| Column Header | Data Type | Description |
| :--- | :---: | :--- |
| `frame` | Integer | Sequentially indexed video frame number. |
| `timestamp` | Float (s) | Elapsed surgical time in seconds. |
| `criticality_score` | Float (0.0-1.0) | Final smoothed criticality index. |
| `contact_score` | Float (0.0-0.35) | Contribution of tool-tissue contact to criticality. |
| `cutting_score` | Float (0.0-0.50) | Contribution of hook electrocautery cutting. |
| `density_score` | Float (0.0-0.15) | Contribution of instrument crowding/count. |
| `movement_score` | Float (0.0-0.15) | Contribution of sudden instrument velocity. |
| `proximity_score` | Float (0.0-0.10) | Contribution of inter-instrument closeness. |
| `instrument_count`| Integer | Total distinct tools detected in the frame. |
| `has_contact` | Boolean | True if tool area intersects segmented tissue. |
| `contact_ratio` | Float | Ratio of instrument bounding box overlapping tissue. |
| `has_hook` | Boolean | True if an L-hook instrument is currently detected. |
| `is_cutting` | Boolean | True during active hook electrocautery resection. |
| `cautery_intensity`| Float | High-value plasma/smoke pixel ratio at the hook tip. |
| `is_rapid_movement`| Boolean | True if tool displacement exceeds movement threshold. |
| `avg_movement` | Float (px) | Mean pixel displacement between consecutive frames. |
| `is_close` | Boolean | True if distance between instruments $< 150 \text{ px}$. |
| `avg_proximity` | Float (px) | Mean pairwise Euclidean distance between instruments. |

---

## 🎓 11. Faculty & Academic Presentation Guide

When presenting this project to university faculty, examiners, or healthcare panels, use the following talking points to highlight technical depth:

### Frequently Asked Questions & Defenses

1. **"Why use YOLOv8 instead of older architectures like Faster R-CNN or SSD?"**
   > *Laparoscopic video requires low-latency, real-time feedback (>30 FPS) so warnings reach the surgeon without lag. YOLOv8 utilizes an anchor-free detection head, C2f feature aggregation, and Mosaic data augmentation, achieving superior mean Average Precision (mAP) while maintaining high inference speeds.*

2. **"How do you prevent false cutting detections when graspers move rapidly?"**
   > *We enforce a strict multi-stage gating rule: only the `Hook` class is permitted to trigger cutting. Graspers, clippers, and irrigators are anatomically non-cutting tools in cholecystectomy and their cutting score is hard-locked to zero. Furthermore, the hook must exhibit both tissue contact and high-intensity electrocautery discharge.*

3. **"Why not use a pure 3D Convolutional Network (e.g., SlowFast / I3D) for cutting?"**
   > *3D CNNs require massive compute clusters, high memory bandwidth, and suffer from high latency. Our hybrid architecture combines 2D spatial deep learning (YOLOv8) with kinematic tracking, color-space tissue segmentation, and temporal moving-average queues. This achieves comparable event detection accuracy at a fraction of the computational footprint.*

4. **"How does the system handle surgical smoke or blood obscuration?"**
   > *Surgical smoke produces high-value, low-saturation pixel blooms in the HSV space. Our algorithm leverages this exact characteristic at the hook tip as an indicator of electrocautery activation, turning an environmental challenge into a discriminative feature for cutting detection.*

---

## 🗺️ 12. Future Scope & Roadmap

- [ ] **Action Triplet Recognition**: Classify `<Instrument, Verb, Target>` triplets (e.g., `<Hook, Dissect, Gallbladder>`).
- [ ] **Automated Bleeding Detection**: Deep segmentation of rapid red-channel pixel expansion to detect arterial hemorrhage.
- [ ] **3D Tool Depth Estimation**: Integrate monocular depth estimation to monitor distance to critical anatomical structures (e.g., Common Bile Duct).
- [ ] **Robotic Surgery Integration**: Connect the telemetry pipeline to ROS2 (Robot Operating System) for da Vinci surgical consoles.

---

## 📚 13. References & Acknowledgments

- **Cholec80 Benchmark**: Twinanda et al., *"EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos"*, IEEE Transactions on Medical Imaging, 2017.
- **YOLOv8 Framework**: Jocher et al., *Ultralytics YOLOv8*, 2023.
- **Pretrained Weights**: Hugging Face repository `cesaraha/yolov8s-surgical-instrument-detection-cholec80`.
- Developed as part of the **B.Tech 7th Semester Healthcare AI Capstone Project**.
