# Traffic-Intensity-Classification-Using-YOLOv8-and-XGBoost
A real-time vehicle detection, tracking, and congestion analysis system built for **Differenbt road conditions** using YOLOv8 and OpenCV.

---

## What It Does

- Loads a traffic video recorded under foggy conditions
- User manually draws **two lane polygons** on the first frame using mouse clicks
- YOLOv8 detects and tracks vehicles frame-by-frame across **both lanes**
- Draws **bounding boxes** around every detected vehicle in both lanes with unique tracking IDs
- Calculates **real-time speed** (km/h) for each tracked vehicle using centroid displacement
- Computes **lane occupancy density** using bounding box intersection area with the polygon
- Classifies congestion level per lane as **LOW / MEDIUM / HIGH**
- Exports all metrics to a **CSV file** every 10 seconds and at the end of the video

---

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| YOLOv8 (Ultralytics) | Vehicle detection & tracking |
| OpenCV | Video processing, bounding box drawing, UI |
| PyTorch (CUDA) | GPU-accelerated inference |
| Shapely | Polygon geometry for lane zones |

---

##  How It Works

### 1. Lane Zone Drawing
On startup, the first frame of the video is shown. The user draws **exactly two polygons** representing the two traffic lanes:
- Left-click to add points
- Right-click to close/finish a polygon
- After both polygons are drawn, press `q` to start processing

### 2. Vehicle Detection & Bounding Boxes
YOLOv8 runs on every frame and detects vehicles. For each detected vehicle:
- A **cyan bounding box** is drawn on screen
- A **unique tracking ID** is displayed above the box
- The centroid of the box is used to check which lane the vehicle belongs to

### 3. Speed Estimation
Speed is calculated from the centroid displacement over time:
```
speed (km/h) = (pixel_distance × meters_per_pixel) / time_elapsed × 3.6
```
Default scale: `0.08 meters/pixel`

### 4. Density Calculation
Per-frame occupancy = total bounding box area inside polygon / polygon area

### 5. Congestion Classification

| Density | Level |
|---------|-------|
| < 0.20 | 🟢 LOW |
| 0.20 – 0.60 | 🟡 MEDIUM |
| > 0.60 | 🔴 HIGH |

### 6. CSV Output
Every 10 seconds, per-lane metrics are written to `data.csv`:

```
video_name, row_type, interval_idx, lane_id, interval_count, cumulative_count, interval_density, interval_speed, Congestion level
```

---

##  How to Run

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (recommended) or CPU

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python test.py
```

> Update `VIDEO_PATH` and `MODEL_PATH` in `test.py` to point to your video and model file.

---

## 📁 Project Structure

```
project/
├── test.py               # Main script
├── dataset/
│   ├── train/            # Training images & labels
│   ├── valid/            # Validation images & labels
│   └── data.yaml         # Dataset config
├── requirements.txt
└── README.md
```

## Model Weights
The model weights are not included in this repo due to file size.
- Download YOLOv8l pretrained weights: [yolov8l.pt](https://github.com/ultralytics/assets/releases)
- Our custom trained weights are available on request — contact me at: itsreswant534@email.com


---

##  Output Sample

```
=== Interval 1 ===
Zone 1: Count=12, Density=0.341200, Speed=38.45 km/h, Congestion=1
Zone 2: Count=8,  Density=0.152300, Speed=54.10 km/h, Congestion=0

=== OVERALL FINAL STATS ===
Zone 1: Total unique vehicles=47, Avg Density=0.312000, Avg Speed=41.20 km/h, Overall Congestion=1
Zone 2: Total unique vehicles=35, Avg Density=0.189000, Avg Speed=58.30 km/h, Overall Congestion=0
```

---

##  Author

**Reswant Raja S**  


