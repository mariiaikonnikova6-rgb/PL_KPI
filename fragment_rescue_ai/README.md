# Fragment Rescue AI

## Short description

**Fragment Rescue AI** is a hackathon MVP for disaster response. It analyzes video from a webcam, video file, screen capture, or future drone stream and helps rescuers notice not only fully visible people, but also partially hidden human signs: pose keypoints, weak person detections, human-like silhouettes, and motion.

> MVP safety note: this is a prototype for demonstration and decision-support research. It is not a certified rescue or medical system.

## Problem

After earthquakes, explosions, fires, floods, or building collapses, a victim may be partially hidden behind debris, furniture, blankets, boxes, smoke, darkness, or camera angle limitations. A normal `person` detector can miss the person if the full body is not visible.

## Solution

Fragment Rescue AI combines:

- **YOLO person detection** for full or mostly visible humans;
- **YOLO pose estimation** for partial body evidence such as wrists, elbows, shoulders, head, knees, and ankles;
- **OpenCV motion detection** to detect suspicious movement;
- **heuristic rescue reasoning** to calculate threat level and human probability;
- **Streamlit dashboard** for a clear, judge-friendly live demo.

The system explains why it made a decision, for example:

- Human fragment detected;
- Motion confirmed;
- Full body appears occluded;
- Rescue priority: HIGH.

## Features

- Webcam input;
- Video file input;
- Demo video mode;
- Placeholder for drone IP camera / RTSP / HTTP stream;
- Screen capture mode for workflows like `scrcpy`;
- YOLO person detection;
- YOLO pose keypoint analysis;
- OpenCV motion detection;
- Threat levels: Green, Yellow, Orange, Red;
- Human probability score from 0 to 100%;
- AI reasoning panel;
- Detection log;
- CSV report export;
- Modern dark Streamlit UI.

## How it works

The pipeline is:

1. Read a frame from the selected source.
2. Run YOLO person detection.
3. Run YOLO pose estimation.
4. Run OpenCV motion detection.
5. Combine the evidence:
   - person confidence;
   - number of visible keypoints;
   - motion near suspicious human signs;
   - partial visibility / occlusion pattern.
6. Calculate human probability and threat level.
7. Draw boxes, keypoints, motion areas, and recommendation on the frame.
8. Show reasoning and log events in the dashboard.

## Installation

```bash
cd fragment_rescue_ai
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The first YOLO run may download model files such as `yolo11n.pt` and `yolo11n-pose.pt`.

## How to run

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## Demo scenario

### Option 1: Built-in demo video

1. Run the app.
2. Select **Demo video**.
3. Click **Run analysis**.
4. Show the jury the video overlay, status cards, AI reasoning panel, and detection log.

### Option 2: Webcam

1. Select **Webcam**.
2. Put a person partially behind a chair, blanket, box, or table.
3. Enable YOLO, pose analysis, and motion detection.
4. Move a hand or head slightly.
5. Explain how motion + partial human evidence increases rescue priority.

### Option 3: Video file

1. Select **Video file**.
2. Upload an `.mp4`, `.avi`, `.mov`, or `.mkv` file.
3. Click **Run analysis**.

### Option 4: Drone/IP camera stream

Edit `config.py`:

```python
DRONE_STREAM_URL = "rtsp://YOUR_DRONE_CAMERA_IP_OR_PHONE_STREAM_HERE"
```

or paste the URL directly in the Streamlit sidebar.

Examples:

```python
DRONE_STREAM_URL = "rtsp://192.168.0.10:554/live"
DRONE_STREAM_URL = "http://192.168.0.10:8080/video"
```

### Option 5: scrcpy screen capture

This workflow is useful when the drone camera is visible only inside a phone app.

1. Connect the phone by USB.
2. Start screen mirroring:

```bash
scrcpy --window-title scrcpy
```

3. Open the drone camera app on the phone.
4. In Fragment Rescue AI select **Screen capture / scrcpy**.
5. Adjust `left`, `top`, `width`, and `height` so the capture region covers the scrcpy window.
6. Click **Run analysis**.

## Technology stack

- Python;
- OpenCV;
- Ultralytics YOLO;
- YOLO pose estimation;
- NumPy;
- Pandas;
- Streamlit;
- MSS for optional screen capture.

## Future improvements

- Thermal camera support;
- GPS mapping;
- Automatic area scanning;
- Rescue priority map;
- Cloud dashboard;
- Offline mode;
- Voice alerts;
- Real drone SDK integration;
- Segmentation model;
- Custom disaster-scene dataset training;
- Human body-part detector trained on occlusion-heavy data;
- Multi-frame tracking and survivor re-identification;
- Edge deployment on Jetson / Raspberry Pi / drone onboard computer.
