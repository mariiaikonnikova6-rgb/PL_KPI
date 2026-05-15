"""
Global configuration for Fragment Rescue AI.
Change the values here when you want to connect a real drone stream,
scrcpy screen capture region, or different YOLO models.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"

REPORT_CSV_PATH = REPORTS_DIR / "detections.csv"
DEMO_VIDEO_PATH = ASSETS_DIR / "demo_video.mp4"

# Ultralytics models. YOLO11 is fast and widely available in ultralytics.
# If your installed ultralytics version supports newer YOLO models, you can replace these names.
DETECTION_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

# Detection thresholds.
PERSON_CONF_THRESHOLD = 0.45       # confident full person detection
LOW_PERSON_CONF_THRESHOLD = 0.20   # weak/partial person candidate
KEYPOINT_CONF_THRESHOLD = 0.25
MIN_KEYPOINTS_FOR_FRAGMENT = 2
MIN_KEYPOINTS_FOR_STRONG_FRAGMENT = 4

# Motion detection.
MOTION_MIN_AREA = 750
MOTION_HISTORY = 350
MOTION_VAR_THRESHOLD = 32

# Video processing.
DEFAULT_FRAME_WIDTH = 960
DEFAULT_MAX_FRAMES = 600
DEFAULT_FRAME_DELAY = 0.03

# Replace this value with your drone IP camera / RTSP / HTTP stream.
# Examples:
#   DRONE_STREAM_URL = "rtsp://192.168.0.10:554/live"
#   DRONE_STREAM_URL = "http://192.168.0.10:8080/video"
DRONE_STREAM_URL = "rtsp://YOUR_DRONE_CAMERA_IP_OR_PHONE_STREAM_HERE"

# For scrcpy workflow: Drone -> phone app -> scrcpy window -> laptop screen capture.
# Start scrcpy separately, then select Screen capture in Streamlit.
# You can tune this region from the Streamlit sidebar.
SCRCPY_SCREEN_REGION = {
    "left": 0,
    "top": 0,
    "width": 1280,
    "height": 720,
}
