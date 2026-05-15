from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_FRAME_DELAY,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MAX_FRAMES,
    DEMO_VIDEO_PATH,
    DETECTION_MODEL,
    DRONE_STREAM_URL,
    KEYPOINT_CONF_THRESHOLD,
    LOW_PERSON_CONF_THRESHOLD,
    MIN_KEYPOINTS_FOR_FRAGMENT,
    MOTION_HISTORY,
    MOTION_MIN_AREA,
    MOTION_VAR_THRESHOLD,
    PERSON_CONF_THRESHOLD,
    POSE_MODEL,
    REPORT_CSV_PATH,
    SCRCPY_SCREEN_REGION,
)
from detector.motion_detector import MotionDetector
from detector.pose_detector import YOLOPoseDetector
from detector.threat_logic import THREAT_COLORS, calculate_threat_level
from detector.utils import (
    MSSScreenCapture,
    append_detection_report,
    create_event_from_result,
    detect_silhouette_fallback,
    draw_keypoints,
    draw_motion_boxes,
    draw_person_boxes,
    draw_status_overlay,
    ensure_report_file,
    motion_near_human_sign,
    resize_frame,
)
from detector.yolo_detector import HOGFallbackPersonDetector, YOLOPersonDetector


st.set_page_config(
    page_title="Fragment Rescue AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #18243a 0, #090d14 42%, #05070b 100%);
        color: #f5f7fb;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #aab4c8;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.055);
        border-radius: 22px;
        padding: 18px 20px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        min-height: 112px;
    }
    .metric-label {
        color: #aab4c8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 850;
        line-height: 1.1;
    }
    .reasoning-box {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(8,12,20,0.72);
        border-radius: 22px;
        padding: 16px 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.18);
    }
    .small-note {
        color: #9aa7bd;
        font-size: 0.88rem;
    }
    div[data-testid="stSidebar"] {
        background: rgba(5, 8, 14, 0.96);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(135deg, #2c72ff, #7f52ff);
        color: white;
        font-weight: 800;
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_yolo_detector(model_path: str, conf: float, low_conf: float) -> YOLOPersonDetector:
    return YOLOPersonDetector(model_path=model_path, conf_threshold=conf, low_conf_threshold=low_conf)


@st.cache_resource(show_spinner=False)
def load_pose_detector(model_path: str, kp_conf: float) -> YOLOPoseDetector:
    return YOLOPoseDetector(model_path=model_path, keypoint_conf_threshold=kp_conf)


@st.cache_resource(show_spinner=False)
def load_hog_fallback() -> HOGFallbackPersonDetector:
    return HOGFallbackPersonDetector()


def init_state() -> None:
    if "detection_log" not in st.session_state:
        st.session_state.detection_log = []
    if "last_logged_status" not in st.session_state:
        st.session_state.last_logged_status = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def metric_card(label: str, value: str, color: str = "#f5f7fb") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """


def create_capture_source(source_type: str, uploaded_file: Any, ip_url: str, screen_region: Dict[str, int]) -> Tuple[Any, str]:
    """Return an OpenCV-like capture object with read()/release()."""
    if source_type == "Demo video":
        return cv2.VideoCapture(str(DEMO_VIDEO_PATH)), str(DEMO_VIDEO_PATH)

    if source_type == "Webcam":
        return cv2.VideoCapture(0), "Webcam index 0"

    if source_type == "Video file":
        if uploaded_file is None:
            raise RuntimeError("Upload a video file first, or choose Demo video/Webcam.")
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.close()
        return cv2.VideoCapture(tmp.name), tmp.name

    if source_type == "IP camera / drone URL":
        if not ip_url or "YOUR_DRONE" in ip_url:
            raise RuntimeError("Replace the IP camera URL in the sidebar or in config.py: DRONE_STREAM_URL.")
        return cv2.VideoCapture(ip_url), ip_url

    if source_type == "Screen capture / scrcpy":
        return MSSScreenCapture(screen_region), "Screen capture region"

    raise RuntimeError(f"Unknown source type: {source_type}")


def process_frame(
    frame: np.ndarray,
    yolo_detector: YOLOPersonDetector | None,
    pose_detector: YOLOPoseDetector | None,
    motion_detector: MotionDetector | None,
    use_yolo: bool,
    use_pose: bool,
    use_motion: bool,
    use_hog_fallback: bool,
    show_debug: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    frame = resize_frame(frame, DEFAULT_FRAME_WIDTH)
    annotated = frame.copy()

    # 1) YOLO full person detection.
    yolo_data = {
        "available": False,
        "error": "YOLO detection disabled.",
        "person_detected": False,
        "best_person_confidence": 0.0,
        "person_boxes": [],
        "weak_person_boxes": [],
    }
    if use_yolo and yolo_detector is not None:
        yolo_data = yolo_detector.detect(frame)

    # Optional classical fallback if YOLO cannot load.
    if use_hog_fallback and (not yolo_data.get("available")):
        hog = load_hog_fallback()
        yolo_data = hog.detect(frame)
        yolo_data["error"] = "Using OpenCV HOG fallback because YOLO detection is unavailable."

    # 2) YOLO pose keypoints.
    pose_data = {
        "available": False,
        "error": "Pose analysis disabled.",
        "keypoints": [],
        "keypoint_count": 0,
        "keypoint_names": [],
        "pose_boxes": [],
        "partial_pose_detected": False,
    }
    if use_pose and pose_detector is not None:
        pose_data = pose_detector.detect(frame)

    # 3) OpenCV motion.
    motion_data = {
        "motion_detected": False,
        "motion_boxes": [],
        "motion_area": 0,
        "motion_score": 0.0,
        "mask": None,
    }
    if use_motion and motion_detector is not None:
        motion_data = motion_detector.detect(frame)

    all_human_boxes = yolo_data.get("person_boxes", []) + yolo_data.get("weak_person_boxes", []) + pose_data.get("pose_boxes", [])
    near_motion = motion_near_human_sign(
        keypoints=pose_data.get("keypoints", []),
        human_boxes=all_human_boxes,
        motion_boxes=motion_data.get("motion_boxes", []),
    )

    weak_candidate = len(yolo_data.get("weak_person_boxes", [])) > 0
    occlusion_detected = (
        (pose_data.get("keypoint_count", 0) >= MIN_KEYPOINTS_FOR_FRAGMENT and not yolo_data.get("person_detected", False))
        or (weak_candidate and not yolo_data.get("person_detected", False))
    )
    silhouette = detect_silhouette_fallback(frame, motion_data.get("motion_boxes", [])) if use_motion else False

    threat = calculate_threat_level(
        person_detected=bool(yolo_data.get("person_detected", False)),
        person_confidence=float(yolo_data.get("best_person_confidence", 0.0)),
        keypoints_found=int(pose_data.get("keypoint_count", 0)),
        motion_detected=bool(motion_data.get("motion_detected", False)),
        occlusion_detected=bool(occlusion_detected),
        weak_person_candidate=bool(weak_candidate),
        motion_near_human_sign=bool(near_motion),
        keypoint_names=pose_data.get("keypoint_names", []),
        silhouette_detected=bool(silhouette),
    ).to_dict()

    # Draw visual layers.
    draw_person_boxes(annotated, yolo_data.get("person_boxes", []), partial=False)
    draw_person_boxes(annotated, yolo_data.get("weak_person_boxes", []), partial=True)
    draw_keypoints(annotated, pose_data.get("keypoints", []))
    if use_motion:
        draw_motion_boxes(annotated, motion_data.get("motion_boxes", []))
    draw_status_overlay(annotated, threat)

    debug = {
        "yolo": yolo_data,
        "pose": {k: v for k, v in pose_data.items() if k != "keypoints"},
        "motion": {k: v for k, v in motion_data.items() if k != "mask"},
        "motion_near_human_sign": near_motion,
        "occlusion_detected": occlusion_detected,
        "silhouette_detected": silhouette,
    }
    threat["debug"] = debug if show_debug else {}
    return annotated, threat


def maybe_log_event(result: Dict[str, Any]) -> None:
    level = result.get("threat_level", "Green")
    status = result.get("status", "")
    probability = result.get("human_probability", 0)

    # Keep log readable: always log non-green changes, and log green only when status changes.
    signature = f"{level}-{status}-{probability // 10}"
    if signature != st.session_state.last_logged_status and (level != "Green" or st.session_state.last_logged_status is None):
        event = create_event_from_result(result)
        st.session_state.detection_log.insert(0, event)
        st.session_state.detection_log = st.session_state.detection_log[:80]
        st.session_state.last_logged_status = signature


def render_reasoning(result: Dict[str, Any]) -> None:
    explanation = result.get("explanation", [])
    recommendation = result.get("recommendation", "Monitor this area")
    level = result.get("threat_level", "Green")
    color = THREAT_COLORS.get(level, "#2ecc71")

    lines = "".join([f"<li>{item}</li>" for item in explanation])
    st.markdown(
        f"""
        <div class="reasoning-box">
            <h3 style="margin-top:0;color:{color};">AI Reasoning Panel</h3>
            <ul>{lines}</ul>
            <p><b>Recommendation:</b> {recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    ensure_report_file(REPORT_CSV_PATH)

    st.markdown('<div class="main-title">🚁 Fragment Rescue AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Drone-based computer vision MVP for detecting full people, hidden human fragments, pose keypoints, and suspicious movement in disaster scenes.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Control panel")
        source_type = st.radio(
            "Video source",
            ["Demo video", "Webcam", "Video file", "IP camera / drone URL", "Screen capture / scrcpy"],
            index=0,
        )

        uploaded_file = None
        if source_type == "Video file":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

        ip_url = DRONE_STREAM_URL
        if source_type == "IP camera / drone URL":
            ip_url = st.text_input("Drone/IP camera URL", value=DRONE_STREAM_URL)
            st.caption("Place for drone stream: RTSP/HTTP URL. You can also edit DRONE_STREAM_URL in config.py.")

        screen_region = dict(SCRCPY_SCREEN_REGION)
        if source_type == "Screen capture / scrcpy":
            st.caption("Open scrcpy first, show the drone camera on your phone, then capture the screen region.")
            c1, c2 = st.columns(2)
            with c1:
                screen_region["left"] = st.number_input("left", min_value=0, value=int(screen_region["left"]), step=10)
                screen_region["width"] = st.number_input("width", min_value=100, value=int(screen_region["width"]), step=10)
            with c2:
                screen_region["top"] = st.number_input("top", min_value=0, value=int(screen_region["top"]), step=10)
                screen_region["height"] = st.number_input("height", min_value=100, value=int(screen_region["height"]), step=10)

        st.divider()
        use_yolo = st.toggle("Enable YOLO person detection", value=True)
        use_pose = st.toggle("Enable pose analysis", value=True)
        use_motion = st.toggle("Enable motion detection", value=True)
        use_hog_fallback = st.toggle("Use OpenCV HOG fallback if YOLO fails", value=True)
        show_debug = st.toggle("Show debug info", value=False)

        st.divider()
        person_conf = st.slider("Person confidence threshold", 0.10, 0.90, float(PERSON_CONF_THRESHOLD), 0.05)
        low_person_conf = st.slider("Weak person threshold", 0.05, 0.60, float(LOW_PERSON_CONF_THRESHOLD), 0.05)
        keypoint_conf = st.slider("Keypoint confidence threshold", 0.05, 0.90, float(KEYPOINT_CONF_THRESHOLD), 0.05)
        max_frames = st.slider("Max frames per run", 30, 3000, int(DEFAULT_MAX_FRAMES), 30)
        frame_delay = st.slider("Frame delay", 0.00, 0.20, float(DEFAULT_FRAME_DELAY), 0.01)

        run_analysis = st.button("▶ Run analysis", use_container_width=True)
        clear_log = st.button("Clear detection log", use_container_width=True)
        if clear_log:
            st.session_state.detection_log = []
            st.session_state.last_logged_status = None
            st.rerun()

    status_col, prob_col, rec_col, source_col = st.columns(4)
    video_col, side_col = st.columns([1.65, 1.0])

    with status_col:
        st.markdown(metric_card("Current status", "Waiting", "#9fb2d7"), unsafe_allow_html=True)
    with prob_col:
        st.markdown(metric_card("Human probability", "0%", "#9fb2d7"), unsafe_allow_html=True)
    with rec_col:
        st.markdown(metric_card("Recommendation", "Start analysis", "#9fb2d7"), unsafe_allow_html=True)
    with source_col:
        st.markdown(metric_card("Source", source_type, "#9fb2d7"), unsafe_allow_html=True)

    status_placeholder = st.empty()
    video_placeholder = video_col.empty()
    reasoning_placeholder = side_col.empty()
    log_placeholder = side_col.empty()
    debug_placeholder = st.empty()

    if not DEMO_VIDEO_PATH.exists():
        st.warning("assets/demo_video.mp4 not found. Webcam/video upload will still work.")

    if run_analysis:
        with st.spinner("Loading models and opening video source..."):
            yolo_detector = load_yolo_detector(DETECTION_MODEL, person_conf, low_person_conf) if use_yolo else None
            pose_detector = load_pose_detector(POSE_MODEL, keypoint_conf) if use_pose else None
            motion_detector = MotionDetector(MOTION_MIN_AREA, MOTION_HISTORY, MOTION_VAR_THRESHOLD) if use_motion else None

            try:
                capture, source_label = create_capture_source(source_type, uploaded_file, ip_url, screen_region)
            except Exception as exc:
                st.error(str(exc))
                return

        frame_idx = 0
        last_result: Dict[str, Any] | None = None

        while frame_idx < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            annotated, result = process_frame(
                frame=frame,
                yolo_detector=yolo_detector,
                pose_detector=pose_detector,
                motion_detector=motion_detector,
                use_yolo=use_yolo,
                use_pose=use_pose,
                use_motion=use_motion,
                use_hog_fallback=use_hog_fallback,
                show_debug=show_debug,
            )
            last_result = result
            st.session_state.last_result = result
            maybe_log_event(result)

            color = THREAT_COLORS.get(result["threat_level"], "#2ecc71")
            with status_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(metric_card("Current status", result["threat_level"], color), unsafe_allow_html=True)
                with c2:
                    st.markdown(metric_card("Human probability", f'{result["human_probability"]}%', color), unsafe_allow_html=True)
                with c3:
                    st.markdown(metric_card("Recommendation", result["recommendation"], color), unsafe_allow_html=True)
                with c4:
                    st.markdown(metric_card("Source", source_label, "#9fb2d7"), unsafe_allow_html=True)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            with reasoning_placeholder.container():
                render_reasoning(result)

            with log_placeholder.container():
                st.subheader("Detection log")
                if st.session_state.detection_log:
                    st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True, hide_index=True)
                else:
                    st.caption("No detection events yet.")

            if show_debug:
                debug_placeholder.json(result.get("debug", {}))

            frame_idx += 1
            if frame_delay > 0:
                time.sleep(frame_delay)

        try:
            capture.release()
        except Exception:
            pass

        if last_result is None:
            st.warning("No frames were read from the selected source.")
        else:
            st.success("Analysis finished for the selected frame limit/source.")

    st.divider()
    report_col, download_col = st.columns([1, 1])
    with report_col:
        if st.button("💾 Save current log to CSV", use_container_width=True):
            if st.session_state.detection_log:
                for event in reversed(st.session_state.detection_log):
                    append_detection_report(REPORT_CSV_PATH, event)
                st.success(f"Saved to {REPORT_CSV_PATH}")
            else:
                st.info("Detection log is empty.")

    with download_col:
        if st.session_state.detection_log:
            csv_data = pd.DataFrame(st.session_state.detection_log).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download CSV report",
                data=csv_data,
                file_name="fragment_rescue_ai_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇ Download CSV report", disabled=True, use_container_width=True)

    st.markdown(
        """
        <p class="small-note">
        MVP note: this project is designed for hackathon demonstration and decision support prototyping.
        It is not a certified rescue, medical, or safety-critical system.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
