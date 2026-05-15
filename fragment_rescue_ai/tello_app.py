import streamlit as st
import cv2
import pandas as pd
from datetime import datetime

from djitellopy import Tello
from ultralytics import YOLO

from detector.motion_detector import MotionDetector
from detector.threat_logic import calculate_threat_level
from detector.utils import draw_status_panel, save_detection_to_csv


# =============================
# Streamlit page settings
# =============================

st.set_page_config(
    page_title="Fragment Rescue AI — Tello Mode",
    page_icon="🚁",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: white;
    }

    .status-card {
        padding: 18px;
        border-radius: 16px;
        background: #111827;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }

    .big-title {
        font-size: 36px;
        font-weight: 800;
        color: #f9fafb;
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="big-title">🚁 Fragment Rescue AI — Tello Drone Mode</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Live drone video analysis using YOLO, pose estimation and OpenCV motion detection.</div>',
    unsafe_allow_html=True
)

st.divider()


# =============================
# Sidebar
# =============================

st.sidebar.header("Tello Drone Settings")

enable_detection = st.sidebar.checkbox("Enable YOLO person detection", True)
enable_pose = st.sidebar.checkbox("Enable pose analysis", True)
enable_motion = st.sidebar.checkbox("Enable motion detection", True)
show_debug = st.sidebar.checkbox("Show debug info", False)

person_model_name = st.sidebar.selectbox(
    "Person detection model",
    ["yolov8n.pt", "yolo11n.pt"],
    index=0
)

pose_model_name = st.sidebar.selectbox(
    "Pose model",
    ["yolov8n-pose.pt", "yolo11n-pose.pt"],
    index=0
)

frame_width = st.sidebar.slider("Frame width", 480, 1280, 960, 80)
frame_height = st.sidebar.slider("Frame height", 360, 720, 540, 60)

run_analysis = st.sidebar.button("Run Tello analysis")


# =============================
# Load YOLO models
# =============================

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


person_model = None
pose_model = None

if enable_detection:
    person_model = load_model(person_model_name)

if enable_pose:
    pose_model = load_model(pose_model_name)


# =============================
# Motion detector
# =============================

motion_detector = MotionDetector()


# =============================
# UI placeholders
# =============================

video_placeholder = st.empty()

col1, col2, col3 = st.columns(3)

with col1:
    status_box = st.empty()

with col2:
    probability_box = st.empty()

with col3:
    recommendation_box = st.empty()

reasoning_box = st.empty()
log_box = st.empty()

detection_log = []


# =============================
# Frame analysis function
# =============================

def analyze_frame(frame):
    """
    Analyze one frame from Tello:
    - YOLO person detection
    - YOLO pose keypoints
    - OpenCV motion detection
    - threat level calculation
    """

    person_detected = False
    person_confidence = 0.0
    keypoints_found = 0
    motion_detected = False
    occlusion_detected = False

    annotated_frame = frame.copy()

    # -----------------------------
    # 1. YOLO person detection
    # -----------------------------
    if enable_detection and person_model is not None:
        results = person_model(frame, verbose=False)

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = person
                if cls_id == 0:
                    person_detected = True
                    person_confidence = max(person_confidence, conf)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if conf >= 0.55:
                        color = (0, 255, 0)
                        label = f"Person {conf:.2f}"
                    else:
                        color = (0, 255, 255)
                        label = f"Partial person? {conf:.2f}"
                        occlusion_detected = True

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

    # -----------------------------
    # 2. YOLO pose estimation
    # -----------------------------
    if enable_pose and pose_model is not None:
        pose_results = pose_model(frame, verbose=False)

        for result in pose_results:
            if result.keypoints is None:
                continue

            if result.keypoints.xy is None:
                continue

            for person_kpts in result.keypoints.xy:
                visible_points = 0

                for point in person_kpts:
                    x = int(point[0])
                    y = int(point[1])

                    if x > 0 and y > 0:
                        visible_points += 1

                        cv2.circle(
                            annotated_frame,
                            (x, y),
                            4,
                            (255, 0, 255),
                            -1
                        )

                keypoints_found = max(keypoints_found, visible_points)

                if 3 <= visible_points <= 8:
                    occlusion_detected = True

                if visible_points >= 3:
                    cv2.putText(
                        annotated_frame,
                        f"Human keypoints: {visible_points}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2
                    )

    # -----------------------------
    # 3. Motion detection
    # -----------------------------
    if enable_motion:
        motion_mask, motion_boxes = motion_detector.detect(frame)

        if len(motion_boxes) > 0:
            motion_detected = True

        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + w, y + h),
                (0, 165, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                "Motion",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

    # -----------------------------
    # 4. Threat logic
    # -----------------------------
    threat_level, human_probability, explanation, recommendation, status = calculate_threat_level(
        person_detected=person_detected,
        person_confidence=person_confidence,
        keypoints_found=keypoints_found,
        motion_detected=motion_detected,
        occlusion_detected=occlusion_detected
    )

    # -----------------------------
    # 5. Draw overlay panel
    # -----------------------------
    annotated_frame = draw_status_panel(
        annotated_frame,
        threat_level=threat_level,
        human_probability=human_probability,
        status=status,
        recommendation=recommendation
    )

    return {
        "frame": annotated_frame,
        "threat_level": threat_level,
        "human_probability": human_probability,
        "explanation": explanation,
        "recommendation": recommendation,
        "status": status,
        "person_detected": person_detected,
        "person_confidence": person_confidence,
        "keypoints_found": keypoints_found,
        "motion_detected": motion_detected,
        "occlusion_detected": occlusion_detected
    }


# =============================
# Main Tello mode
# =============================

if not run_analysis:
    st.info("Підключи ноутбук до Wi-Fi Tello, потім натисни Run Tello analysis.")
    st.markdown(
        """
        ### Як запустити

        1. Увімкни Tello.
        2. Підключи ноутбук до Wi-Fi мережі Tello.
        3. Переконайся, що ти знаходишся у папці проєкту.
        4. Запусти:

        ```bash
        streamlit run tello_app.py
        ```

        5. Натисни **Run Tello analysis**.
        """
    )

else:
    tello = None

    try:
        st.sidebar.info("Connecting to Tello...")

        tello = Tello()
        tello.connect()

        battery = tello.get_battery()
        st.sidebar.success(f"Tello connected. Battery: {battery}%")

        tello.streamoff()
        tello.streamon()

        frame_reader = tello.get_frame_read()

        st.success("Tello video stream started.")

        while True:
            frame = frame_reader.frame

            if frame is None:
                st.warning("No frame received from Tello.")
                continue

            frame = cv2.resize(frame, (frame_width, frame_height))

            result = analyze_frame(frame)

            annotated_frame = result["frame"]

            video_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            threat_level = result["threat_level"]
            human_probability = result["human_probability"]
            explanation = result["explanation"]
            recommendation = result["recommendation"]
            status = result["status"]

            status_box.markdown(
                f"""
                <div class="status-card">
                    <h3>Current status</h3>
                    <h2>{threat_level}</h2>
                    <p>{status}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            probability_box.markdown(
                f"""
                <div class="status-card">
                    <h3>Human probability</h3>
                    <h2>{human_probability:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            recommendation_box.markdown(
                f"""
                <div class="status-card">
                    <h3>Recommendation</h3>
                    <h2>{recommendation}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            reasoning_text = "\n".join([f"- ✓ {item}" for item in explanation])

            reasoning_box.markdown(
                f"""
                ### AI Reasoning Panel

                {reasoning_text}
                """
            )

            now = datetime.now().strftime("%H:%M:%S")

            if threat_level != "Green":
                log_item = {
                    "timestamp": now,
                    "threat_level": threat_level,
                    "confidence": round(human_probability, 2),
                    "status": status,
                    "explanation": "; ".join(explanation),
                    "recommendation": recommendation
                }

                detection_log.append(log_item)

                save_detection_to_csv(
                    timestamp=now,
                    threat_level=threat_level,
                    confidence=human_probability,
                    status=status,
                    explanation="; ".join(explanation),
                    recommendation=recommendation
                )

            if len(detection_log) > 0:
                log_df = pd.DataFrame(detection_log[-20:])
                log_box.dataframe(log_df, use_container_width=True)

            if show_debug:
                st.sidebar.write(
                    {
                        "person_detected": result["person_detected"],
                        "person_confidence": result["person_confidence"],
                        "keypoints_found": result["keypoints_found"],
                        "motion_detected": result["motion_detected"],
                        "occlusion_detected": result["occlusion_detected"]
                    }
                )

    except Exception as e:
        st.error(f"Tello error: {e}")

    finally:
        if tello is not None:
            try:
                tello.streamoff()
                tello.end()
            except Exception:
                pass