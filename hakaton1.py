from djitellopy import Tello
from ultralytics import YOLO
import cv2
import numpy as np
import time

# =========================================================
# CONFIG
# =========================================================

FRAME_W = 512
FRAME_H = 384

MODEL_PATH = "yolov8n-pose.pt"

DETECTION_INTERVAL = 5

DETECT_CONF = 0.25
KP_CONF = 0.25

LIGHT_THRESHOLD = 185
LIGHT_AREA_MIN = 40

# =========================================================
# POSE STRUCTURE
# =========================================================

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),

    (5, 6),

    (5, 11), (6, 12),
    (11, 12),

    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

LIMBS = {
    "HEAD": [0,1,2,3,4],
    "LEFT_ARM": [5,7,9],
    "RIGHT_ARM": [6,8,10],
    "TORSO": [5,6,11,12],
    "LEFT_LEG": [11,13,15],
    "RIGHT_LEG": [12,14,16]
}

# =========================================================
# DRONE SETUP
# =========================================================

print("Connecting to Tello...")

tello = Tello()

tello.connect()

time.sleep(2)

battery = tello.get_battery()

print(f"Battery: {battery}%")

print("Resetting stream...")

tello.streamoff()
time.sleep(1)

tello.streamon()
time.sleep(3)

frame_read = tello.get_frame_read()

# Flush startup frames
for _ in range(30):
    _ = frame_read.frame
    time.sleep(0.03)

print("Drone stream ready")

# =========================================================
# MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded")

# =========================================================
# CACHE
# =========================================================

cached_results = None
frame_count = 0

light_detection_on = True

# =========================================================
# BODY ANALYSIS
# =========================================================

def limb_visible(kp, indices):

    visible = 0

    for idx in indices:
        if kp[idx][2] > KP_CONF:
            visible += 1

    return visible / len(indices)


def analyze_pose(person_kp):

    limb_scores = {}

    for limb_name, indices in LIMBS.items():
        limb_scores[limb_name] = limb_visible(person_kp, indices)

    strong_limbs = sum(score >= 0.5 for score in limb_scores.values())

    avg_score = np.mean(list(limb_scores.values()))

    if avg_score > 0.55:
        status = "PERSON"
        color = (0, 255, 0)

    elif strong_limbs >= 2:
        status = "BODY FRAGMENT"
        color = (0, 165, 255)

    elif strong_limbs >= 1:
        status = "POSSIBLE SURVIVOR"
        color = (0, 0, 255)

    else:
        return None

    return {
        "status": status,
        "color": color,
        "scores": limb_scores
    }

# =========================================================
# LIGHT ANALYSIS
# =========================================================

def analyze_light(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    avg = gray.mean()

    _, thresh = cv2.threshold(
        gray,
        LIGHT_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3,3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < LIGHT_AREA_MIN:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        detections.append((x, y, w, h))

    return avg, detections

# =========================================================
# MAIN LOOP
# =========================================================

print("Controls:")
print("Q = Quit")
print("L = Toggle light detection")

fps_timer = time.time()
fps_counter = 0
fps = 0

try:

    while True:

        frame = frame_read.frame

        if frame is None:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # =================================================
        # YOLO INFERENCE
        # =================================================

        if frame_count % DETECTION_INTERVAL == 0:

            try:

                cached_results = model(
                    frame,
                    classes=[0],
                    conf=DETECT_CONF,
                    verbose=False
                )

            except Exception as e:
                print("Inference error:", e)

        frame_count += 1

        # =================================================
        # LIGHT DETECTION
        # =================================================

        if light_detection_on:

            avg_light, lights = analyze_light(frame)

            if avg_light < 40:
                txt = "LOW LIGHT"
                clr = (0, 0, 255)

            elif avg_light > 200:
                txt = "OVEREXPOSED"
                clr = (0, 165, 255)

            else:
                txt = f"LIGHT OK {int(avg_light)}"
                clr = (0, 255, 0)

            cv2.putText(
                frame,
                txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                clr,
                2
            )

            for (x, y, w, h) in lights:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "LIGHT",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

        # =================================================
        # DRAW DETECTIONS
        # =================================================

        if cached_results is not None:

            for r in cached_results:

                if r.boxes is None or r.keypoints is None:
                    continue

                boxes = r.boxes
                keypoints = r.keypoints.data

                for i, person_kp in enumerate(keypoints):

                    analysis = analyze_pose(person_kp)

                    if analysis is None:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        boxes[i].xyxy[0]
                    )

                    color = analysis["color"]

                    # BOX
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    # STATUS
                    cv2.putText(
                        frame,
                        analysis["status"],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2
                    )

                    # SKELETON
                    for a, b in SKELETON:

                        kp_a = person_kp[a]
                        kp_b = person_kp[b]

                        if kp_a[2] > KP_CONF and kp_b[2] > KP_CONF:

                            pt1 = (
                                int(kp_a[0]),
                                int(kp_a[1])
                            )

                            pt2 = (
                                int(kp_b[0]),
                                int(kp_b[1])
                            )

                            cv2.line(
                                frame,
                                pt1,
                                pt2,
                                (255, 255, 255),
                                2
                            )

                    # KEYPOINTS
                    for kp in person_kp:

                        x, y, c = kp

                        if c < KP_CONF:
                            continue

                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            4,
                            color,
                            -1
                        )

        # =================================================
        # FPS COUNTER
        # =================================================

        fps_counter += 1

        if time.time() - fps_timer >= 1:

            fps = fps_counter

            fps_counter = 0

            fps_timer = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, FRAME_H - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # =================================================
        # UI
        # =================================================

        cv2.putText(
            frame,
            "GREEN = PERSON",
            (10, FRAME_H - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        cv2.putText(
            frame,
            "ORANGE = BODY FRAGMENT",
            (10, FRAME_H - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1
        )

        cv2.putText(
            frame,
            "RED = POSSIBLE SURVIVOR",
            (10, FRAME_H - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

        cv2.imshow("Rescue AI", frame)

        # =================================================
        # KEYS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("l"):

            light_detection_on = not light_detection_on

            print(
                "Light detection:",
                "ON" if light_detection_on else "OFF"
            )

except KeyboardInterrupt:
    pass

except Exception as e:
    print("Fatal error:", e)

finally:

    print("Shutting down...")

    tello.streamoff()

    cv2.destroyAllWindows()
