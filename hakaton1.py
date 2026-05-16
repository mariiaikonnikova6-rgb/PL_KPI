from djitellopy import Tello
from ultralytics import YOLO
import cv2
import numpy as np
import time

# =========================
# CONFIG
# =========================

FRAME_W = 512
FRAME_H = 384

MODEL_PATH = "yolov8n-pose.pt"
DETECTION_INTERVAL = 5
DETECT_CONF = 0.25

LIGHT_THRESHOLD = 185
LIGHT_AREA_MIN = 40

# =========================
# POSE STRUCTURE
# =========================

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

# =========================
# TEMPORAL MEMORY (KEY FIX)
# =========================

prev_kps = {}
frame_memory = 3

HAND_BOOST = 1.4
HAND_THRESHOLD = 0.15
BODY_THRESHOLD = 0.25


def remember_kp(kp, idx, person_id):
    key = f"{person_id}_{idx}"

    conf = kp[idx][2]

    if conf > HAND_THRESHOLD:
        prev_kps[key] = (kp[idx][0], kp[idx][1], conf, frame_memory)

    if key in prev_kps:
        x, y, c, life = prev_kps[key]

        if life > 0:
            prev_kps[key] = (x, y, c, life - 1)
            return np.array([x, y, c])

    return kp[idx]


def limb_score(kp, indices, is_hand=False):

    scores = []

    for i in indices:

        x, y, c = kp[i]

        # FORCE PYTHON FLOAT (THIS FIXES YOUR CRASH)
        c = float(c)

        if is_hand:
            c = c * HAND_BOOST

        scores.append(c)

    return sum(scores) / len(scores)


def analyze_pose(person_kp, person_id=0):

    kp = person_kp

    limb_scores = {}

    for limb, idxs in LIMBS.items():

        recovered = []

        for i in idxs:
            recovered.append(remember_kp(kp, i, person_id))

        limb_scores[limb] = limb_score(
            recovered,
            list(range(len(recovered))),
            is_hand=(limb in ["LEFT_ARM", "RIGHT_ARM"])
        )

    avg = np.mean(list(limb_scores.values()))
    strong = sum(v > 0.25 for v in limb_scores.values())

    if avg > 0.38:
        return "PERSON", (0,255,0), limb_scores

    if strong >= 2:
        return "BODY FRAGMENT", (0,165,255), limb_scores

    if strong >= 1:
        return "POSSIBLE SURVIVOR", (0,0,255), limb_scores

    return None, (255,255,255), limb_scores


# =========================
# LIGHT DETECTION
# =========================

def analyze_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()

    _, thresh = cv2.threshold(gray, LIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for c in contours:
        if cv2.contourArea(c) < LIGHT_AREA_MIN:
            continue
        x,y,w,h = cv2.boundingRect(c)
        detections.append((x,y,w,h))

    return avg, detections


# =========================
# DRONE INIT
# =========================

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

tello.streamoff()
time.sleep(1)
tello.streamon()
time.sleep(3)

cap = tello.get_frame_read()

# =========================
# MODEL
# =========================

model = YOLO(MODEL_PATH)

cached = None
frame_i = 0
light_on = True

print("Running... Q quit | L toggle light")

# =========================
# MAIN LOOP
# =========================

while True:

    frame = cap.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # ---------------------
    # YOLO (throttled)
    # ---------------------

    if frame_i % DETECTION_INTERVAL == 0:
        cached = model(frame, classes=[0], conf=DETECT_CONF, verbose=False)

    frame_i += 1

    # ---------------------
    # LIGHT
    # ---------------------

    if light_on:
        avg, lights = analyze_light(frame)

        if avg < 40:
            txt, col = "LOW LIGHT", (0,0,255)
        elif avg > 200:
            txt, col = "OVEREXPOSED", (0,165,255)
        else:
            txt, col = f"LIGHT OK {int(avg)}", (0,255,0)

        cv2.putText(frame, txt, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        for x,y,w,h in lights:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

    # ---------------------
    # DETECTION DRAW
    # ---------------------

    if cached is not None:

        for r in cached:

            if r.boxes is None or r.keypoints is None:
                continue

            boxes = r.boxes
            kps = r.keypoints.data

            for i, kp in enumerate(kps):

                status, color, scores = analyze_pose(kp, i)

                if status is None:
                    continue

                x1,y1,x2,y2 = map(int, boxes[i].xyxy[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                cv2.putText(frame, status, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # skeleton
                for a,b in SKELETON:
                    if kp[a][2] > 0.15 and kp[b][2] > 0.15:
                        cv2.line(frame,
                                 (int(kp[a][0]), int(kp[a][1])),
                                 (int(kp[b][0]), int(kp[b][1])),
                                 (255,255,255), 2)

                # keypoints
                for j in kp:
                    if j[2] < 0.15:
                        continue
                    cv2.circle(frame,
                               (int(j[0]), int(j[1])),
                               4,
                               color,
                               -1)

    cv2.imshow("Rescue AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("l"):
        light_on = not light_on
        print("Light:", light_on)

tello.streamoff()
cv2.destroyAllWindows()
