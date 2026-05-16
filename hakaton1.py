from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

model = YOLO("yolov8n-pose.pt")

cap = tello.get_frame_read()

KEYPOINT_NAMES = {
    0: "Nose",        1: "Left Eye",      2: "Right Eye",
    3: "Left Ear",    4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder",
    7: "Left Elbow",  8: "Right Elbow",
    9: "Left Wrist",  10: "Right Wrist",
    11: "Left Hip",   12: "Right Hip",
    13: "Left Knee",  14: "Right Knee",
    15: "Left Ankle", 16: "Right Ankle",
}

KEYPOINT_COLORS = {
    0:  (255, 255, 0),   1:  (255, 255, 0),   2:  (255, 255, 0),
    3:  (255, 255, 0),   4:  (255, 255, 0),
    5:  (0, 165, 255),   6:  (0, 165, 255),
    7:  (0, 255, 255),   8:  (0, 255, 255),
    9:  (0, 255, 0),     10: (0, 255, 0),
    11: (255, 0, 255),   12: (255, 0, 255),
    13: (0, 0, 255),     14: (0, 0, 255),
    15: (255, 165, 0),   16: (255, 165, 0),
}

SKELETON = [
    (0, 1),  (0, 2),  (1, 3),  (2, 4),
    (5, 6),  (5, 7),  (7, 9),  (6, 8),  (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13),(13, 15),(12, 14),(14, 16),
]

CONF = 0.4


def analyze_body(keypoints, i):
    if keypoints is None or len(keypoints.data) == 0 or i >= len(keypoints.data):
        return None, [], (255, 255, 255), 0

    kp = keypoints.data[i]

    parts = {
        "Head":      [0, 1, 2, 3, 4],
        "Shoulders": [5, 6],
        "Arms":      [7, 8, 9, 10],
        "Torso":     [11, 12],
        "Legs":      [13, 14, 15, 16],
    }

    visible = []
    for name, indices in parts.items():
        count = sum(1 for idx in indices if kp[idx][2] > CONF)
        ratio = count / len(indices)
        if ratio >= 0.5:
            visible.append((name, ratio))

    total = sum(1 for p in kp if p[2] > CONF)
    full_ratio = total / len(kp)

    if full_ratio >= 0.6:
        status = "PERSON"
        color = (0, 255, 0)
    elif len(visible) >= 2:
        status = "FRAGMENT"
        color = (0, 165, 255)
    elif len(visible) == 1:
        status = "HIDDEN SURVIVOR?"
        color = (0, 0, 255)
    else:
        return None, [], (255, 255, 255), 0

    return status, visible, color, int(full_ratio * 100)


def analyze_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = gray.mean()

    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_pixels = cv2.countNonZero(bright_mask)
    bright_ratio = bright_pixels / (frame.shape[0] * frame.shape[1])

    if avg_brightness < 40:
        light_status = "DARK - LOW VISIBILITY"
        light_color = (0, 0, 255)
    elif avg_brightness > 200:
        light_status = "OVEREXPOSED"
        light_color = (0, 165, 255)
    else:
        light_status = f"LIGHT OK  avg={int(avg_brightness)}"
        light_color = (0, 255, 0)

    signal_detected = bright_ratio > 0.015 and avg_brightness < 100

    return light_status, light_color, signal_detected, bright_mask


print("Stream starting... press Q to quit")

while True:
    frame = cap.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    light_status, light_color, signal_detected, bright_mask = analyze_light(frame)

    cv2.putText(frame, light_status,
                (frame.shape[1] - 310, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)

    if signal_detected:
        cv2.putText(frame, ">>> SIGNAL LIGHT DETECTED <<<",
                    (frame.shape[1] // 2 - 185, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) > 25:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "LIGHT",
                            (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    results = model(frame, classes=[0], verbose=False)

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            try:
                status, visible_parts, color, body_pct = analyze_body(keypoints, i)
            except Exception:
                status, visible_parts, color, body_pct = "PERSON", [], (0, 255, 0), int(conf * 100)

            if status is None:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"{status} {conf:.0%}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if visible_parts:
                parts_text = " | ".join(
                    f"{name} {int(r * 100)}%" for name, r in visible_parts
                )
                cv2.putText(frame, parts_text,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if status == "HIDDEN SURVIVOR?":
                cv2.putText(frame, ">>> CHECK THIS AREA <<<",
                            (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if keypoints is not None:
            for person_kp in keypoints.data:

                for a, b in SKELETON:
                    kp_a = person_kp[a]
                    kp_b = person_kp[b]
                    if kp_a[2] > CONF and kp_b[2] > CONF:
                        pt1 = (int(kp_a[0]), int(kp_a[1]))
                        pt2 = (int(kp_b[0]), int(kp_b[1]))
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

                for idx, kp in enumerate(person_kp):
                    x, y, conf_kp = float(kp[0]), float(kp[1]), float(kp[2])
                    if conf_kp < CONF:
                        continue

                    cx, cy = int(x), int(y)
                    dot_color = KEYPOINT_COLORS.get(idx, (255, 255, 255))
                    name = KEYPOINT_NAMES.get(idx, "")

                    cv2.circle(frame, (cx, cy), 6, dot_color, -1)
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 1)
                    cv2.putText(frame, name,
                                (cx + 10, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1)

    cv2.putText(frame, "GREEN  = Full person",     (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(frame, "ORANGE = Body fragment",   (10, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    cv2.putText(frame, "RED    = Hidden survivor", (10, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(frame, "CYAN   = Signal light",    (10, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, "YELLOW=Head  ORANGE=Shoulder  CYAN=Elbow", (10, 416), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "GREEN=Wrist  PURPLE=Hip  RED=Knee",         (10, 434), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "Q - quit",                                  (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("Fragment Rescue AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
