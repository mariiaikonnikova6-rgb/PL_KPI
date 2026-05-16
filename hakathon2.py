from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time
import numpy as np

# ——— Підключення дрона ———
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

# ——— Модель ———
model = YOLO("yolov8n-pose.pt")
cap = tello.get_frame_read()

KEYPOINT_NAMES = {
    0: "Nose", 1: "L.Eye", 2: "R.Eye",
    3: "L.Ear", 4: "R.Ear",
    5: "L.Shoulder", 6: "R.Shoulder",
    7: "L.Elbow", 8: "R.Elbow",
    9: "L.Wrist", 10: "R.Wrist",
    11: "L.Hip", 12: "R.Hip",
    13: "L.Knee", 14: "R.Knee",
    15: "L.Ankle", 16: "R.Ankle",
}

KEYPOINT_COLORS = {
    0: (255, 255, 0), 1: (255, 255, 0), 2: (255, 255, 0),
    3: (255, 255, 0), 4: (255, 255, 0),
    5: (0, 165, 255), 6: (0, 165, 255),
    7: (0, 255, 255), 8: (0, 255, 255),
    9: (0, 255, 0),  10: (0, 255, 0),
    11: (255, 0, 255), 12: (255, 0, 255),
    13: (0, 80, 255), 14: (0, 80, 255),
    15: (255, 130, 0), 16: (255, 130, 0),
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CONF = 0.4

def draw_label(frame, text, x, y, color, bg=True, font_scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if bg:
        cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + 4),
                      (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=30):
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

def draw_top_bar(frame, battery):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # ——— Логотип ———
    logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
    if logo is not None:
        logo_h = 60
        ratio = logo_h / logo.shape[0]
        logo_w = int(logo.shape[1] * ratio)
        logo = cv2.resize(logo, (logo_w, logo_h))

        if logo.shape[2] == 4:
            alpha = logo[:, :, 3] / 255.0
            for c in range(3):
                frame[5:5+logo_h, 10:10+logo_w, c] = (
                    alpha * logo[:, :, c] +
                    (1 - alpha) * frame[5:5+logo_h, 10:10+logo_w, c]
                )
        else:
            frame[5:5+logo_h, 10:10+logo_w] = logo

    bat_color = (0, 255, 0) if battery > 30 else (0, 0, 255)
    cv2.putText(frame, f"BAT: {battery}%", (w - 180, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bat_color, 2, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 380, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 380, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1, cv2.LINE_AA)

def draw_legend(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 110), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    items = [
        ("● PERSON",          (0, 255, 0)),
        ("● FRAGMENT",        (0, 165, 255)),
        ("● HIDDEN SURVIVOR", (0, 0, 255)),
    ]
    x = 16
    for label, color in items:
        cv2.putText(frame, label, (x, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        x += tw + 30

    dots = [
        ("HEAD",     (255, 255, 0)),
        ("SHOULDER", (0, 165, 255)),
        ("ELBOW",    (0, 255, 255)),
        ("WRIST",    (0, 255, 0)),
        ("HIP",      (255, 0, 255)),
        ("KNEE",     (0, 80, 255)),
        ("ANKLE",    (255, 130, 0)),
    ]
    x = 16
    for label, color in dots:
        cv2.circle(frame, (x + 7, h - 42), 7, color, -1)
        cv2.putText(frame, label, (x + 20, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        x += tw + 36

    cv2.putText(frame, "Press Q to quit", (w - 200, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

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
        return "PERSON", visible, (0, 255, 0), int(full_ratio * 100)
    elif len(visible) >= 2:
        return "FRAGMENT", visible, (0, 165, 255), int(full_ratio * 100)
    elif len(visible) == 1:
        return "HIDDEN SURVIVOR?", visible, (0, 0, 255), int(full_ratio * 100)
    return None, [], (255, 255, 255), 0

battery = tello.get_battery()

# ——— Повноекранне вікно 16:9 ———
cv2.namedWindow("Fragment Rescue AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Fragment Rescue AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Stream starting... press Q to quit")

while True:
    frame = cap.frame
    if frame is None:
        continue

    # 1920x1080 — Full HD 16:9
    frame = cv2.resize(frame, (1440, 1080))

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

            draw_corner_box(frame, x1, y1, x2, y2, color, thickness=3)

            label = f"{status}  {conf:.0%}"
            draw_label(frame, label, x1, y1 - 14, color, bg=True, font_scale=0.85, thickness=2)

            if visible_parts:
                parts_text = "  |  ".join(
                    f"{name} {int(r * 100)}%" for name, r in visible_parts
                )
                draw_label(frame, parts_text, x1, y2 + 30, color, bg=True, font_scale=0.6)

            if status == "HIDDEN SURVIVOR?":
                draw_label(frame, "!!! CHECK THIS AREA !!!",
                           x1, y1 - 50, (0, 0, 255), bg=True, font_scale=0.85, thickness=2)

        if keypoints is not None:
            for person_kp in keypoints.data:

                for a, b in SKELETON:
                    kp_a = person_kp[a]
                    kp_b = person_kp[b]
                    if kp_a[2] > CONF and kp_b[2] > CONF:
                        pt1 = (int(kp_a[0]), int(kp_a[1]))
                        pt2 = (int(kp_b[0]), int(kp_b[1]))
                        cv2.line(frame, pt1, pt2, (0, 0, 0), 4)
                        cv2.line(frame, pt1, pt2, (220, 220, 220), 2)

                for idx, kp in enumerate(person_kp):
                    x, y, conf_kp = float(kp[0]), float(kp[1]), float(kp[2])
                    if conf_kp < CONF:
                        continue

                    cx, cy = int(x), int(y)
                    dot_color = KEYPOINT_COLORS.get(idx, (255, 255, 255))
                    name = KEYPOINT_NAMES.get(idx, "")

                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
                    cv2.circle(frame, (cx, cy), 8, dot_color, -1)
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 1)

                    draw_label(frame, name, cx + 14, cy + 7,
                               dot_color, bg=True, font_scale=0.5)

    draw_top_bar(frame, battery)
    draw_legend(frame)

    cv2.imshow("Fragment Rescue AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()