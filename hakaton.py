from djitellopy import Tello
from ultralytics import YOLO
import cv2
import numpy as np
import time

FRAME_W = 320
FRAME_H = 240

MODEL_PATH = "yolov8n-pose.pt"

DETECT_CONF = 0.25
KP_CONF = 0.25

LIGHT_THRESHOLD = 185
LIGHT_AREA_MIN = 40

DISPLAY_W = 1280
DISPLAY_H = 720
PANEL_W = 260

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

cap = tello.get_frame_read()
model = YOLO(MODEL_PATH)

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LIMBS = {
    "HEAD":      [0, 1, 2, 3, 4],
    "LEFT_ARM":  [5, 7, 9],
    "RIGHT_ARM": [6, 8, 10],
    "TORSO":     [5, 6, 11, 12],
    "LEFT_LEG":  [11, 13, 15],
    "RIGHT_LEG": [12, 14, 16],
}

KP_COLORS = {
    0:  (0, 0, 200),   1:  (0, 0, 200),
    2:  (0, 0, 200),   3:  (0, 0, 200),
    4:  (0, 0, 200),   5:  (0, 130, 200),
    6:  (0, 130, 200), 7:  (0, 200, 200),
    8:  (0, 200, 200), 9:  (0, 180, 0),
    10: (0, 180, 0),   11: (180, 0, 180),
    12: (180, 0, 180), 13: (200, 80, 0),
    14: (200, 80, 0),  15: (150, 150, 0),
    16: (150, 150, 0),
}

KP_NAMES = {
    0: "Nose",       1: "L.Eye",      2: "R.Eye",
    3: "L.Ear",      4: "R.Ear",      5: "L.Shoulder",
    6: "R.Shoulder", 7: "L.Elbow",    8: "R.Elbow",
    9: "L.Wrist",   10: "R.Wrist",   11: "L.Hip",
    12: "R.Hip",    13: "L.Knee",    14: "R.Knee",
    15: "L.Ankle",  16: "R.Ankle",
}

cached_results = None
frame_count = 0
light_detection_on = True
light_boost_on = False
skeleton_on = True
labels_on = True


def limb_visible(kp, indices):
    return sum(1 for idx in indices if kp[idx][2] > KP_CONF) / len(indices)


def analyze_pose(person_kp):
    limb_scores = {name: limb_visible(person_kp, idx) for name, idx in LIMBS.items()}
    strong_limbs = sum(s >= 0.5 for s in limb_scores.values())
    avg_score = np.mean(list(limb_scores.values()))
    if avg_score > 0.55:
        return {"status": "PERSON",            "color": (0, 200, 0),   "scores": limb_scores}
    elif strong_limbs >= 2:
        return {"status": "BODY FRAGMENT",     "color": (0, 130, 255), "scores": limb_scores}
    elif strong_limbs >= 1:
        return {"status": "POSSIBLE SURVIVOR", "color": (0, 0, 220),   "scores": limb_scores}
    return None


def analyze_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    _, thresh = cv2.threshold(gray, LIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= LIGHT_AREA_MIN:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))
    return avg, detections


def draw_panel(canvas, battery, fps, avg_light=None):
    px = DISPLAY_W - PANEL_W

    # Напівпрозорий фон панелі
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, 0), (DISPLAY_W, DISPLAY_H), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

    cv2.line(canvas, (px, 0), (px, DISPLAY_H), (0, 200, 255), 2)

    y = 20


    logo = cv2.imread(
        r"C:\Users\Khrystyna PC\Desktop\PythonProject\PythonProject\logo.png",
        cv2.IMREAD_UNCHANGED
    )
    if logo is not None:
        lw = PANEL_W - 20
        lh = int(logo.shape[0] * lw / logo.shape[1])
        logo = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_LANCZOS4)
        if logo.shape[2] == 4:
            # PNG з прозорістю — накладаємо по альфа-каналу
            alpha = logo[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+lh, px+10:px+10+lw, c] = (
                    alpha * logo[:, :, c] +
                    (1 - alpha) * canvas[y:y+lh, px+10:px+10+lw, c]
                ).astype(np.uint8)
        else:
            canvas[y:y+lh, px+10:px+10+lw] = logo
        y += lh + 12
    else:
        cv2.putText(canvas, "Mireon", (px+10, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2, cv2.LINE_AA)
        y += 40

    cv2.line(canvas, (px+10, y), (DISPLAY_W-10, y), (180, 180, 180), 1)
    y += 14

  
    def section_title(text, yy):
        cv2.putText(canvas, text, (px+10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return yy + 18

    def info_row(label, value, color, yy):
        cv2.putText(canvas, label, (px+10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (px+120, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return yy + 16

 
    y = section_title("SYSTEM", y)
    bat_color = (0, 150, 0) if battery > 30 else (0, 0, 200)
    y = info_row("Battery:", f"{battery}%", bat_color, y)
    y = info_row("FPS:", str(fps), (0, 0, 0), y)
    y = info_row("Time:", time.strftime("%H:%M:%S"), (0, 0, 0), y)

    if avg_light is not None:
        if avg_light < 40:
            lv, lc = "LOW", (0, 0, 200)
        elif avg_light > 200:
            lv, lc = "OVER", (0, 130, 200)
        else:
            lv, lc = "OK", (0, 150, 0)
        y = info_row("Light:", f"{lv} {int(avg_light)}", lc, y)

    y += 6
    cv2.line(canvas, (px+10, y), (DISPLAY_W-10, y), (180, 180, 180), 1)
    y += 14

    y = section_title("DETECTION STATUS", y)
    statuses = [
        ("PERSON",            (0, 180, 0)),
        ("BODY FRAGMENT",     (0, 100, 200)),
        ("POSSIBLE SURVIVOR", (0, 0, 200)),
    ]
    for label, color in statuses:
        cv2.circle(canvas, (px+16, y-4), 5, color, -1)
        cv2.putText(canvas, label, (px+26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y += 16

    y += 6
    cv2.line(canvas, (px+10, y), (DISPLAY_W-10, y), (180, 180, 180), 1)
    y += 14

  
    y = section_title("BODY PARTS", y)
    parts = [
        ("Head",     (0, 0, 200)),
        ("Shoulder", (0, 130, 200)),
        ("Elbow",    (0, 200, 200)),
        ("Wrist",    (0, 180, 0)),
        ("Hip",      (180, 0, 180)),
        ("Knee",     (200, 80, 0)),
        ("Ankle",    (150, 150, 0)),
    ]
    for label, color in parts:
        cv2.circle(canvas, (px+16, y-4), 5, color, -1)
        cv2.putText(canvas, label, (px+26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y += 15

    y += 6
    cv2.line(canvas, (px+10, y), (DISPLAY_W-10, y), (180, 180, 180), 1)
    y += 14

    
    y = section_title("FUNCTIONS", y)

    def toggle_row(key, label, state, yy):
        on_off = "ON" if state else "OFF"
        clr_box = (0, 180, 0) if state else (180, 0, 0)
        cv2.rectangle(canvas, (px+10, yy-12), (px+42, yy+4), clr_box, -1)
        cv2.putText(canvas, on_off, (px+12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[{key}] {label}", (px+48, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        return yy + 20

    y = toggle_row("L", "Light detect", light_detection_on, y)
    y = toggle_row("B", "Night boost",  light_boost_on,     y)
    y = toggle_row("S", "Skeleton",     skeleton_on,        y)
    y = toggle_row("N", "Labels",       labels_on,          y)

    y += 6
    cv2.line(canvas, (px+10, y), (DISPLAY_W-10, y), (180, 180, 180), 1)
    y += 14

    # ——— QUIT ———
    cv2.rectangle(canvas, (px+10, y), (DISPLAY_W-10, y+22), (80, 80, 80), -1)
    cv2.putText(canvas, "[Q]  QUIT", (px+60, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)



battery = tello.get_battery()
VIDEO_W = DISPLAY_W - PANEL_W
VIDEO_H = DISPLAY_H

cv2.namedWindow("Mireon", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mireon", DISPLAY_W, DISPLAY_H)

print("Q=quit | L=light | B=boost | S=skeleton | N=labels")

fps_timer = time.time()
fps_counter = 0
fps = 0
avg_light_val = None

while True:
    frame = cap.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    if light_boost_on:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=40)

    if frame_count % 5 == 0:
        cached_results = model(frame, classes=[0], conf=DETECT_CONF,
                               verbose=False, imgsz=320)
    frame_count += 1

    # Canvas
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), 30, dtype=np.uint8)

    # Відео на ліву частину
    video_frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas[0:VIDEO_H, 0:VIDEO_W] = video_frame

    scale_x = VIDEO_W / FRAME_W
    scale_y = VIDEO_H / FRAME_H

    # Light detection
    if light_detection_on:
        avg_light_val, lights = analyze_light(frame)
        if avg_light_val < 40:
            txt, clr = "LOW LIGHT", (0, 0, 255)
        elif avg_light_val > 200:
            txt, clr = "OVEREXPOSED", (0, 165, 255)
        else:
            txt, clr = f"LIGHT OK {int(avg_light_val)}", (0, 220, 0)
        cv2.putText(canvas, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2, cv2.LINE_AA)

    # Детекції
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

                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                sx1 = int(x1 * scale_x)
                sy1 = int(y1 * scale_y)
                sx2 = int(x2 * scale_x)
                sy2 = int(y2 * scale_y)
                color = analysis["color"]

                # Кутова рамка
                cl = 20
                for p1, p2 in [
                    ((sx1, sy1), (sx1+cl, sy1)), ((sx1, sy1), (sx1, sy1+cl)),
                    ((sx2, sy1), (sx2-cl, sy1)), ((sx2, sy1), (sx2, sy1+cl)),
                    ((sx1, sy2), (sx1+cl, sy2)), ((sx1, sy2), (sx1, sy2-cl)),
                    ((sx2, sy2), (sx2-cl, sy2)), ((sx2, sy2), (sx2, sy2-cl)),
                ]:
                    cv2.line(canvas, p1, p2, color, 2)

                # Статус
                (tw, th), _ = cv2.getTextSize(
                    analysis["status"], cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(canvas,
                              (sx1-2, sy1-th-14), (sx1+tw+4, sy1-2),
                              (0, 0, 0), -1)
                cv2.putText(canvas, analysis["status"], (sx1, sy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                # Скелет
                if skeleton_on:
                    for a, b in SKELETON:
                        kp_a = person_kp[a]
                        kp_b = person_kp[b]
                        if kp_a[2] > KP_CONF and kp_b[2] > KP_CONF:
                            pt1 = (int(kp_a[0]*scale_x), int(kp_a[1]*scale_y))
                            pt2 = (int(kp_b[0]*scale_x), int(kp_b[1]*scale_y))
                            cv2.line(canvas, pt1, pt2, (0, 0, 0), 3)
                            cv2.line(canvas, pt1, pt2, (220, 220, 220), 1)

                # Крапки з назвами
                for idx, kp in enumerate(person_kp):
                    x, y, c = kp
                    if c < KP_CONF:
                        continue
                    cx = int(x * scale_x)
                    cy = int(y * scale_y)
                    dot_color = KP_COLORS.get(idx, (255, 255, 255))
                    name = KP_NAMES.get(idx, "")

                    cv2.circle(canvas, (cx, cy), 7, (0, 0, 0), -1)
                    cv2.circle(canvas, (cx, cy), 5, dot_color, -1)
                    cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1)

                    if labels_on:
                        (tw, th), _ = cv2.getTextSize(
                            name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                        cv2.rectangle(canvas,
                                      (cx+9, cy-th-3), (cx+9+tw+4, cy+3),
                                      (0, 0, 0), -1)
                        cv2.putText(canvas, name, (cx+11, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    dot_color, 1, cv2.LINE_AA)

    # FPS
    fps_counter += 1
    if time.time() - fps_timer >= 1:
        fps = fps_counter
        fps_counter = 0
        fps_timer = time.time()

    if light_boost_on:
        cv2.putText(canvas, "NIGHT BOOST ON", (10, VIDEO_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    # Права панель
    draw_panel(canvas, battery, fps, avg_light_val)

    cv2.imshow("Mireon", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("l"):
        light_detection_on = not light_detection_on
    elif key == ord("b"):
        light_boost_on = not light_boost_on
    elif key == ord("s"):
        skeleton_on = not skeleton_on
    elif key == ord("n"):
        labels_on = not labels_on

tello.streamoff()
cv2.destroyAllWindows()
