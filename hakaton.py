from djitellopy import Tello
from ultralytics import YOLO
import cv2
import numpy as np
import time

FRAME_W = 320
FRAME_H = 240

MODEL_PATH = "yolov8n-pose.pt"

DETECTION_INTERVAL = 2
DETECT_CONF = 0.25
KP_CONF = 0.25

LIGHT_THRESHOLD = 185
LIGHT_AREA_MIN = 40

DISPLAY_W = 1280
DISPLAY_H = 720

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

cap = tello.get_frame_read()
model = YOLO(MODEL_PATH)

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
    "HEAD":      [0, 1, 2, 3, 4],
    "LEFT_ARM":  [5, 7, 9],
    "RIGHT_ARM": [6, 8, 10],
    "TORSO":     [5, 6, 11, 12],
    "LEFT_LEG":  [11, 13, 15],
    "RIGHT_LEG": [12, 14, 16],
}

KP_COLORS = {
    0:  (0, 0, 255),
    1:  (0, 0, 255),
    2:  (0, 0, 255),
    3:  (0, 0, 255),
    4:  (0, 0, 255),
    5:  (0, 165, 255),
    6:  (0, 165, 255),
    7:  (0, 255, 255),
    8:  (0, 255, 255),
    9:  (0, 255, 0),
    10: (0, 255, 0),
    11: (255, 0, 255),
    12: (255, 0, 255),
    13: (255, 100, 0),
    14: (255, 100, 0),
    15: (200, 200, 0),
    16: (200, 200, 0),
}

KP_NAMES = {
    0:  "Nose",
    1:  "L.Eye",
    2:  "R.Eye",
    3:  "L.Ear",
    4:  "R.Ear",
    5:  "L.Shoulder",
    6:  "R.Shoulder",
    7:  "L.Elbow",
    8:  "R.Elbow",
    9:  "L.Wrist",
    10: "R.Wrist",
    11: "L.Hip",
    12: "R.Hip",
    13: "L.Knee",
    14: "R.Knee",
    15: "L.Ankle",
    16: "R.Ankle",
}

cached_results = None
frame_count = 0
light_detection_on = True
light_boost_on = False

def limb_visible(kp, indices):
    return sum(1 for idx in indices if kp[idx][2] > KP_CONF) / len(indices)

def analyze_pose(person_kp):
    limb_scores = {name: limb_visible(person_kp, idx) for name, idx in LIMBS.items()}
    strong_limbs = sum(s >= 0.5 for s in limb_scores.values())
    avg_score = np.mean(list(limb_scores.values()))

    if avg_score > 0.55:
        return {"status": "PERSON",            "color": (0, 255, 0),   "scores": limb_scores}
    elif strong_limbs >= 2:
        return {"status": "BODY FRAGMENT",     "color": (0, 165, 255), "scores": limb_scores}
    elif strong_limbs >= 1:
        return {"status": "POSSIBLE SURVIVOR", "color": (0, 0, 255),   "scores": limb_scores}
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

def draw_top_bar(canvas, battery):
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, 50), (15, 15, 15), -1)
    cv2.line(canvas, (0, 50), (DISPLAY_W, 50), (0, 200, 255), 1)

    logo = cv2.imread(r"C:\Users\Khrystyna PC\Desktop\PythonProject\PythonProject\logo.png", cv2.IMREAD_UNCHANGED)
    if logo is not None:
        lh = 40
        lw = int(logo.shape[1] * lh / logo.shape[0])
        logo = cv2.resize(logo, (lw, lh))
        if logo.shape[2] == 4:
            alpha = logo[:, :, 3] / 255.0
            for c in range(3):
                canvas[5:5+lh, 10:10+lw, c] = (
                    alpha * logo[:, :, c] + (1 - alpha) * canvas[5:5+lh, 10:10+lw, c]
                )
        else:
            canvas[5:5+lh, 10:10+lw] = logo
    else:
        cv2.putText(canvas, "FRAGMENT RESCUE AI", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    bat_color = (0, 255, 0) if battery > 30 else (0, 0, 255)
    cv2.putText(canvas, f"BAT: {battery}%", (DISPLAY_W - 160, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bat_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, time.strftime("%H:%M:%S"), (DISPLAY_W - 340, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

def draw_bottom_bar(canvas):
    cv2.rectangle(canvas, (0, DISPLAY_H - 100), (DISPLAY_W, DISPLAY_H), (15, 15, 15), -1)
    cv2.line(canvas, (0, DISPLAY_H - 100), (DISPLAY_W, DISPLAY_H - 100), (0, 200, 255), 1)

    statuses = [
        ("● PERSON",            (0, 255, 0)),
        ("● BODY FRAGMENT",     (0, 165, 255)),
        ("● POSSIBLE SURVIVOR", (0, 0, 255)),
    ]
    x = 14
    for label, color in statuses:
        cv2.putText(canvas, label, (x, DISPLAY_H - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x += tw + 20

    parts = [
        ("HEAD",     (0, 0, 255)),
        ("SHOULDER", (0, 165, 255)),
        ("ELBOW",    (0, 255, 255)),
        ("WRIST",    (0, 255, 0)),
        ("HIP",      (255, 0, 255)),
        ("KNEE",     (255, 100, 0)),
        ("ANKLE",    (200, 200, 0)),
    ]
    x = 14
    for label, color in parts:
        cv2.circle(canvas, (x + 6, DISPLAY_H - 40), 6, color, -1)
        cv2.putText(canvas, label, (x + 16, DISPLAY_H - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x += tw + 28

    light_clr = (0, 255, 0) if light_detection_on else (100, 100, 100)
    boost_clr = (0, 255, 255) if light_boost_on else (100, 100, 100)
    cv2.putText(canvas, "[L] LIGHT DETECT", (DISPLAY_W - 340, DISPLAY_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_clr, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[B] NIGHT BOOST", (DISPLAY_W - 340, DISPLAY_H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, boost_clr, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[Q] QUIT", (DISPLAY_W - 130, DISPLAY_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

battery = tello.get_battery()

cv2.namedWindow("Rescue AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rescue AI", DISPLAY_W, DISPLAY_H)

print("Q = quit | L = toggle light | B = night boost")

fps_timer = time.time()
fps_counter = 0
fps = 0

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

    VIDEO_H = DISPLAY_H - 150
    VIDEO_W = int(VIDEO_H * 16 / 9)
    video_x = (DISPLAY_W - VIDEO_W) // 2

    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    video_frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas[50:50+VIDEO_H, video_x:video_x+VIDEO_W] = video_frame

    scale_x = VIDEO_W / FRAME_W
    scale_y = VIDEO_H / FRAME_H

    if light_detection_on:
        avg_light, lights = analyze_light(frame)
        if avg_light < 40:
            txt, clr = "LOW LIGHT", (0, 0, 255)
        elif avg_light > 200:
            txt, clr = "OVEREXPOSED", (0, 165, 255)
        else:
            txt, clr = f"LIGHT OK {int(avg_light)}", (0, 255, 0)

        cv2.putText(canvas, txt, (video_x + 10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2, cv2.LINE_AA)

        for (x, y, w, h) in lights:
            sx = int(x * scale_x) + video_x
            sy = int(y * scale_y) + 50
            sw = int(w * scale_x)
            sh = int(h * scale_y)
            cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 1)

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
                sx1 = int(x1 * scale_x) + video_x
                sy1 = int(y1 * scale_y) + 50
                sx2 = int(x2 * scale_x) + video_x
                sy2 = int(y2 * scale_y) + 50
                color = analysis["color"]

                # Кутова рамка
                cl = 20
                for p1, p2 in [
                    ((sx1,sy1),(sx1+cl,sy1)), ((sx1,sy1),(sx1,sy1+cl)),
                    ((sx2,sy1),(sx2-cl,sy1)), ((sx2,sy1),(sx2,sy1+cl)),
                    ((sx1,sy2),(sx1+cl,sy2)), ((sx1,sy2),(sx1,sy2-cl)),
                    ((sx2,sy2),(sx2-cl,sy2)), ((sx2,sy2),(sx2,sy2-cl)),
                ]:
                    cv2.line(canvas, p1, p2, color, 2)

                # Статус
                cv2.putText(canvas, analysis["status"],
                            (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                # Скелет
                for a, b in SKELETON:
                    kp_a = person_kp[a]
                    kp_b = person_kp[b]
                    if kp_a[2] > KP_CONF and kp_b[2] > KP_CONF:
                        pt1 = (int(kp_a[0]*scale_x)+video_x, int(kp_a[1]*scale_y)+50)
                        pt2 = (int(kp_b[0]*scale_x)+video_x, int(kp_b[1]*scale_y)+50)
                        cv2.line(canvas, pt1, pt2, (0, 0, 0), 3)
                        cv2.line(canvas, pt1, pt2, (220, 220, 220), 1)

                # Крапки з назвами
                for idx, kp in enumerate(person_kp):
                    x, y, c = kp
                    if c < KP_CONF:
                        continue
                    cx = int(x * scale_x) + video_x
                    cy = int(y * scale_y) + 50
                    dot_color = KP_COLORS.get(idx, (255, 255, 255))
                    name = KP_NAMES.get(idx, "")

                    # Крапка
                    cv2.circle(canvas, (cx, cy), 7, (0, 0, 0), -1)
                    cv2.circle(canvas, (cx, cy), 5, dot_color, -1)
                    cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1)

                    # Назва з темним фоном
                    (tw, th), _ = cv2.getTextSize(
                        name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(canvas,
                                  (cx + 9, cy - th - 3),
                                  (cx + 9 + tw + 4, cy + 3),
                                  (0, 0, 0), -1)
                    cv2.putText(canvas, name,
                                (cx + 11, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                dot_color, 1, cv2.LINE_AA)

    fps_counter += 1
    if time.time() - fps_timer >= 1:
        fps = fps_counter
        fps_counter = 0
        fps_timer = time.time()

    cv2.putText(canvas, f"FPS: {fps}",
                (video_x + VIDEO_W - 90, 50 + VIDEO_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if light_boost_on:
        cv2.putText(canvas, "NIGHT BOOST ON",
                    (video_x + 10, 50 + VIDEO_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    draw_top_bar(canvas, battery)
    draw_bottom_bar(canvas)

    cv2.imshow("Rescue AI", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("l"):
        light_detection_on = not light_detection_on
    elif key == ord("b"):
        light_boost_on = not light_boost_on

tello.streamoff()
cv2.destroyAllWindows()