import cv2
import mediapipe as mp
import numpy as np
import time
from djitellopy import Tello

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
FRAME_W = 960
FRAME_H = 720

# ─────────────────────────────
# MODELS
# ─────────────────────────────
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def is_full_body(pose_res):
    if not pose_res.pose_landmarks:
        return False

    visible = sum(
        1 for lm in pose_res.pose_landmarks.landmark
        if lm.visibility > 0.6
    )
    return visible > 20


def draw_pose(frame, pose_res):
    if not pose_res.pose_landmarks:
        return

    h, w = frame.shape[:2]
    for l in pose_res.pose_landmarks.landmark:
        x, y = int(l.x * w), int(l.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def draw_hands(frame, hand_res):
    if not hand_res.multi_hand_landmarks:
        return

    h, w = frame.shape[:2]
    for hand in hand_res.multi_hand_landmarks:
        for lm in hand.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)


def detect_hand_motion(hand_res):
    if not hand_res.multi_hand_landmarks:
        return []

    actions = []

    for hand in hand_res.multi_hand_landmarks:
        wrist = hand.landmark[0]
        index = hand.landmark[8]

        dx = abs(index.x - wrist.x)
        dy = abs(index.y - wrist.y)

        if dx < 0.03 and dy < 0.03:
            actions.append("Pinch-like gesture")

        if dy < 0.05 and index.y < wrist.y:
            actions.append("Hand raised")

    return actions


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():

    tello = Tello()
    tello.connect()

    print(f"[DRONE] Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_reader = tello.get_frame_read()

    fps_t = time.time()
    fps = 0

    print("[READY] Drone vision pipeline running")

    while True:

        frame = frame_reader.frame
        if frame is None:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        mode = "UNKNOWN"

        if is_full_body(pose_res):
            mode = "FULL_BODY"
        elif hand_res.multi_hand_landmarks and not pose_res.pose_landmarks:
            mode = "HAND_ONLY"

        # ─────────────────────────────
        # DRAWING
        # ─────────────────────────────
        if mode == "FULL_BODY":
            draw_pose(frame, pose_res)

        draw_hands(frame, hand_res)

        # ─────────────────────────────
        # HAND LOGIC
        # ─────────────────────────────
        hand_actions = detect_hand_motion(hand_res)

        y = 30
        cv2.putText(frame, f"Mode: {mode}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for i, act in enumerate(hand_actions):
            cv2.putText(frame, act, (10, y + 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ─────────────────────────────
        # FPS
        # ─────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
        fps_t = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Drone Body Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
