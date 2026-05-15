from ultralytics import YOLO
from djitellopy import Tello
import cv2


# ---------------- IoU ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea != 0 else 0


# ---------------- Pose → Box ----------------
def pose_to_box(kp):
    x_coords = kp[:, 0]
    y_coords = kp[:, 1]

    return (
        int(min(x_coords)),
        int(min(y_coords)),
        int(max(x_coords)),
        int(max(y_coords))
    )


# ---------------- MODELS ----------------
detector = YOLO("yolov8n.pt")
pose = YOLO("yolov8n-pose.pt")


# ---------------- DRONE ----------------
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()


# ---------------- MAIN LOOP ----------------
while True:

    frame = frame_read.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    # -------- DETECTION --------
    det_results = detector(frame, classes=[0], verbose=False)

    detection_boxes = []
    for r in det_results:
        if r.boxes is None:
            continue
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detection_boxes.append((x1, y1, x2, y2))

    # -------- POSE --------
    pose_results = pose(frame, verbose=False)

    pose_boxes = []
    keypoints_list = []

    for r in pose_results:
        if r.keypoints is None:
            continue

        kps = r.keypoints.xy.cpu().numpy()

        for person_kp in kps:
            pose_boxes.append(pose_to_box(person_kp))
            keypoints_list.append(person_kp)

    # -------- IoU MATCHING --------
    matches = []

    for det_box in detection_boxes:

        best_iou = 0
        best_idx = -1

        for i, pose_box in enumerate(pose_boxes):

            score = iou(det_box, pose_box)

            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou > 0.3 and best_idx != -1:
            matches.append((det_box, keypoints_list[best_idx]))

    # -------- DRAW RESULTS --------
    annotated = frame.copy()

    # draw detections
    for x1, y1, x2, y2 in detection_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # draw pose
    for person in keypoints_list:
        for x, y in person:
            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)

    # highlight matched pairs
    for det_box, kp in matches:
        x1, y1, x2, y2 = det_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("Dual System (IoU Merged)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------- CLEANUP ----------------
tello.streamoff()
cv2.destroyAllWindows()
