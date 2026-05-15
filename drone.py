from djitellopy import Tello
from ultralytics import YOLO
import cv2

# CONNECT TO TELLO
tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

# START VIDEO STREAM
tello.streamon()

frame_reader = tello.get_frame_read()

# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

while True:

    # GET FRAME
    frame = frame_reader.frame

    # RUN DETECTION
    results = model(frame, verbose=False)

    # DRAW RESULTS
    annotated = results[0].plot()

    # SHOW VIDEO
    cv2.imshow("Tello YOLO Feed", annotated)

    key = cv2.waitKey(1)

    # PRESS Q TO EXIT
    if key == ord('q'):
        break

# CLEANUP
tello.streamoff()
cv2.destroyAllWindows()