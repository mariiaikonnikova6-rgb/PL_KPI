# Mireon — Fragment Rescue AI

**Mireon** is a drone-based computer vision project created for the **INFOMATRIX 2026 Romania Hackathon** in the category of **Disaster Prep & Emergency Response**.

The main idea of the project is simple but important:

> In a disaster zone, a person may not be fully visible.  
> Mireon does not only search for a complete human body — it also searches for human fragments, body keypoints, suspicious motion, and light signals.

The system analyzes a live video stream from a drone and helps rescuers quickly notice areas that may contain a hidden or partially visible survivor.

---

## Project Idea

After an earthquake, explosion, fire, building collapse, or another emergency, victims can be hidden under debris, behind objects, under blankets, in dark corners, or only partially visible.

A normal object detector may fail if the full human body is not clearly visible.

**Mireon solves this problem by detecting indirect signs of a human presence:**

- head;
- shoulders;
- arms;
- wrists;
- legs;
- knees;
- ankles;
- torso fragments;
- body pose keypoints;
- suspicious light signals;
- low-light conditions;
- possible human fragments in the video stream.

The goal is not to replace rescuers, but to help them understand where they should look first.

---

## Main Features

### 1. Drone Video Stream

The current version uses a **DJI Tello drone**.

The drone connects to the computer through Wi-Fi, sends a live video stream, and the Python program processes each frame in real time.

```python
tello = Tello()
tello.connect()
tello.streamon()
cap = tello.get_frame_read()
- Edge deployment on Jetson / Raspberry Pi / drone onboard computer.
