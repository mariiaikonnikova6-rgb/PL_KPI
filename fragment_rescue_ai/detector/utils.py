"""Utility functions for drawing, reporting, and video sources."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import cv2
import numpy as np

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mss = None


BGR_COLORS = {
    "Green": (80, 220, 120),
    "Yellow": (0, 220, 255),
    "Orange": (0, 150, 255),
    "Red": (60, 60, 255),
    "Person": (90, 220, 255),
    "Fragment": (255, 190, 70),
    "Motion": (255, 90, 90),
    "Text": (240, 240, 240),
    "Black": (15, 15, 18),
}


def ensure_report_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "threat_level", "confidence", "status", "explanation", "recommendation"])


def append_detection_report(path: Path, event: Dict[str, Any]) -> None:
    ensure_report_file(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            event.get("timestamp", current_timestamp()),
            event.get("threat_level", ""),
            event.get("confidence", ""),
            event.get("status", ""),
            event.get("explanation", ""),
            event.get("recommendation", ""),
        ])


def current_timestamp() -> str:
    return time.strftime("%H:%M:%S")


def resize_frame(frame: np.ndarray, width: int = 960) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def point_inside_box(point: Tuple[float, float], box: Iterable[float], margin: int = 0) -> bool:
    x, y = point
    x1, y1, x2, y2 = map(float, box)
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def boxes_intersect(a: Iterable[float], b: Iterable[float], margin: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ax1 -= margin; ay1 -= margin; ax2 += margin; ay2 += margin
    bx1 -= margin; by1 -= margin; bx2 += margin; by2 += margin
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def motion_near_human_sign(
    keypoints: List[Dict[str, Any]],
    human_boxes: List[Dict[str, Any]],
    motion_boxes: List[Dict[str, Any]],
    margin: int = 55,
) -> bool:
    if not motion_boxes:
        return False

    for mb in motion_boxes:
        motion_box = mb["box"]
        for kp in keypoints:
            if point_inside_box((kp["x"], kp["y"]), motion_box, margin=margin):
                return True
        for hb in human_boxes:
            if boxes_intersect(hb["box"], motion_box, margin=margin):
                return True
    return False


def detect_silhouette_fallback(frame: np.ndarray, motion_boxes: List[Dict[str, Any]]) -> bool:
    """
    Very simple MVP fallback for silhouette-like evidence.
    It looks for tall moving contours. This is not a real body-part detector,
    but it gives the demo a backup signal when YOLO is unavailable.
    """
    if not motion_boxes:
        return False
    for item in motion_boxes:
        x1, y1, x2, y2 = item["box"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        ratio = h / float(w)
        if area > 1600 and 1.2 <= ratio <= 5.5:
            return True
    return False


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 8, y)
    cv2.rectangle(frame, (x, y - th - baseline - 7), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, (15, 15, 18), thickness, cv2.LINE_AA)


def draw_person_boxes(frame: np.ndarray, boxes: List[Dict[str, Any]], partial: bool = False) -> None:
    color = BGR_COLORS["Fragment"] if partial else BGR_COLORS["Person"]
    for item in boxes:
        x1, y1, x2, y2 = map(int, item["box"])
        conf = float(item.get("confidence", 0.0))
        label = f"Weak person sign {conf:.2f}" if partial else f"Person detected {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color)


def draw_keypoints(frame: np.ndarray, keypoints: List[Dict[str, Any]]) -> None:
    for kp in keypoints:
        x, y = int(kp["x"]), int(kp["y"])
        name = kp.get("name", "kp")
        conf = kp.get("confidence", 0.0)
        cv2.circle(frame, (x, y), 4, BGR_COLORS["Fragment"], -1)
        cv2.circle(frame, (x, y), 8, BGR_COLORS["Fragment"], 1)
        cv2.putText(frame, f"{name} {conf:.2f}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_COLORS["Text"], 1, cv2.LINE_AA)


def draw_motion_boxes(frame: np.ndarray, boxes: List[Dict[str, Any]]) -> None:
    for item in boxes:
        x1, y1, x2, y2 = map(int, item["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BGR_COLORS["Motion"], 2)
        draw_label(frame, "Motion", x1, y1, BGR_COLORS["Motion"], scale=0.48)


def draw_status_overlay(frame: np.ndarray, result: Dict[str, Any]) -> None:
    level = result.get("threat_level", "Green")
    color = BGR_COLORS.get(level, BGR_COLORS["Green"])
    status = result.get("status", "CLEAR AREA")
    probability = result.get("human_probability", 0)
    recommendation = result.get("recommendation", "Monitor this area")

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (15, 15, 18), -1)
    cv2.rectangle(overlay, (0, 0), (w, 7), color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, f"CURRENT STATUS: {level.upper()} | {status}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Human probability: {probability}%", (18, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.72, BGR_COLORS["Text"], 2, cv2.LINE_AA)
    cv2.putText(frame, recommendation, (max(18, w - 410), 67), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)


def create_event_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    explanation = " | ".join(result.get("explanation", []))
    return {
        "timestamp": current_timestamp(),
        "threat_level": result.get("threat_level", "Green"),
        "confidence": result.get("human_probability", 0),
        "status": result.get("status", ""),
        "explanation": explanation,
        "recommendation": result.get("recommendation", ""),
    }


class MSSScreenCapture:
    """OpenCV-like screen capture source for scrcpy or browser/drone preview windows."""

    def __init__(self, region: Dict[str, int]):
        if mss is None:
            raise RuntimeError("mss is not installed. Install it with: pip install mss")
        self.region = {
            "left": int(region.get("left", 0)),
            "top": int(region.get("top", 0)),
            "width": int(region.get("width", 1280)),
            "height": int(region.get("height", 720)),
        }
        self.sct = mss.mss()

    def read(self) -> Tuple[bool, np.ndarray | None]:
        img = np.array(self.sct.grab(self.region))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        try:
            self.sct.close()
        except Exception:
            pass



import cv2
import csv
import os


def draw_status_panel(frame, threat_level, human_probability, status, recommendation):
    """
    Draw a simple status overlay on the video frame.
    """

    colors = {
        "Green": (0, 180, 0),
        "Yellow": (0, 255, 255),
        "Orange": (0, 165, 255),
        "Red": (0, 0, 255),
    }

    color = colors.get(threat_level, (255, 255, 255))

    # Background panel
    cv2.rectangle(frame, (10, 10), (620, 150), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (620, 150), color, 2)

    cv2.putText(
        frame,
        f"CURRENT STATUS: {threat_level}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Human probability: {human_probability:.1f}%",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"{status}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2
    )

    cv2.putText(
        frame,
        f"Recommendation: {recommendation}",
        (25, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2
    )

    return frame


def save_detection_to_csv(timestamp, threat_level, confidence, status, explanation, recommendation):
    """
    Save one detection event to reports/detections.csv.
    """

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(reports_dir, "detections.csv")

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "threat_level",
                "confidence",
                "status",
                "explanation",
                "recommendation"
            ])

        writer.writerow([
            timestamp,
            threat_level,
            round(confidence, 2),
            status,
            explanation,
            recommendation
        ])