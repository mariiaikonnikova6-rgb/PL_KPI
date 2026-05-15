"""OpenCV motion detector for rescue-scene MVP."""
from __future__ import annotations

from typing import Dict, Any, List

import cv2
import numpy as np


class MotionDetector:
    """
    Motion detection using a combination of background subtraction and frame differencing.
    This helps classify a static suspicious fragment vs. a possible living survivor.
    """

    def __init__(self, min_area: int = 750, history: int = 350, var_threshold: int = 32):
        self.min_area = int(min_area)
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=int(history),
            varThreshold=int(var_threshold),
            detectShadows=True,
        )
        self.previous_gray = None

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        result = {
            "motion_detected": False,
            "motion_boxes": [],
            "motion_area": 0,
            "motion_score": 0.0,
            "mask": None,
        }
        if frame is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        bg_mask = self.background.apply(frame)
        _, bg_mask = cv2.threshold(bg_mask, 200, 255, cv2.THRESH_BINARY)

        if self.previous_gray is None:
            self.previous_gray = gray
            result["mask"] = bg_mask
            return result

        frame_delta = cv2.absdiff(self.previous_gray, gray)
        _, diff_mask = cv2.threshold(frame_delta, 24, 255, cv2.THRESH_BINARY)
        self.previous_gray = gray

        mask = cv2.bitwise_or(bg_mask, diff_mask)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Dict[str, Any]] = []
        total_area = 0
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            total_area += area
            boxes.append({
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "area": area,
            })

        frame_area = frame.shape[0] * frame.shape[1]
        motion_score = min(1.0, (total_area / max(1, frame_area)) * 8.0)

        result["motion_detected"] = len(boxes) > 0
        result["motion_boxes"] = boxes
        result["motion_area"] = total_area
        result["motion_score"] = motion_score
        result["mask"] = mask
        return result
