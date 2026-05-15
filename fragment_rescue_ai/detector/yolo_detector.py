"""YOLO person detector wrapper."""
from __future__ import annotations

from typing import Dict, List, Any

import cv2
import numpy as np


class YOLOPersonDetector:
    """
    Wrapper around Ultralytics YOLO for class "person".
    If the model cannot be loaded, the detector becomes unavailable instead of crashing the app.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.45, low_conf_threshold: float = 0.20):
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.low_conf_threshold = float(low_conf_threshold)
        self.model = None
        self.available = False
        self.error_message = ""

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_path)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local environment
            self.error_message = f"Could not load YOLO model '{model_path}': {exc}"
            self.available = False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        result = {
            "available": self.available,
            "error": self.error_message,
            "person_detected": False,
            "best_person_confidence": 0.0,
            "person_boxes": [],
            "weak_person_boxes": [],
        }

        if not self.available or self.model is None or frame is None:
            return result

        try:
            predictions = self.model.predict(frame, conf=self.low_conf_threshold, verbose=False, imgsz=640)
        except Exception as exc:  # pragma: no cover
            result["available"] = False
            result["error"] = f"YOLO inference error: {exc}"
            return result

        person_boxes: List[Dict[str, Any]] = []
        weak_boxes: List[Dict[str, Any]] = []
        best_conf = 0.0

        for pred in predictions:
            names = getattr(pred, "names", {}) or {}
            boxes = getattr(pred, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    cls_id = int(box.cls[0].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                except Exception:
                    continue

                if cls_name != "person" and cls_id != 0:
                    continue

                best_conf = max(best_conf, conf)
                item = {
                    "box": xyxy,
                    "confidence": conf,
                    "class_name": "person",
                }
                if conf >= self.conf_threshold:
                    person_boxes.append(item)
                elif conf >= self.low_conf_threshold:
                    weak_boxes.append(item)

        result["person_detected"] = len(person_boxes) > 0
        result["best_person_confidence"] = best_conf
        result["person_boxes"] = person_boxes
        result["weak_person_boxes"] = weak_boxes
        return result


class HOGFallbackPersonDetector:
    """
    Optional classical OpenCV fallback.
    It is much weaker than YOLO but useful for demo mode if ultralytics is unavailable.
    """

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        result = {
            "available": True,
            "error": "",
            "person_detected": False,
            "best_person_confidence": 0.0,
            "person_boxes": [],
            "weak_person_boxes": [],
        }
        if frame is None:
            return result

        resized = frame
        scale = 1.0
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / float(w)
            resized = cv2.resize(frame, (640, int(h * scale)))

        rects, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        for (x, y, bw, bh), weight in zip(rects, weights):
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)
            conf = float(max(0.0, min(1.0, weight / 2.5)))
            item = {"box": [x1, y1, x2, y2], "confidence": conf, "class_name": "person"}
            if conf >= 0.45:
                result["person_boxes"].append(item)
            else:
                result["weak_person_boxes"].append(item)
            result["best_person_confidence"] = max(result["best_person_confidence"], conf)

        result["person_detected"] = len(result["person_boxes"]) > 0
        return result
