"""YOLO pose detector wrapper for body keypoints."""
from __future__ import annotations

from typing import Dict, List, Any

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

IMPORTANT_FRAGMENT_POINTS = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


class YOLOPoseDetector:
    """
    Wrapper around Ultralytics YOLO pose model.
    It returns visible keypoints, not just full person boxes, which is the core idea of the MVP.
    """

    def __init__(self, model_path: str, keypoint_conf_threshold: float = 0.25):
        self.model_path = model_path
        self.keypoint_conf_threshold = float(keypoint_conf_threshold)
        self.model = None
        self.available = False
        self.error_message = ""

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_path)
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error_message = f"Could not load YOLO pose model '{model_path}': {exc}"
            self.available = False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        result = {
            "available": self.available,
            "error": self.error_message,
            "keypoints": [],
            "keypoint_count": 0,
            "keypoint_names": [],
            "pose_boxes": [],
            "partial_pose_detected": False,
        }

        if not self.available or self.model is None or frame is None:
            return result

        try:
            predictions = self.model.predict(frame, conf=0.15, verbose=False, imgsz=640)
        except Exception as exc:  # pragma: no cover
            result["available"] = False
            result["error"] = f"YOLO pose inference error: {exc}"
            return result

        all_keypoints: List[Dict[str, Any]] = []
        pose_boxes: List[Dict[str, Any]] = []

        for pred in predictions:
            # Pose model can also return boxes around detected people.
            boxes = getattr(pred, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    try:
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                        pose_boxes.append({"box": xyxy, "confidence": conf, "class_name": "pose_person"})
                    except Exception:
                        continue

            keypoints_obj = getattr(pred, "keypoints", None)
            if keypoints_obj is None:
                continue

            data = getattr(keypoints_obj, "data", None)
            if data is None:
                continue

            try:
                kp_array = data.detach().cpu().numpy()
            except Exception:
                continue

            # Expected shape: [people, 17, 3] where last values are x, y, confidence.
            if kp_array.ndim != 3:
                continue

            for person_kps in kp_array:
                for idx, point in enumerate(person_kps):
                    if len(point) < 2:
                        continue
                    x = float(point[0])
                    y = float(point[1])
                    conf = float(point[2]) if len(point) >= 3 else 1.0
                    if x <= 0 or y <= 0:
                        continue
                    if conf < self.keypoint_conf_threshold:
                        continue
                    name = COCO_KEYPOINT_NAMES[idx] if idx < len(COCO_KEYPOINT_NAMES) else f"kp_{idx}"
                    if name not in IMPORTANT_FRAGMENT_POINTS:
                        continue
                    all_keypoints.append({
                        "name": name,
                        "x": x,
                        "y": y,
                        "confidence": conf,
                    })

        keypoint_names = [kp["name"] for kp in all_keypoints]
        result["keypoints"] = all_keypoints
        result["keypoint_count"] = len(all_keypoints)
        result["keypoint_names"] = keypoint_names
        result["pose_boxes"] = pose_boxes
        result["partial_pose_detected"] = len(all_keypoints) >= 2
        return result
