"""
Threat-level reasoning for Fragment Rescue AI.
This module intentionally uses readable heuristic logic because the project is an MVP.
The goal is not medical/operational certainty, but a clear demo decision pipeline:
YOLO person evidence + pose keypoints + motion + occlusion -> rescue priority.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Any


THREAT_COLORS = {
    "Green": "#2ecc71",
    "Yellow": "#f1c40f",
    "Orange": "#e67e22",
    "Red": "#e74c3c",
}


@dataclass
class ThreatResult:
    threat_level: str
    status: str
    human_probability: int
    explanation: List[str]
    recommendation: str
    rescue_priority: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_keypoint_names(names: Iterable[str] | None) -> List[str]:
    if not names:
        return []
    seen = set()
    clean = []
    for name in names:
        if name and name not in seen:
            clean.append(name)
            seen.add(name)
    return clean


def calculate_threat_level(
    person_detected: bool,
    person_confidence: float,
    keypoints_found: int,
    motion_detected: bool,
    occlusion_detected: bool,
    weak_person_candidate: bool = False,
    motion_near_human_sign: bool = False,
    keypoint_names: Iterable[str] | None = None,
    silhouette_detected: bool = False,
) -> ThreatResult:
    """
    Calculate threat level and human probability.

    Parameters
    ----------
    person_detected:
        True when YOLO sees a confident full/mostly visible person.
    person_confidence:
        Best YOLO confidence for class "person" in range [0, 1].
    keypoints_found:
        Number of visible human pose keypoints.
    motion_detected:
        True when OpenCV detects motion in the frame.
    occlusion_detected:
        True when the system sees body evidence but not a full person box.
    weak_person_candidate:
        True when YOLO sees a low-confidence person-like region.
    motion_near_human_sign:
        True when motion overlaps/appears near a person box or pose keypoints.
    keypoint_names:
        Names of visible keypoints for explanation.
    silhouette_detected:
        Optional fallback flag for contour/silhouette-like evidence.

    Returns
    -------
    ThreatResult
        Threat level, probability, explanation, and recommendation.
    """
    person_confidence = max(0.0, min(1.0, float(person_confidence or 0.0)))
    keypoints_found = max(0, int(keypoints_found or 0))

    # MVP scoring: interpretable and easy to explain to judges.
    person_score = person_confidence * 50.0
    keypoint_score = min(keypoints_found / 6.0, 1.0) * 30.0
    motion_score = 0.0
    if motion_near_human_sign:
        motion_score = 20.0
    elif motion_detected:
        motion_score = 12.0

    occlusion_bonus = 0.0
    if occlusion_detected and (keypoints_found >= 2 or weak_person_candidate or silhouette_detected):
        occlusion_bonus = 8.0

    silhouette_score = 6.0 if silhouette_detected else 0.0

    human_probability = int(round(min(100.0, person_score + keypoint_score + motion_score + occlusion_bonus + silhouette_score)))

    names = _clean_keypoint_names(keypoint_names)
    explanation: List[str] = []

    if person_detected:
        explanation.append(f"Full/mostly visible person detected by YOLO ({person_confidence:.2f} confidence).")
    elif weak_person_candidate:
        explanation.append(f"Weak person-like YOLO evidence found ({person_confidence:.2f} confidence).")
    else:
        explanation.append("No confident full-person box detected.")

    if keypoints_found > 0:
        preview = ", ".join(names[:6]) if names else "human pose keypoints"
        explanation.append(f"Visible human keypoints: {keypoints_found} ({preview}).")
    else:
        explanation.append("No reliable pose keypoints found.")

    if silhouette_detected:
        explanation.append("Human-like contour/silhouette evidence detected by fallback logic.")

    if motion_near_human_sign:
        explanation.append("Motion overlaps or appears close to human evidence.")
    elif motion_detected:
        explanation.append("Motion detected in the scene, but not clearly linked to a human sign.")
    else:
        explanation.append("No significant motion detected.")

    if occlusion_detected:
        explanation.append("Partial visibility / occlusion pattern detected: body evidence exists without a clear full-body box.")

    # Threat decision tree.
    # Green means no human evidence and no relevant motion.
    if not person_detected and not weak_person_candidate and keypoints_found == 0 and not silhouette_detected and not motion_detected:
        level = "Green"
        status = "CLEAR AREA"
        recommendation = "No action needed"
        priority = "LOW"
    # Red means hidden survivor pattern: fragment + motion, or high probability under occlusion.
    elif (
        (occlusion_detected and keypoints_found >= 2 and motion_near_human_sign)
        or (not person_detected and keypoints_found >= 4 and motion_detected)
        or (human_probability >= 75 and (occlusion_detected or motion_near_human_sign))
    ):
        level = "Red"
        status = "CRITICAL HIDDEN SURVIVOR"
        recommendation = "High priority rescue check"
        priority = "HIGH"
    # Orange means strong suspicion: fragment + any motion, or full person with motion.
    elif (
        (keypoints_found >= 2 and motion_detected)
        or (weak_person_candidate and motion_detected)
        or (person_detected and motion_detected)
        or human_probability >= 60
    ):
        level = "Orange"
        status = "POSSIBLE HIDDEN SURVIVOR"
        recommendation = "Check this area first"
        priority = "MEDIUM-HIGH"
    # Yellow means weak human signs: partial body evidence or low confidence person.
    elif person_detected or weak_person_candidate or keypoints_found >= 2 or silhouette_detected or human_probability >= 30:
        level = "Yellow"
        status = "POSSIBLE HUMAN FRAGMENT"
        recommendation = "Check this area"
        priority = "MEDIUM"
    else:
        level = "Green"
        status = "CLEAR AREA"
        recommendation = "Monitor this area"
        priority = "LOW"

    explanation.append(f"Rescue priority: {priority}.")

    return ThreatResult(
        threat_level=level,
        status=status,
        human_probability=human_probability,
        explanation=explanation,
        recommendation=recommendation,
        rescue_priority=priority,
    )
