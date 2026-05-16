"""
Face emotion detection from webcam frames or video files.

Uses MediaPipe Face Mesh landmarks to classify expression into:
  happy | sad | neutral

Designed for signer-facing video (ASL / PSL) without extra ML dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PRIMARY_EMOTIONS = ("happy", "sad", "neutral")

# MediaPipe Face Mesh landmark indices
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_UPPER_LIP = 13
_LOWER_LIP = 14
_NOSE_TIP = 1
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

# Tunable thresholds (normalized face coordinates)
_SMILE_THRESHOLD = 0.018
_FROWN_THRESHOLD = -0.008


def emotion_emoji(emotion: str) -> str:
    return {"happy": "😊", "sad": "😢", "neutral": "😐"}.get(emotion, "❓")


def _format_analysis(
    emotion: str,
    scores: Dict[str, float],
    face_detected: bool,
    frame_samples: int = 1,
    processing_time_ms: int = 0,
) -> Dict:
    return {
        "emotion": emotion,
        "sentiment_label": _emotion_to_sentiment(emotion),
        "sentiment_score": _emotion_to_score(emotion, scores),
        "emotion_scores": scores,
        "face_detected": face_detected,
        "emotion_emoji": emotion_emoji(emotion),
        "frame_samples": frame_samples,
        "processing_time_ms": processing_time_ms,
    }


def _emotion_to_sentiment(emotion: str) -> str:
    if emotion == "happy":
        return "positive"
    if emotion == "sad":
        return "negative"
    return "neutral"


def _emotion_to_score(emotion: str, scores: Dict[str, float]) -> float:
    if emotion == "happy":
        return round(0.35 + 0.6 * scores.get("happy", 0.5), 3)
    if emotion == "sad":
        return round(-0.35 - 0.6 * scores.get("sad", 0.5), 3)
    return round((scores.get("happy", 0.33) - scores.get("sad", 0.33)) * 0.5, 3)


class FaceEmotionDetector:
    """Detect happy / sad / neutral from a BGR frame using Face Mesh geometry."""

    def __init__(self):
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def detect_from_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Analyze one BGR frame.

        Returns dict with keys: emotion, emotion_scores, face_detected, emotion_emoji
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return self._no_face_result()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return self._no_face_result()

        lm = result.multi_face_landmarks[0].landmark
        emotion, scores = self._classify_landmarks(lm)
        return {
            "emotion": emotion,
            "emotion_scores": scores,
            "face_detected": True,
            "emotion_emoji": emotion_emoji(emotion),
        }

    def detect_from_video(
        self,
        video_path: str,
        max_samples: int = 20,
    ) -> Dict:
        """Sample frames evenly across a video and aggregate emotion votes."""
        import time

        start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return _format_analysis("neutral", _default_scores("neutral"), False, 0, 0)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release()
            return _format_analysis("neutral", _default_scores("neutral"), False, 0, 0)

        step = max(1, total // max_samples)
        indices = list(range(0, total, step))[:max_samples]

        votes: List[str] = []
        score_acc = {e: 0.0 for e in PRIMARY_EMOTIONS}
        faces = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            det = self.detect_from_frame(frame)
            if det["face_detected"]:
                faces += 1
                votes.append(det["emotion"])
                for k, v in det["emotion_scores"].items():
                    score_acc[k] += v

        cap.release()
        elapsed = int((time.time() - start) * 1000)

        if not votes:
            return _format_analysis(
                "neutral",
                _default_scores("neutral"),
                False,
                len(indices),
                elapsed,
            )

        dominant = max(set(votes), key=votes.count)
        n = len(votes)
        avg_scores = {k: round(score_acc[k] / n, 3) for k in PRIMARY_EMOTIONS}
        # Boost winner slightly for display confidence
        avg_scores[dominant] = min(0.95, round(avg_scores[dominant] + 0.1, 3))

        return _format_analysis(dominant, avg_scores, True, n, elapsed)

    def _classify_landmarks(self, landmarks) -> Tuple[str, Dict[str, float]]:
        def pt(i):
            p = landmarks[i]
            return np.array([p.x, p.y], dtype=np.float32)

        mouth_left = pt(_MOUTH_LEFT)
        mouth_right = pt(_MOUTH_RIGHT)
        upper_lip = pt(_UPPER_LIP)
        lower_lip = pt(_LOWER_LIP)
        nose = pt(_NOSE_TIP)
        left_eye = pt(_LEFT_EYE_OUTER)
        right_eye = pt(_RIGHT_EYE_OUTER)

        mouth_center = (upper_lip + lower_lip) / 2.0
        corner_avg = (mouth_left + mouth_right) / 2.0
        # Corners above lip center → smile (y grows downward)
        smile_metric = float(mouth_center[1] - corner_avg[1])

        face_h = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
        smile_norm = smile_metric / face_h

        mouth_h = float(lower_lip[1] - upper_lip[1]) / face_h
        brow_metric = float((left_eye[1] + right_eye[1]) / 2.0 - nose[1]) / face_h

        # Soft scores before argmax
        happy_score = _sigmoid((smile_norm - _SMILE_THRESHOLD) * 80)
        sad_score = _sigmoid((-_SMILE_THRESHOLD - smile_norm) * 60 + max(0, mouth_h - 0.12) * 2)
        neutral_score = 1.0 - 0.5 * (happy_score + sad_score)
        neutral_score = max(0.05, neutral_score)

        scores = {
            "happy": happy_score,
            "sad": sad_score,
            "neutral": neutral_score,
        }
        total = sum(scores.values()) or 1.0
        scores = {k: round(v / total, 3) for k, v in scores.items()}

        if smile_norm >= _SMILE_THRESHOLD:
            emotion = "happy"
        elif smile_norm <= _FROWN_THRESHOLD or (mouth_h > 0.14 and brow_metric < -0.02):
            emotion = "sad"
        else:
            emotion = "neutral"

        scores[emotion] = min(0.95, round(scores[emotion] + 0.08, 3))
        renorm = sum(scores.values())
        scores = {k: round(v / renorm, 3) for k, v in scores.items()}
        return emotion, scores

    @staticmethod
    def _no_face_result() -> Dict:
        return {
            "emotion": "neutral",
            "emotion_scores": _default_scores("neutral"),
            "face_detected": False,
            "emotion_emoji": emotion_emoji("neutral"),
        }


def _default_scores(dominant: str) -> Dict[str, float]:
    base = {e: 0.1 for e in PRIMARY_EMOTIONS}
    base[dominant] = 0.7
    return base


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


_detector: Optional[FaceEmotionDetector] = None


def get_face_emotion_detector() -> FaceEmotionDetector:
    global _detector
    if _detector is None:
        _detector = FaceEmotionDetector()
    return _detector


async def detect_emotion_from_video_async(video_path: str) -> Dict:
    """Async wrapper — runs video sampling in a thread."""
    import asyncio

    loop = asyncio.get_event_loop()
    detector = get_face_emotion_detector()
    return await loop.run_in_executor(None, detector.detect_from_video, video_path)


async def detect_emotion_from_frame_async(frame_bgr: np.ndarray) -> Dict:
    import asyncio

    loop = asyncio.get_event_loop()
    detector = get_face_emotion_detector()
    return await loop.run_in_executor(None, detector.detect_from_frame, frame_bgr)
