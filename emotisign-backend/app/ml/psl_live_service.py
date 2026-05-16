"""
psl_live_service.py - Real-time PSL Alphabet Recognition Service

Provides live webcam-based Pakistan Sign Language alphabet recognition using:
- MediaPipe Hands for single-hand landmark detection (21 landmarks)
- AlphabetClassifier (42 → 128 → 64 → num_classes) feedforward neural network
- Translation and scale invariant coordinate normalization
- Per-session instances (MediaPipe Hands is not thread-safe across sessions)

Architecture matches PSL/model.py exactly.
Normalization matches PSL/preprocessor.py:normalize_hand_coords() exactly.
"""

import os
import json
import time
import base64
import logging
from typing import Dict, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL — must match PSL/model.py exactly
# =============================================================================

class AlphabetClassifier(nn.Module):
    """
    Feedforward neural network for PSL alphabet classification.
    Architecture: Input(42) → FC1(128) → BN → ReLU → Dropout(0.3)
                           → FC2(64)  → BN → ReLU → Dropout(0.3)
                           → FC3(num_classes)
    """

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        num_classes: int = 23,
        dropout: float = 0.3,
    ):
        super(AlphabetClassifier, self).__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


# =============================================================================
# SERVICE
# =============================================================================

class PSLLiveService:
    """
    Service for real-time PSL alphabet recognition via webcam frames.

    Each WebSocket session creates a fresh instance — MediaPipe Hands is
    not thread-safe across concurrent sessions.
    """

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
    ):
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.model: Optional[AlphabetClassifier] = None
        self.label_map: Optional[Dict[int, str]] = None
        self.hands = None  # MediaPipe Hands instance
        self._face_emotion = None  # lazy FaceEmotionDetector per session

        # Session statistics
        self.total_frames: int = 0
        self.frames_with_hands: int = 0
        self.predictions_made: int = 0
        self.total_confidence: float = 0.0
        self.session_start: float = time.time()

        logger.info(f"PSLLiveService created (device={device}, threshold={confidence_threshold})")

    async def load_model(self) -> None:
        """Load AlphabetClassifier checkpoint and initialize MediaPipe Hands."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PSL model not found: {self.model_path}")
        if not os.path.exists(self.label_map_path):
            raise FileNotFoundError(f"PSL label map not found: {self.label_map_path}")

        logger.info(f"Loading PSL AlphabetClassifier from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        num_classes = checkpoint["num_classes"]
        label_map_raw = checkpoint.get("label_map", {})

        self.model = AlphabetClassifier(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"PSL model loaded: {num_classes} classes")

        # Load label map from JSON (string keys)
        with open(self.label_map_path, encoding="utf-8") as f:
            lm_json = json.load(f)
        self.label_map = {int(k): v for k, v in lm_json.items()}

        # Initialize MediaPipe Hands
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipe Hands initialized")

    def _get_face_emotion_detector(self):
        if self._face_emotion is None:
            from app.ml.face_emotion_service import FaceEmotionDetector
            self._face_emotion = FaceEmotionDetector()
        return self._face_emotion

    def _emotion_payload(self, frame: np.ndarray) -> dict:
        try:
            det = self._get_face_emotion_detector().detect_from_frame(frame)
            return {
                "emotion": det["emotion"],
                "emotion_emoji": det["emotion_emoji"],
                "emotion_scores": det["emotion_scores"],
                "face_detected": det["face_detected"],
            }
        except Exception:
            return {
                "emotion": "neutral",
                "emotion_emoji": "😐",
                "emotion_scores": {"happy": 0.1, "sad": 0.1, "neutral": 0.7},
                "face_detected": False,
            }

    def normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply translation and scale invariant normalization.

        Mirrors PSL/preprocessor.py:normalize_hand_coords() exactly:
        1. Reshape to (21, 2)
        2. Subtract centroid
        3. Divide by max bounding box dimension
        4. Return flat (42,) float32

        Returns zero vector for degenerate case (all landmarks identical).
        """
        if coords.shape == (42,):
            pts = coords.reshape(21, 2)
        else:
            pts = coords.copy()

        centroid = pts.mean(axis=0)
        translated = pts - centroid

        min_xy = translated.min(axis=0)
        max_xy = translated.max(axis=0)
        bbox_size = max_xy - min_xy
        max_dim = bbox_size.max()

        if max_dim < 1e-9:
            logger.warning("Degenerate hand pose: all landmarks identical")
            return np.zeros(42, dtype=np.float32)

        scaled = translated / max_dim
        return scaled.flatten().astype(np.float32)

    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Extract 21 hand landmarks from a BGR frame using MediaPipe Hands.

        Returns:
            Tuple of (normalized_coords_42, raw_landmarks_list) or None if no hand.
            raw_landmarks_list is [[x_px, y_px], ...] for frontend visualization.
        """
        if self.hands is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lms = result.multi_hand_landmarks[0]

        # Pixel coordinates
        pixel_coords = np.array(
            [[lm.x * w, lm.y * h] for lm in lms.landmark],
            dtype=np.float32,
        )  # shape (21, 2)

        raw_landmarks = pixel_coords.tolist()  # [[x, y], ...] for frontend
        normalized = self.normalize_coordinates(pixel_coords.flatten())

        return normalized, raw_landmarks

    async def predict(self, frame: np.ndarray) -> Dict:
        """
        Process a single BGR frame and return a prediction event dict.

        Returns one of:
          {"event": "no_hand", "message": ..., "timestamp": ...}
          {"event": "low_confidence", "data": {...}, "timestamp": ...}
          {"event": "result", "data": {...}, "timestamp": ...}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.total_frames += 1
        ts = time.time()

        extraction = self.extract_hand_landmarks(frame)
        if extraction is None:
            return {
                "event": "no_hand",
                "message": "No hand detected in frame",
                "timestamp": ts,
            }

        normalized, raw_landmarks = extraction
        self.frames_with_hands += 1

        # Inference
        x = torch.from_numpy(normalized).float().unsqueeze(0)  # (1, 42)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze()  # (num_classes,)

        conf, idx = probs.max(0)
        confidence = conf.item()
        label_idx = idx.item()
        label = self.label_map.get(label_idx, f"class_{label_idx}")

        emotion_info = self._emotion_payload(frame)

        if confidence < self.confidence_threshold:
            return {
                "event": "low_confidence",
                "data": {
                    "predicted_label": label,
                    "confidence": confidence,
                    **emotion_info,
                },
                "timestamp": ts,
            }

        self.predictions_made += 1
        self.total_confidence += confidence

        return {
            "event": "result",
            "data": {
                "predicted_label": label,
                "urdu_text": label,  # label IS the Urdu character from the dataset
                "confidence": confidence,
                "landmarks": raw_landmarks,
                **emotion_info,
            },
            "timestamp": ts,
        }

    def get_session_stats(self) -> Dict:
        """Return session statistics."""
        avg_conf = (
            self.total_confidence / self.predictions_made
            if self.predictions_made > 0
            else 0.0
        )
        return {
            "total_frames": self.total_frames,
            "frames_with_hands": self.frames_with_hands,
            "predictions_made": self.predictions_made,
            "average_confidence": round(avg_conf, 4),
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if self._face_emotion is not None:
            self._face_emotion.close()
            self._face_emotion = None


# =============================================================================
# FACTORY — new instance per WebSocket session (not a singleton)
# =============================================================================

def get_psl_live_service() -> PSLLiveService:
    """
    Create a new PSLLiveService instance for a WebSocket session.

    NOT a singleton — MediaPipe Hands is not thread-safe across concurrent
    WebSocket sessions. Each session gets its own MediaPipe instance.
    """
    model_path = os.getenv(
        "PSL_LIVE_MODEL_PATH",
        "app/ml/models/psl/alphabet_classifier.pt",
    )
    label_map_path = os.getenv(
        "PSL_LIVE_LABEL_MAP_PATH",
        "app/ml/models/psl/label_map.json",
    )
    device = os.getenv("ML_DEVICE", "cpu")
    confidence_threshold = float(os.getenv("PSL_CONFIDENCE_THRESHOLD", "0.70"))

    return PSLLiveService(
        model_path=model_path,
        label_map_path=label_map_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
