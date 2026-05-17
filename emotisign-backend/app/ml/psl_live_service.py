"""
psl_live_service.py - Real-time PSL Alphabet Recognition Service

Provides live webcam-based Pakistan Sign Language alphabet recognition using:
- MediaPipe Hands for single-hand landmark detection (21 landmarks)
- AlphabetClassifier (42 → 128 → 64 → num_classes) feedforward neural network
- NEW PSL normalization: wrist-relative + landmark-9 scale + StandardScaler
- CLAHE contrast preprocessing for robust detection under real-world lighting
- Temporal smoothing via majority vote over a rolling prediction buffer
- Per-session instances (MediaPipe Hands is not thread-safe across sessions)

Normalization pipeline (matches NEW PSL/web/psl_alphabet_classifier.js exactly):
  1. Subtract wrist (landmark 0) from all 21 landmarks
  2. Divide by Euclidean distance from wrist to middle-MCP (landmark 9)
  3. Flatten to 42 values
  4. Standardize: (x - featureMean) / featureStd  (from training StandardScaler)
"""

import os
import json
import time
import collections
import logging
from typing import Dict, Optional

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

    Improvements over baseline:
      1. CLAHE preprocessing — normalises contrast before MediaPipe runs,
         improving hand detection under dim or uneven lighting.
      2. Lower confidence threshold (0.60) — the softmax distribution across
         23 classes rarely peaks above 0.80 for correct predictions; 0.60
         surfaces correct results that were previously silently dropped.
      3. Temporal smoothing — majority vote over a rolling 12-frame buffer
         eliminates per-frame flicker and absorbs transition-frame noise.
    """

    # ── Smoothing parameters ──────────────────────────────────────────────────
    # Number of recent raw predictions to keep in the vote buffer.
    SMOOTH_BUFFER_SIZE: int = 12
    # Fraction of buffer that must agree before emitting a smoothed result.
    SMOOTH_VOTE_THRESHOLD: float = 0.60

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.60,  # lowered from 0.80
    ):
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.model: Optional[AlphabetClassifier] = None
        self.label_map: Optional[Dict[int, str]] = None
        self.hands = None  # MediaPipe Hands instance
        self._face_emotion = None  # lazy FaceEmotionDetector per session

        # StandardScaler parameters loaded from checkpoint
        # Shape: (42,) each — applied after wrist-relative normalization
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

        # CLAHE instance — reused across frames (avoids re-allocation overhead)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Rolling buffer of raw label predictions for temporal smoothing.
        # Stores the predicted label string (not index) so it works across
        # model reloads without index remapping.
        self._vote_buffer: collections.deque = collections.deque(
            maxlen=self.SMOOTH_BUFFER_SIZE
        )

        # Session statistics
        self.total_frames: int = 0
        self.frames_with_hands: int = 0
        self.predictions_made: int = 0
        self.total_confidence: float = 0.0
        self.session_start: float = time.time()

        logger.info(
            f"PSLLiveService created (device={device}, "
            f"threshold={confidence_threshold}, "
            f"smooth_buffer={self.SMOOTH_BUFFER_SIZE}, "
            f"vote_threshold={self.SMOOTH_VOTE_THRESHOLD})"
        )

    async def load_model(self) -> None:
        """Load AlphabetClassifier checkpoint and initialize MediaPipe Hands."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PSL model not found: {self.model_path}")
        if not os.path.exists(self.label_map_path):
            raise FileNotFoundError(f"PSL label map not found: {self.label_map_path}")

        logger.info(f"Loading PSL AlphabetClassifier from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        num_classes = checkpoint["num_classes"]

        self.model = AlphabetClassifier(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"PSL model loaded: {num_classes} classes")

        # Load StandardScaler parameters (required for NEW PSL normalization)
        if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
            self._scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
            self._scaler_std = np.array(checkpoint["scaler_std"], dtype=np.float32)
            logger.info("PSL StandardScaler parameters loaded from checkpoint")
        else:
            # Fallback: identity transform (no standardization)
            self._scaler_mean = np.zeros(42, dtype=np.float32)
            self._scaler_std = np.ones(42, dtype=np.float32)
            logger.warning(
                "No scaler parameters in checkpoint — using identity transform. "
                "Re-run convert_new_psl_model.py to include scaler parameters."
            )

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

    # ── Improvement 1: CLAHE preprocessing ───────────────────────────────────

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast normalisation in LAB colour space.

        Converts the L (lightness) channel with a contrast-limited adaptive
        histogram equaliser before passing the frame to MediaPipe.  This
        makes hand edges visible under dim, uneven, or backlit conditions
        where the raw frame would cause MediaPipe to miss the hand entirely.

        Matches the preprocessing used in keypoint_extractor.py and
        Model Files/5_inference.py so the landmark quality is consistent
        with what the model was effectively trained against.

        Args:
            frame: BGR image (H, W, 3)
        Returns:
            Contrast-normalised BGR image (H, W, 3)
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Improvement 3: Temporal smoothing ────────────────────────────────────

    def _smooth_prediction(self, raw_label: str) -> Optional[str]:
        """
        Add raw_label to the vote buffer and return the smoothed label if a
        majority has formed, otherwise return None.

        A majority is defined as SMOOTH_VOTE_THRESHOLD fraction of the
        buffer agreeing on the same label.  This absorbs:
          - Per-frame flicker caused by micro-movements
          - Garbage predictions during hand-shape transitions
          - Occasional MediaPipe landmark jitter

        Returns:
            The winning label string if consensus reached, else None.
        """
        self._vote_buffer.append(raw_label)

        if len(self._vote_buffer) < self.SMOOTH_BUFFER_SIZE:
            # Buffer not yet full — don't emit until we have enough history
            return None

        counter = collections.Counter(self._vote_buffer)
        top_label, top_count = counter.most_common(1)[0]
        vote_fraction = top_count / len(self._vote_buffer)

        if vote_fraction >= self.SMOOTH_VOTE_THRESHOLD:
            return top_label
        return None

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

    def normalize_coordinates(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks using the NEW PSL pipeline.

        Matches NEW PSL/web/psl_alphabet_classifier.js normalizeLandmarks() exactly,
        followed by StandardScaler standardization using training statistics.

        Pipeline:
          1. Subtract wrist (landmark 0) from all 21 landmarks
          2. Compute scale = Euclidean distance from wrist to middle-MCP (landmark 9)
          3. Divide all coordinates by scale (safe: clamp to 1 if scale < 1e-6)
          4. Flatten to 42 values
          5. Standardize: (x - scaler_mean) / scaler_std

        Args:
            pixel_coords: flat (42,) array of pixel coordinates [x0,y0, x1,y1, ...]
                          OR (21, 2) array — both accepted.
        Returns:
            Normalized (42,) float32 array ready for model inference.
        """
        if pixel_coords.shape == (42,):
            pts = pixel_coords.reshape(21, 2).astype(np.float32)
        else:
            pts = pixel_coords.astype(np.float32)

        # Step 1: subtract wrist (landmark 0)
        wrist = pts[0]
        relative = pts - wrist  # (21, 2)

        # Step 2: scale = distance from wrist to middle-MCP (landmark 9)
        middle_mcp = relative[9]
        scale = float(np.hypot(middle_mcp[0], middle_mcp[1]))
        safe_scale = scale if scale > 1e-6 else 1.0

        # Step 3: divide by scale
        normalized = relative / safe_scale  # (21, 2)

        # Step 4: flatten
        features = normalized.flatten()  # (42,)

        # Step 5: standardize with training scaler
        if self._scaler_mean is not None and self._scaler_std is not None:
            std_safe = np.where(self._scaler_std > 1e-9, self._scaler_std, 1.0)
            features = (features - self._scaler_mean) / std_safe

        return features.astype(np.float32)

    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Extract 21 hand landmarks from a BGR frame using MediaPipe Hands.

        Applies CLAHE preprocessing (improvement 1) before detection so
        MediaPipe receives a contrast-normalised image.

        Returns:
            Tuple of (normalized_coords_42, raw_landmarks_list) or None if no hand.
            raw_landmarks_list is [[x_px, y_px], ...] for frontend visualization.
        """
        if self.hands is None:
            return None

        # Improvement 1: normalise contrast before MediaPipe runs
        preprocessed = self._preprocess_frame(frame)

        h, w = preprocessed.shape[:2]
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lms = result.multi_hand_landmarks[0]

        # Pixel coordinates (derived from original frame dimensions)
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

        Pipeline:
          1. Extract landmarks (with CLAHE preprocessing)
          2. Run model inference
          3. Apply confidence gate (improvement 2: threshold = 0.60)
          4. Feed raw label into vote buffer (improvement 3: smoothing)
          5. Emit "result" only when majority consensus is reached,
             "pending" while the buffer is filling, or "low_confidence" /
             "no_hand" when appropriate.

        Returns one of:
          {"event": "no_hand",       "message": ..., "timestamp": ...}
          {"event": "low_confidence","data": {...},  "timestamp": ...}
          {"event": "pending",       "data": {...},  "timestamp": ...}
          {"event": "result",        "data": {...},  "timestamp": ...}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.total_frames += 1
        ts = time.time()

        extraction = self.extract_hand_landmarks(frame)
        if extraction is None:
            # No hand — clear the vote buffer so stale votes don't bleed
            # into the next sign attempt
            self._vote_buffer.clear()
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

        # Improvement 2: lowered threshold (0.60 instead of 0.80)
        if confidence < self.confidence_threshold:
            # Below threshold — don't feed into vote buffer
            return {
                "event": "low_confidence",
                "data": {
                    "predicted_label": label,
                    "confidence": confidence,
                    **emotion_info,
                },
                "timestamp": ts,
            }

        # Improvement 3: feed into temporal vote buffer
        smoothed_label = self._smooth_prediction(label)

        if smoothed_label is None:
            # Buffer filling up — emit a "pending" event so the frontend
            # can show a subtle "detecting…" indicator without committing
            # to a letter yet
            return {
                "event": "pending",
                "data": {
                    "predicted_label": label,   # raw best guess for display
                    "confidence": confidence,
                    "buffer_fill": len(self._vote_buffer),
                    "buffer_size": self.SMOOTH_BUFFER_SIZE,
                    **emotion_info,
                },
                "timestamp": ts,
            }

        # Consensus reached — emit confirmed result
        self.predictions_made += 1
        self.total_confidence += confidence

        return {
            "event": "result",
            "data": {
                "predicted_label": smoothed_label,
                "urdu_text": smoothed_label,
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
    # Improvement 2: default threshold lowered to 0.60
    confidence_threshold = float(os.getenv("PSL_CONFIDENCE_THRESHOLD", "0.60"))

    return PSLLiveService(
        model_path=model_path,
        label_map_path=label_map_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
