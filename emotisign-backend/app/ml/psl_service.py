"""
psl_service.py - PSL (Pakistan Sign Language) Recognition Service

Provides sign-to-text translation for 12 Urdu signs using:
- MediaPipe Hands + Pose keypoint extraction (110-dim features)
- MLP classifier (PSLClassifier) trained on poseDataset
- Frame-by-frame prediction with weighted majority voting

Architecture matches PSL INTE/train.py exactly.
Feature extraction matches PSL INTE/demo.py (table="pose").
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound / blocking work (MediaPipe + OpenCV)
_psl_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="psl_worker")


# =========================
# MODEL (must match train.py)
# =========================

class PSLClassifier(nn.Module):
    """MLP classifier for PSL signs. Architecture must match train.py."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# SERVICE
# =========================

class PSLModelService:
    """
    Service for PSL (Pakistan Sign Language) model inference.

    Handles:
    - Model loading and initialization
    - Per-frame feature extraction using MediaPipe Hands + Pose
    - Video processing with weighted majority voting
    - Returns Urdu label strings from label_map.json
    """

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.60,
    ):
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.model: Optional[PSLClassifier] = None
        self.label_map: Optional[Dict[int, str]] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.input_dim: int = 110
        self.num_classes: int = 12

        logger.info(f"PSLModelService initialized with device={device}")

    async def load_model(self):
        """Load PSL model checkpoint and label_map.json."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PSL model not found: {self.model_path}")
        if not os.path.exists(self.label_map_path):
            raise FileNotFoundError(f"PSL label map not found: {self.label_map_path}")

        logger.info(f"Loading PSL model from {self.model_path}")
        # weights_only=False required — checkpoint contains Python objects
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.input_dim = checkpoint["input_dim"]
        self.num_classes = checkpoint["num_classes"]

        self.model = PSLClassifier(self.input_dim, self.num_classes)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        logger.info(
            f"PSL model loaded: input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, table={checkpoint.get('table', 'pose')}"
        )

        with open(self.label_map_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.label_map = {int(k): v for k, v in meta["label_map"].items()}
        self.mean = np.array(meta["mean"], dtype=np.float32)
        self.std = np.array(meta["std"], dtype=np.float32)
        logger.info(f"PSL label map loaded: {list(self.label_map.values())}")

    def extract_features_from_frame(self, frame: np.ndarray, mp_hands, mp_pose) -> Optional[np.ndarray]:
        """
        Extract 110-dim features from a single BGR frame.

        Feature layout (matches poseDataset / demo.py table="pose"):
          - Right hand: 21 landmarks × 2 (x,y), wrist-normalized, scaled to 150 units = 42 dims
          - Left hand:  same = 42 dims
          - Pose:       13 key landmarks × 2 (x*1000, y*1000) = 26 dims
          Total: 110 dims

        Args:
            frame: BGR image (numpy array)
            mp_hands: mediapipe Hands solution instance
            mp_pose:  mediapipe Pose solution instance

        Returns:
            110-dim float32 array, or None if no hands detected
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_res = mp_hands.process(rgb)
        pose_res = mp_pose.process(rgb)

        # ── Hand features ──────────────────────────────────────────────────
        rhand = np.zeros(42, dtype=np.float32)
        lhand = np.zeros(42, dtype=np.float32)

        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for hand_lm, handedness in zip(
                hand_res.multi_hand_landmarks, hand_res.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                coords = np.array(
                    [[lm.x, lm.y] for lm in hand_lm.landmark], dtype=np.float32
                ).flatten()
                if label == "Right":
                    rhand = coords
                else:
                    lhand = coords

        # Normalize right hand relative to wrist, scale to 150 units
        if np.any(rhand != 0):
            wrist = rhand[:2].copy()
            rhand_rel = rhand.reshape(21, 2) - wrist
            scale = np.linalg.norm(rhand_rel[9] - rhand_rel[0])
            if scale > 1e-6:
                rhand_rel = rhand_rel * (150.0 / scale)
            rhand_rel[0] = [150.0, 150.0]  # wrist anchor
            rhand = rhand_rel.flatten()

        # Normalize left hand relative to wrist, scale to 150 units
        if np.any(lhand != 0):
            wrist = lhand[:2].copy()
            lhand_rel = lhand.reshape(21, 2) - wrist
            scale = np.linalg.norm(lhand_rel[9] - lhand_rel[0])
            if scale > 1e-6:
                lhand_rel = lhand_rel * (150.0 / scale)
            lhand_rel[0] = [150.0, 150.0]  # wrist anchor
            lhand = lhand_rel.flatten()

        # Skip frames with no hand detections
        if not np.any(rhand != 0) and not np.any(lhand != 0):
            return None

        # ── Pose features ──────────────────────────────────────────────────
        pose_kp = np.zeros(26, dtype=np.float32)
        if pose_res.pose_landmarks:
            # 13 key landmarks: shoulders, elbows, wrists, hips, nose, eyes
            key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 3, 4]
            for i, idx in enumerate(key_indices):
                lm = pose_res.pose_landmarks.landmark[idx]
                pose_kp[i * 2]     = lm.x * 1000  # scale to pixel-like range
                pose_kp[i * 2 + 1] = lm.y * 1000

        return np.concatenate([rhand, lhand, pose_kp])  # 42 + 42 + 26 = 110

    async def predict_from_video(self, video_path: str) -> Dict:
        """
        Process a video file frame-by-frame and return the majority-voted PSL sign.

        Each frame's features are extracted, normalized, and classified.
        Frames with no hand detections are skipped.
        Final prediction uses weighted majority voting (sum of softmax confidences).

        Args:
            video_path: Path to video file

        Returns:
            dict with keys: recognized_text, urdu_label, confidence,
                            top_predictions, frame_count, processing_time_ms, sign_language
        """
        if self.model is None:
            raise RuntimeError("PSL model not loaded. Call load_model() first.")

        start_time = time.time()

        def _process_video_sync() -> Dict:
            """Blocking video processing — runs in thread pool."""
            import mediapipe as mp

            # Create fresh MediaPipe instances per call (thread-safe)
            mp_hands_obj = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
            mp_pose_obj = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                mp_hands_obj.close()
                mp_pose_obj.close()
                raise ValueError(f"Cannot open video file: {video_path}")

            # Collect (label_idx, confidence) pairs from valid frames
            frame_predictions: List[tuple] = []
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    try:
                        feats = self.extract_features_from_frame(
                            frame, mp_hands_obj, mp_pose_obj
                        )
                    except Exception as e:
                        logger.warning(f"Frame {frame_count} feature extraction failed: {e}")
                        continue

                    if feats is None:
                        continue  # No hands detected — skip frame

                    # Normalize using training statistics
                    feats_norm = (feats - self.mean) / self.std
                    x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        logits = self.model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    frame_predictions.append((idx, probs))  # store full prob vector

            finally:
                cap.release()
                mp_hands_obj.close()
                mp_pose_obj.close()

            logger.info(
                f"PSL video processed: {frame_count} total frames, "
                f"{len(frame_predictions)} frames with hands"
            )
            return frame_predictions, frame_count

        loop = asyncio.get_event_loop()
        frame_predictions, frame_count = await loop.run_in_executor(
            _psl_executor, _process_video_sync
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # ── Majority voting ────────────────────────────────────────────────
        if not frame_predictions:
            # No hands detected in any frame
            return {
                "recognized_text": "نامعلوم",
                "urdu_label": "نامعلوم",
                "confidence": 0.0,
                "top_predictions": [],
                "frame_count": frame_count,
                "processing_time_ms": processing_time_ms,
                "sign_language": "PSL",
            }

        # Weighted vote: sum softmax probability vectors across all valid frames
        num_classes = self.num_classes
        vote_scores = np.zeros(num_classes, dtype=np.float64)
        for _, probs in frame_predictions:
            vote_scores += probs

        # Normalize to get average probability per class
        vote_scores /= len(frame_predictions)

        # Sort by score descending
        sorted_indices = np.argsort(vote_scores)[::-1]
        best_idx = int(sorted_indices[0])
        best_conf = float(vote_scores[best_idx])

        best_label = self.label_map.get(best_idx, f"unknown_{best_idx}")

        # Build top predictions list (all classes with non-trivial scores)
        top_predictions = [
            {
                "label": self.label_map.get(int(i), f"unknown_{i}"),
                "confidence": float(vote_scores[i]),
            }
            for i in sorted_indices
            if float(vote_scores[i]) > 0.01
        ][:5]  # cap at 5

        logger.info(
            f"PSL prediction: {best_label} (conf={best_conf:.3f}, "
            f"frames_with_hands={len(frame_predictions)}/{frame_count}, "
            f"time={processing_time_ms}ms)"
        )

        return {
            "recognized_text": best_label,
            "urdu_label": best_label,
            "confidence": best_conf,
            "top_predictions": top_predictions,
            "frame_count": frame_count,
            "processing_time_ms": processing_time_ms,
            "sign_language": "PSL",
        }


# =========================
# SINGLETON
# =========================

_psl_service: Optional[PSLModelService] = None


async def get_psl_service() -> PSLModelService:
    """Get or create PSL service singleton."""
    global _psl_service

    if _psl_service is None:
        model_path = os.getenv(
            "PSL_MODEL_PATH",
            "app/ml/models/psl/psl_model.pt",
        )
        label_map_path = os.getenv(
            "PSL_LABEL_MAP_PATH",
            "app/ml/models/psl/label_map.json",
        )
        device = os.getenv("ML_DEVICE", "cpu")
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

        _psl_service = PSLModelService(
            model_path=model_path,
            label_map_path=label_map_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        await _psl_service.load_model()

    return _psl_service
