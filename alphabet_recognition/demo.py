"""
Real-Time Webcam Demo for ANN-Based PSL Alphabet Recognition System

Uses MediaPipe Hands for landmark detection and the trained AlphabetClassifier
for real-time alphabet recognition.

Controls:
    q — quit

Display:
    - Predicted alphabet label (green ≥ 0.85, yellow 0.70–0.84)
    - Confidence percentage
    - Hand skeleton overlay
    - FPS counter
    - "No hand detected" when no hand is visible
    - "Press 'q' to quit" instruction
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import time
import logging
from pathlib import Path
from typing import Dict

from config import CONFIDENCE_THRESHOLD, MODEL_PATH
from model import AlphabetClassifier
from preprocessor import normalize_hand_coords

logger = logging.getLogger(__name__)


def _load_model_and_labels(model_path: str = MODEL_PATH):
    """Load model and label map from checkpoint."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train.py first to train the model."
        )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint['num_classes']
        label_map   = checkpoint['label_map']
        state_dict  = checkpoint['model_state_dict']
    else:
        state_dict  = checkpoint
        num_classes = state_dict['fc3.weight'].shape[0]
        label_map   = {i: str(i) for i in range(num_classes)}
        logger.warning("Legacy checkpoint — label names unavailable.")

    model = AlphabetClassifier(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded: {num_classes} classes")
    return model, label_map


def run_demo(
    model_path: str = MODEL_PATH,
    camera_index: int = 0,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> None:
    """
    Run real-time webcam demo for PSL alphabet recognition.

    Args:
        model_path:   Path to trained model checkpoint
        camera_index: Webcam device index (default 0)
        threshold:    Confidence threshold (default 0.70)
    """
    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, label_map = _load_model_and_labels(model_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        logger.error("Failed to open webcam")
        return

    print("\n" + "=" * 60)
    print("PSL Alphabet Recognition — Real-Time Demo")
    print("=" * 60)
    print(f"  Classes          : {len(label_map)}")
    print(f"  Confidence thresh: {threshold:.2f}")
    print("  Press 'q' to quit")
    print("=" * 60 + "\n")

    fps_t0      = time.time()
    fps_frames  = 0
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam")
            break

        frame = cv2.flip(frame, 1)          # mirror
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        label      = "No hand detected"
        confidence = 0.0
        color      = (128, 128, 128)        # grey

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]

            # Extract 21 (x, y) pixel coords → flat (42,)
            coords = np.array(
                [[lm.x * w, lm.y * h] for lm in lms.landmark],
                dtype=np.float32,
            ).flatten()

            normalized = normalize_hand_coords(coords)

            with torch.no_grad():
                inp    = torch.from_numpy(normalized).float().unsqueeze(0)
                probs  = F.softmax(model(inp), dim=1).squeeze()
                conf, idx = probs.max(0)
                confidence = conf.item()

            if confidence >= 0.85:
                label = label_map[idx.item()]
                color = (0, 220, 0)         # green — high confidence
            elif confidence >= threshold:
                label = label_map[idx.item()]
                color = (0, 220, 220)       # yellow — medium confidence
            else:
                label = "Low Confidence"
                color = (0, 165, 255)       # orange

            # Draw hand skeleton
            mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        # ── Overlay ───────────────────────────────────────────────────────────
        # Predicted label (large)
        cv2.putText(frame, label,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        # Confidence percentage
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Quit instruction
        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # FPS
        fps_frames += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps    = fps_frames / elapsed
            fps_t0 = time.time()
            fps_frames = 0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("PSL Alphabet Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDemo closed.")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    run_demo()
