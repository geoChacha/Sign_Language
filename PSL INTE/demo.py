"""
demo.py — Real-time PSL sign recognition using webcam.

Uses MediaPipe to extract hand + pose keypoints from webcam frames,
then classifies using the trained PSLClassifier.

Usage:
    python demo.py
    python demo.py --camera 0
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from config import MODEL_PATH, LABEL_MAP_PATH, CONFIDENCE_THRESHOLD, DROPOUT


# =========================
# MODEL (must match train.py)
# =========================

class PSLClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# KEYPOINT EXTRACTION
# =========================

def extract_features(pose_res, hand_res, table: str = "pose") -> np.ndarray | None:
    """Extract features matching the training table format.

    pose table: right hand (42) + left hand (42) + pose (26) = 110 dims
    word table: right hand only (42 dims)
    """
    # Right hand: 21 landmarks × 2 (x, y)
    rhand = np.zeros(42, dtype=np.float32)
    lhand = np.zeros(42, dtype=np.float32)

    if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
        for hand_lm, handedness in zip(
            hand_res.multi_hand_landmarks, hand_res.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y])
            coords = np.array(coords, dtype=np.float32)

            if label == "Right":
                rhand = coords
            else:
                lhand = coords

    if table == "word":
        # wordDataset uses right hand only, wrist anchored at (150, 150)
        # We normalize relative to wrist
        if np.any(rhand != 0):
            wrist = rhand[:2].copy()
            rhand_rel = rhand.reshape(21, 2) - wrist
            # Scale to match training data range (~150 pixel units)
            scale = np.linalg.norm(rhand_rel[9] - rhand_rel[0])
            if scale > 1e-6:
                rhand_rel = rhand_rel * (150.0 / scale)
            rhand_rel[0] = [150.0, 150.0]  # wrist anchor
            return rhand_rel.flatten()
        return None

    else:  # pose table
        # poseDataset: Rx/Ry (right hand), Lx/Ly (left hand), Px/Py (pose)
        pose_kp = np.zeros(26, dtype=np.float32)
        if pose_res.pose_landmarks:
            # Use key pose landmarks: shoulders, elbows, wrists, hips (13 points)
            key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 3, 4]
            for i, idx in enumerate(key_indices):
                lm = pose_res.pose_landmarks.landmark[idx]
                pose_kp[i*2]   = lm.x * 1000  # scale to pixel-like range
                pose_kp[i*2+1] = lm.y * 1000

        # Normalize right hand relative to wrist (matching training format)
        if np.any(rhand != 0):
            wrist = rhand[:2].copy()
            rhand_rel = rhand.reshape(21, 2) - wrist
            scale = np.linalg.norm(rhand_rel[9] - rhand_rel[0])
            if scale > 1e-6:
                rhand_rel = rhand_rel * (150.0 / scale)
            rhand_rel[0] = [150.0, 150.0]
            rhand = rhand_rel.flatten()

        if np.any(lhand != 0):
            wrist = lhand[:2].copy()
            lhand_rel = lhand.reshape(21, 2) - wrist
            scale = np.linalg.norm(lhand_rel[9] - lhand_rel[0])
            if scale > 1e-6:
                lhand_rel = lhand_rel * (150.0 / scale)
            lhand_rel[0] = [150.0, 150.0]
            lhand = lhand_rel.flatten()

        return np.concatenate([rhand, lhand, pose_kp])


# =========================
# URDU TEXT RENDERING
# =========================

# Try to load a font that supports Urdu/Arabic script
_FONT = None
_FONT_LARGE = None

def _get_font(size: int):
    global _FONT, _FONT_LARGE
    # Common fonts that support Urdu on Windows
    candidates = [
        "C:/Windows/Fonts/NotoNaskhArabic-Regular.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_urdu(frame: np.ndarray, text: str, pos: tuple,
                 font_size: int = 32, color=(255, 255, 255)) -> np.ndarray:
    """Render Urdu/Arabic text on a frame using Pillow (supports RTL)."""
    # Reshape and apply bidi for correct Urdu rendering
    try:
        reshaped = arabic_reshaper.reshape(text)
        display_text = get_display(reshaped)
    except Exception:
        display_text = text

    font = _get_font(font_size)

    # Convert frame to PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Shadow
    draw.text((pos[0]+2, pos[1]+2), display_text, font=font, fill=(0, 0, 0))
    draw.text(pos, display_text, font=font, fill=tuple(reversed(color)))  # PIL uses RGB

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _text(frame, text, pos, scale=0.8, color=(255,255,255), thickness=2):
    """Fallback ASCII text using OpenCV."""
    cv2.putText(frame, text, (pos[0]+2, pos[1]+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_bar(frame, fill: float, label: str):
    h, w = frame.shape[:2]
    bx0, bx1 = 10, w - 10
    by = h - 24
    cv2.rectangle(frame, (bx0, by), (bx1, by+16), (50,50,50), -1)
    fw = int((bx1-bx0) * fill)
    color = (0, 200, 80) if fill >= 1.0 else (0, 140, 255)
    if fw > 0:
        cv2.rectangle(frame, (bx0, by), (bx0+fw, by+16), color, -1)
    cv2.rectangle(frame, (bx0, by), (bx1, by+16), (180,180,180), 1)
    cv2.putText(frame, label, (bx0+4, by+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


# =========================
# DEMO
# =========================

def run_demo(camera_index: int = 0):
    # Load model
    if not __import__('pathlib').Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run: python train.py")
        sys.exit(1)

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    table      = checkpoint.get("table", "pose")
    input_dim  = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]

    model = PSLClassifier(input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    label_map = {int(k): v for k, v in meta["label_map"].items()}
    mean = np.array(meta["mean"], dtype=np.float32)
    std  = np.array(meta["std"],  dtype=np.float32)

    print(f"  Table: {table}Dataset  |  Classes: {list(label_map.values())}")

    # MediaPipe
    mp_pose_obj  = mp.solutions.pose.Pose(static_image_mode=False)
    mp_hands_obj = mp.solutions.hands.Hands(max_num_hands=2)
    mp_draw      = mp.solutions.drawing_utils

    # Webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        sys.exit(1)

    print("PSL Demo running — press 'q' to quit\n")

    # Smoothing: keep last N predictions and vote
    SMOOTH_N = 10
    pred_buffer: deque = deque(maxlen=SMOOTH_N)
    current_label = "Waiting..."
    current_conf  = 0.0
    fps = 0.0
    frame_times: deque = deque(maxlen=30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = mp_pose_obj.process(rgb)
        hand_res = mp_hands_obj.process(rgb)

        # Draw landmarks
        if pose_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks,
                                   mp.solutions.pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(
                                       color=(80,80,255), thickness=1, circle_radius=2))
        if hand_res.multi_hand_landmarks:
            for hl in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl,
                                       mp.solutions.hands.HAND_CONNECTIONS,
                                       landmark_drawing_spec=mp_draw.DrawingSpec(
                                           color=(0,220,0), thickness=2, circle_radius=3))

        # Extract features
        feats = extract_features(pose_res, hand_res, table)

        if feats is not None and np.any(feats != 0):
            # Normalize using training stats
            feats_norm = (feats - mean) / std
            x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0].numpy()
                idx    = int(np.argmax(probs))
                conf   = float(probs[idx])

            pred_buffer.append((idx, conf))

            # Majority vote over buffer
            if len(pred_buffer) >= SMOOTH_N // 2:
                votes = {}
                for pidx, pconf in pred_buffer:
                    votes[pidx] = votes.get(pidx, 0) + pconf
                best_idx = max(votes, key=votes.get)
                best_conf = votes[best_idx] / len(pred_buffer)

                if best_conf >= CONFIDENCE_THRESHOLD:
                    current_label = label_map[best_idx]
                    current_conf  = best_conf
                else:
                    current_label = "Unknown"
                    current_conf  = best_conf
        else:
            pred_buffer.clear()
            current_label = "No hands detected"
            current_conf  = 0.0

        # Overlay
        box_color = (0, 160, 60) if current_label not in ("Unknown", "No hands detected", "Waiting...") else (0, 60, 180)
        cv2.rectangle(frame, (8, 8), (500, 110), (20,20,20), -1)
        cv2.rectangle(frame, (8, 8), (500, 110), box_color, 2)

        # Render Urdu label with proper RTL support
        frame = _render_urdu(frame, f"اشارہ: {current_label}", (16, 15), font_size=36)
        _text(frame, f"Conf: {current_conf:.2f}", (16, 95), scale=0.65, color=(200,200,200))

        # FPS (ASCII — no Urdu needed)
        fps_text = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        _text(frame, fps_text, (frame.shape[1] - tw - 12, 30), scale=0.55, color=(180,255,180))

        # Buffer fill bar
        fill = min(len(pred_buffer) / SMOOTH_N, 1.0)
        _draw_bar(frame, fill, f"Smoothing: {len(pred_buffer)}/{SMOOTH_N} frames")

        cv2.imshow("PSL Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_times.append(time.time() - t0)
        if len(frame_times) > 1:
            fps = 1.0 / (sum(frame_times) / len(frame_times))

    cap.release()
    mp_pose_obj.close()
    mp_hands_obj.close()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_demo(args.camera)
