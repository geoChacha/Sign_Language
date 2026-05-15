"""
test_video.py — Test the trained PSL model on a recorded video file.

Extracts MediaPipe keypoints frame by frame, classifies each frame,
then reports the majority prediction with confidence.

Usage:
    python test_video.py test_vid/know_psl.mp4
    python test_video.py test_vid/phone_psl.mp4
    python test_video.py test_vid/know_psl.mp4 --true جانتا
    python test_video.py test_vid/phone_psl.mp4 --true فون
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from config import MODEL_PATH, LABEL_MAP_PATH, CONFIDENCE_THRESHOLD, DROPOUT
from psl_features import extract_from_mediapipe


# =========================
# MODEL
# =========================

class PSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x): return self.net(x)


# =========================
# FEATURE EXTRACTION (via psl_features.py)
# =========================
# Uses extract_from_mediapipe() — exact original repo normalization.


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--true", default=None, help="True label (optional, for accuracy check)")
    parser.add_argument("--show", action="store_true", help="Show video with overlay while processing")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH} — run: python train.py")
        sys.exit(1)

    # Load model
    print(f"Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    input_dim   = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]

    model = PSLClassifier(input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    label_map = {int(k): v for k, v in meta["label_map"].items()}
    mean = np.array(meta["mean"], dtype=np.float32)
    std  = np.array(meta["std"],  dtype=np.float32)

    print(f"Model loaded. Classes: {list(label_map.values())}")
    print(f"\nProcessing: {video_path.name}")

    # MediaPipe
    mp_pose_obj  = mp.solutions.pose.Pose(static_image_mode=False)
    mp_hands_obj = mp.solutions.hands.Hands(max_num_hands=2)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {total_frames} frames @ {fps:.1f} FPS  ({total_frames/fps:.1f}s)  {w}×{h}")

    frame_predictions = []   # (label, confidence) per frame
    frame_idx = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = mp_pose_obj.process(rgb)
        hand_res = mp_hands_obj.process(rgb)

        feats = extract_from_mediapipe(pose_res, hand_res, w, h)

        if feats is not None:
            feats_norm = (feats - mean) / (std + 1e-8)
            x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0].numpy()
                idx    = int(np.argmax(probs))
                conf   = float(probs[idx])

            frame_predictions.append((label_map[idx], conf))
            detected_frames += 1

            if args.show:
                label_text = f"{label_map[idx]} ({conf:.2f})"
                cv2.putText(frame, label_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("PSL Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_idx += 1

    cap.release()
    mp_pose_obj.close()
    mp_hands_obj.close()
    if args.show:
        cv2.destroyAllWindows()

    # =========================
    # RESULTS
    # =========================
    print(f"\n{'='*50}")
    print(f"Frames processed : {frame_idx}")
    print(f"Frames with hands: {detected_frames} ({100*detected_frames/max(frame_idx,1):.1f}%)")

    if not frame_predictions:
        print("No hands detected in any frame — cannot classify.")
        sys.exit(0)

    # Per-label vote weighted by confidence
    vote_scores: dict = {}
    for label, conf in frame_predictions:
        vote_scores[label] = vote_scores.get(label, 0.0) + conf

    # Sort by score
    sorted_preds = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
    total_score  = sum(v for _, v in sorted_preds)

    print(f"\nPrediction breakdown:")
    for label, score in sorted_preds:
        pct = 100 * score / total_score
        bar = "█" * int(pct / 5)
        print(f"  {label:<15} {pct:5.1f}%  {bar}")

    top_label = sorted_preds[0][0]
    top_conf  = sorted_preds[0][1] / total_score

    print(f"\n{'='*50}")
    print(f"PREDICTION : {top_label}")
    print(f"CONFIDENCE : {top_conf:.4f}")

    if args.true:
        correct = (top_label == args.true)
        print(f"TRUE LABEL : {args.true}")
        print(f"RESULT     : {'✓ CORRECT' if correct else '✗ WRONG'}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
