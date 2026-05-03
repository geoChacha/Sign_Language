"""
5_inference.py - Real-time ASL recognition from webcam or video file.

Usage:
    python 5_inference.py                   # webcam
    python 5_inference.py --video path.mp4  # video file
    python 5_inference.py --video path.mp4 --no_display  # headless

Controls:
    q / ESC  - quit
    s        - save screenshot

Shows sliding-window predictions with confidence bars.
"""

import os
import sys
import json
import argparse
import collections
import time
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast

import config as cfg
from model import SignLanguageTransformer


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ──────────────────────────────────────────────────────────────────────────────

try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    _USE_LEGACY_API = True
except AttributeError:
    _USE_LEGACY_API = False

if not _USE_LEGACY_API:
    print("[WARN] MediaPipe >= 0.10 detected. Install mediapipe==0.9.3 for best results.")


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE contrast normalization in LAB space.
    Makes phone-camera frames look closer to studio-lit WLASL recordings,
    improving MediaPipe keypoint quality on real-world input.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def build_extractor():
    if _USE_LEGACY_API:
        return _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,           # best accuracy for real-world frames
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        # New tasks API - requires .task model files
        import mediapipe as mp
        VisionRunningMode = mp.tasks.vision.RunningMode

        hand_opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.VIDEO,
        )
        return {
            "hand": mp.tasks.vision.HandLandmarker.create_from_options(hand_opts),
            "pose": mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts),
            "mp": mp,
        }


def extract_keypoints_legacy(frame: np.ndarray, extractor) -> np.ndarray:
    frame  = preprocess_frame(frame)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = extractor.process(rgb)

    def lm2arr(lm, n):
        if lm is None:
            return np.zeros(n * 3, np.float32)
        return np.array([[l.x, l.y, l.z] for l in lm.landmark], np.float32).flatten()

    lh   = lm2arr(result.left_hand_landmarks,  21)
    rh   = lm2arr(result.right_hand_landmarks, 21)
    pose = lm2arr(result.pose_landmarks,       33)
    return np.concatenate([lh, rh, pose])  # (225,)


def extract_keypoints_new(frame: np.ndarray, extractor: dict, timestamp_ms: int) -> np.ndarray:
    mp   = extractor["mp"]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    lh_arr   = np.zeros(63, np.float32)
    rh_arr   = np.zeros(63, np.float32)
    pose_arr = np.zeros(99, np.float32)

    hand_res = extractor["hand"].detect_for_video(mp_image, timestamp_ms)
    for i, handedness in enumerate(hand_res.handedness):
        label = handedness[0].category_name
        pts   = np.array([[l.x, l.y, l.z] for l in hand_res.hand_landmarks[i]], np.float32).flatten()
        if label == "Left":
            lh_arr = pts
        else:
            rh_arr = pts

    pose_res = extractor["pose"].detect_for_video(mp_image, timestamp_ms)
    if pose_res.pose_landmarks:
        pts = np.array([[l.x, l.y, l.z] for l in pose_res.pose_landmarks[0]], np.float32).flatten()
        pose_arr = pts[:99]

    return np.concatenate([lh_arr, rh_arr, pose_arr])


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_vocab(device: torch.device):
    ckpt_path  = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "vocab.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    mc   = ckpt["config"]
    model = SignLanguageTransformer(
        feature_dim=mc["feature_dim"],
        num_classes=mc["num_classes"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=0.0,
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    with open(vocab_path) as f:
        vocab = json.load(f)

    return model, vocab


# ──────────────────────────────────────────────────────────────────────────────
# Prediction with TTA + temperature scaling
# ──────────────────────────────────────────────────────────────────────────────

def load_temperature(ckpt_dir: str) -> float:
    """Load calibrated temperature if available, else default 1.0."""
    t_path = os.path.join(ckpt_dir, "temperature.json")
    if os.path.exists(t_path):
        with open(t_path) as f:
            t = json.load(f).get("temperature", 1.0)
        print(f"[Inference] Loaded calibrated temperature: {t:.4f}")
        return float(t)
    return 1.0


def build_tta_variants(seq: np.ndarray, n_frames: int) -> list:
    """
    Build TTA variants of a keypoint sequence.
    Each variant applies a slightly different temporal resampling or
    spatial perturbation. We average their predictions to get a more
    robust result on out-of-distribution (real-world) input.

    Returns list of (n_frames, 225) arrays.
    """
    T = seq.shape[0]
    variants = []

    # 1. Standard centre sample
    idx = np.linspace(0, T - 1, n_frames).astype(int)
    variants.append(seq[idx])

    # 2. Slight speed-up (sample from first 85% of frames)
    end = max(int(T * 0.85), n_frames)
    idx = np.linspace(0, end - 1, n_frames).astype(int)
    variants.append(seq[idx])

    # 3. Slight slow-down (start at 15%)
    start = min(int(T * 0.15), T - n_frames)
    idx = np.linspace(start, T - 1, n_frames).astype(int)
    variants.append(seq[idx])

    # 4. Horizontal mirror (swap L/R hands, flip x)
    mirrored = seq.copy()
    lh   = mirrored[:, :63].copy()
    rh   = mirrored[:, 63:126].copy()
    pose = mirrored[:, 126:].copy()
    for block in [lh, rh, pose]:
        block[:, 0::3] = 1.0 - block[:, 0::3]
    mirrored[:, :63]    = rh
    mirrored[:, 63:126] = lh
    mirrored[:, 126:]   = pose
    idx = np.linspace(0, T - 1, n_frames).astype(int)
    variants.append(mirrored[idx])

    return variants


def predict(model, seq_buffer: list, device: torch.device,
            temperature: float = 1.0, use_tta: bool = True):
    """
    seq_buffer: list of (225,) arrays - recent keyframes
    temperature: calibrated temperature for confidence scaling
    use_tta: run 4 variants and average logits (more robust, ~4x slower)

    Returns: [(gloss_idx, prob), ...] sorted by prob descending
    """
    if len(seq_buffer) < 2:
        return []

    arr = np.array(seq_buffer, dtype=np.float32)

    if use_tta:
        variants = build_tta_variants(arr, cfg.NUM_FRAMES)
    else:
        idx = np.linspace(0, len(arr) - 1, cfg.NUM_FRAMES).astype(int)
        variants = [arr[idx]]

    # Drop pose features - use hands only (first 126 dims)
    variants = [v[:, :126] for v in variants]
    # Stack all variants into one batch for efficiency
    batch = np.stack(variants, axis=0)                     # (V, T, 126)
    x = torch.from_numpy(batch).to(device)

    with torch.no_grad(), autocast():
        logits = model(x)                                  # (V, num_classes)

    # Average logits across variants, then apply temperature
    avg_logits = logits.mean(dim=0, keepdim=True)          # (1, num_classes)
    probs = torch.softmax(avg_logits / temperature, dim=-1)[0].cpu().numpy()

    top5_idx = np.argsort(probs)[::-1][:5]
    return [(int(i), float(probs[i])) for i in top5_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, predictions: list, vocab: dict,
                 smoothed_word: str, frame_count: int, fps: float) -> np.ndarray:
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel
    panel_h = min(220, H // 2)
    cv2.rectangle(overlay, (0, 0), (360, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Title
    cv2.putText(frame, "ASL Recognition", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 180), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (W - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Top prediction (large)
    if smoothed_word:
        cv2.putText(frame, smoothed_word.upper(), (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 80), 3)

    # Top-5 bar chart
    bar_y0 = 85
    for rank, (cls_idx, prob) in enumerate(predictions[:5]):
        gloss = vocab.get(str(cls_idx), vocab.get(cls_idx, f"#{cls_idx}"))
        bar_w = int(prob * 280)
        color = (0, 200, 100) if rank == 0 else (100, 160, 255)
        y     = bar_y0 + rank * 26
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 18), color, -1)
        cv2.putText(frame, f"{gloss} {prob*100:.1f}%", (14, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # Frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file (default: webcam)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA (faster but less accurate)")
    parser.add_argument("--confidence_threshold", type=float, default=0.25,
                        help="Minimum confidence to show prediction (default 0.25)")
    parser.add_argument("--buffer_frames", type=int, default=90,
                        help="Sliding window size in frames (default 90 = 3s at 30fps)")
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)
    print(f"[Inference] Device: {device}")

    model, vocab = load_model_and_vocab(device)
    temperature  = load_temperature(cfg.CHECKPOINTS_DIR)
    use_tta      = not args.no_tta
    print(f"[Inference] TTA: {'ON' if use_tta else 'OFF'}  |  Confidence threshold: {args.confidence_threshold}")

    extractor    = build_extractor()

    source = args.video if args.video else args.cam
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    seq_buffer     = collections.deque(maxlen=args.buffer_frames)
    word_votes     = collections.deque(maxlen=10)   # temporal smoothing
    predictions    = []
    frame_count    = 0
    prev_time      = time.time()
    fps_display    = 30.0
    smoothed_word  = ""
    timestamp_ms   = 0

    print("[Inference] Running... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                break
            continue

        frame_count += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Extract keypoints
        if _USE_LEGACY_API:
            kp = extract_keypoints_legacy(frame, extractor)
        else:
            kp = extract_keypoints_new(frame, extractor, timestamp_ms)
        seq_buffer.append(kp)

        # Predict every 10 frames (reduce latency vs. accuracy tradeoff)
        if frame_count % 10 == 0 and len(seq_buffer) >= cfg.NUM_FRAMES:
            predictions = predict(model, list(seq_buffer), device,
                                  temperature=temperature, use_tta=use_tta)
            if predictions:
                top_idx, top_prob = predictions[0]
                if top_prob > args.confidence_threshold:
                    gloss = vocab.get(str(top_idx), vocab.get(top_idx, ""))
                    word_votes.append(gloss)

        # Smooth: most common in last 10 predictions
        if word_votes:
            counter = collections.Counter(word_votes)
            smoothed_word = counter.most_common(1)[0][0]

        # FPS
        now      = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        if not args.no_display:
            vis = draw_overlay(frame, predictions, vocab, smoothed_word, frame_count, fps_display)
            cv2.imshow("WLASL - ASL Recognition", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{frame_count}.png", vis)

    cap.release()
    cv2.destroyAllWindows()
    if _USE_LEGACY_API:
        extractor.close()
    else:
        extractor["hand"].close()
        extractor["pose"].close()

    print(f"[Inference] Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()