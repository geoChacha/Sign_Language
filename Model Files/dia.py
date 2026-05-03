"""
diagnose_video.py - Check what MediaPipe extracts from any video.

Run:
    python diagnose_video.py --video path/to/your/video.mp4

Shows:
- Per-frame detection rates for hands and pose
- Saves an annotated video so you can see exactly what MediaPipe sees
- Runs the model and shows top-5 predictions with confidence
- Tells you whether the failure is in extraction or in the model
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

import config as cfg

# MediaPipe setup
try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    _mp_drawing  = mp.solutions.drawing_utils
    _USE_LEGACY  = True
except AttributeError:
    _USE_LEGACY  = False
    print("[WARN] mediapipe >= 0.10 - landmark overlay disabled")


def extract_and_annotate(video_path: str, out_video_path: str):
    """
    Extract keypoints frame by frame, draw landmarks, save annotated video.
    Returns the keypoint sequence and per-frame detection stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {W}x{H}  {fps:.1f}fps  {total} frames  ({total/fps:.1f}s)")

    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    # CLAHE preprocessor
    def preprocess(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    keypoints  = []
    lh_frames  = 0
    rh_frames  = 0
    pose_frames = 0
    frame_idx  = 0

    if _USE_LEGACY:
        holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        proc  = preprocess(frame)
        rgb   = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        lh_arr   = np.zeros(63,  np.float32)
        rh_arr   = np.zeros(63,  np.float32)
        pose_arr = np.zeros(99,  np.float32)

        if _USE_LEGACY:
            result = holistic.process(rgb)

            if result.left_hand_landmarks:
                lh_arr = np.array([[l.x, l.y, l.z]
                    for l in result.left_hand_landmarks.landmark], np.float32).flatten()
                lh_frames += 1
                _mp_drawing.draw_landmarks(frame, result.left_hand_landmarks,
                    _mp_holistic.HAND_CONNECTIONS)

            if result.right_hand_landmarks:
                rh_arr = np.array([[l.x, l.y, l.z]
                    for l in result.right_hand_landmarks.landmark], np.float32).flatten()
                rh_frames += 1
                _mp_drawing.draw_landmarks(frame, result.right_hand_landmarks,
                    _mp_holistic.HAND_CONNECTIONS)

            if result.pose_landmarks:
                pose_arr = np.array([[l.x, l.y, l.z]
                    for l in result.pose_landmarks.landmark], np.float32).flatten()[:99]
                pose_frames += 1
                _mp_drawing.draw_landmarks(frame, result.pose_landmarks,
                    _mp_holistic.POSE_CONNECTIONS)

        # Status overlay on frame
        lh_status   = "LH: OK"  if lh_arr.sum()   != 0 else "LH: MISS"
        rh_status   = "RH: OK"  if rh_arr.sum()   != 0 else "RH: MISS"
        pose_status = "Pose: OK" if pose_arr.sum() != 0 else "Pose: MISS"
        lh_color    = (0, 255, 0)   if lh_arr.sum()   != 0 else (0, 0, 255)
        rh_color    = (0, 255, 0)   if rh_arr.sum()   != 0 else (0, 0, 255)
        pose_color  = (0, 255, 0)   if pose_arr.sum() != 0 else (0, 0, 255)

        cv2.putText(frame, lh_status,   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lh_color,   2)
        cv2.putText(frame, rh_status,   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rh_color,   2)
        cv2.putText(frame, pose_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)
        cv2.putText(frame, f"Frame {frame_idx}/{total}",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)
        keypoints.append(np.concatenate([lh_arr, rh_arr, pose_arr]))

    cap.release()
    writer.release()
    if _USE_LEGACY:
        holistic.close()

    return np.array(keypoints, np.float32), lh_frames, rh_frames, pose_frames, frame_idx


def run_model(keypoints: np.ndarray, vocab: dict, device):
    import torch
    from torch.cuda.amp import autocast
    from model import SignLanguageTransformer

    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("[WARN] No checkpoint found - skipping model prediction")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    mc   = ckpt["config"]
    model = SignLanguageTransformer(
        feature_dim=mc["feature_dim"], num_classes=mc["num_classes"],
        d_model=mc["d_model"], nhead=mc["nhead"], num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=0.0,
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Load temperature
    t_path = os.path.join(cfg.CHECKPOINTS_DIR, "temperature.json")
    temperature = 1.0
    if os.path.exists(t_path):
        with open(t_path) as f:
            temperature = json.load(f).get("temperature", 1.0)

    # Resample to NUM_FRAMES, then drop pose - use hands only
    T = keypoints.shape[0]
    idx = np.linspace(0, T - 1, cfg.NUM_FRAMES).astype(int)
    seq = keypoints[idx]
    seq = seq[:, :126]   # L-hand + R-hand only, drop pose

    # Check how many frames had zero hands
    lh_zero  = (seq[:, :63].sum(axis=1)   == 0).sum()
    rh_zero  = (seq[:, 63:126].sum(axis=1) == 0).sum()
    print(f"\n[Keypoints] After resampling to {cfg.NUM_FRAMES} frames:")
    print(f"  Left hand  missing: {lh_zero}/{cfg.NUM_FRAMES} frames ({lh_zero/cfg.NUM_FRAMES*100:.0f}%)")
    print(f"  Right hand missing: {rh_zero}/{cfg.NUM_FRAMES} frames ({rh_zero/cfg.NUM_FRAMES*100:.0f}%)")

    if lh_zero + rh_zero > cfg.NUM_FRAMES:
        print("\n  [!] CRITICAL: More than 50% of frames have NO hand detections.")
        print("      The model is mostly seeing zeros - predictions will be unreliable.")
        print("      Fix: ensure hands are visible, well-lit, and not cut off by frame edge.")
    else:
        print("  [OK] Hand detection rate is acceptable.")

    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad(), autocast():
        logits = model(x)
    probs = torch.softmax(logits / temperature, dim=-1)[0].cpu().numpy()
    top5  = np.argsort(probs)[::-1][:5]

    print("\n[Predictions] Top-5:")
    for rank, idx in enumerate(top5):
        gloss = vocab.get(str(idx), vocab.get(idx, f"class_{idx}"))
        bar   = "#" * int(probs[idx] * 40)
        print(f"  {rank+1}. {gloss:<20} {probs[idx]*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file to diagnose")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] File not found: {args.video}")
        sys.exit(1)

    out_dir   = cfg.CHECKPOINTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    base      = os.path.splitext(os.path.basename(args.video))[0]
    out_video = os.path.join(out_dir, f"{base}_annotated.mp4")

    print(f"\n[Diagnose] Processing: {args.video}")
    print(f"[Diagnose] Annotated output: {out_video}\n")

    keypoints, lh_f, rh_f, pose_f, total_f = extract_and_annotate(args.video, out_video)

    print(f"\n[Detection Summary] over {total_f} frames:")
    print(f"  Left hand:  {lh_f}/{total_f} ({lh_f/total_f*100:.0f}%)")
    print(f"  Right hand: {rh_f}/{total_f} ({rh_f/total_f*100:.0f}%)")
    print(f"  Pose:       {pose_f}/{total_f} ({pose_f/total_f*100:.0f}%)")

    if lh_f + rh_f == 0:
        print("\n  [!!] NO HANDS DETECTED AT ALL.")
        print("  Possible causes:")
        print("  - Hands not visible (cropped, out of frame, behind back)")
        print("  - Video resolution too low for MediaPipe")
        print("  - Very dark or overexposed lighting")
        print("  - Hands too close or too far from camera")
    elif (lh_f + rh_f) / (2 * total_f) < 0.4:
        print("\n  [!] Low hand detection rate - predictions will be unreliable.")

    # Load vocab and run model
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        import torch
        device = torch.device(cfg.DEVICE)
        run_model(keypoints, vocab, device)
    else:
        print("[WARN] vocab.json not found - skipping model prediction")

    print(f"\n[Diagnose] Watch the annotated video to see what MediaPipe detected:")
    print(f"  {out_video}")
    print("  GREEN text = detected, RED text = missed")


if __name__ == "__main__":
    main()