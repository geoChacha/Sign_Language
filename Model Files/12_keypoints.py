"""
12_extract_keypoints_slp.py - Extract keypoints optimized for Sign Language Production.

Differences from 1_extract_keypoints.py (recognition):
  - Hip-centered normalization: subtracts midpoint of hips so coords are
    body-relative, not camera-relative. Avatar won't drift.
  - Scale normalization: divides by shoulder width so different signers
    map to same coordinate space on the avatar.
  - Full 225 dims always saved (hands + pose) - avatar needs pose for arms.
  - Gaussian smoothing per sequence - cleaner animation.
  - Saves to KEYPOINTS_DIR_SLP (separate from recognition keypoints).

Run:
    python 12_extract_keypoints_slp.py

Then retrain SLP model:
    python 9_slp_train.py
    python 11_export_sign_poses.py
"""

import os
import json
import time
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

try:
    import mediapipe as mp
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,          # complexity 1 is enough for production
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    HAS_MP = True
except Exception as e:
    print(f"[WARN] MediaPipe not available: {e}")
    HAS_MP = False

import config as cfg


# ── Output directory ──────────────────────────────────────────────────────────
SLP_KP_DIR = os.path.join(cfg.DATASET_ROOT, "keypoints_slp")
os.makedirs(SLP_KP_DIR, exist_ok=True)


def preprocess_frame(frame):
    """CLAHE for consistent lighting."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def extract_frame(frame, holistic):
    """Extract MediaPipe landmarks from one frame. Returns (225,) array."""
    rgb = cv2.cvtColor(preprocess_frame(frame), cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)

    lh   = np.zeros(63,  np.float32)
    rh   = np.zeros(63,  np.float32)
    pose = np.zeros(99,  np.float32)

    if result.left_hand_landmarks:
        lh = np.array([[l.x, l.y, l.z]
            for l in result.left_hand_landmarks.landmark], np.float32).flatten()

    if result.right_hand_landmarks:
        rh = np.array([[l.x, l.y, l.z]
            for l in result.right_hand_landmarks.landmark], np.float32).flatten()

    if result.pose_landmarks:
        pose = np.array([[l.x, l.y, l.z]
            for l in result.pose_landmarks.landmark], np.float32).flatten()[:99]

    return np.concatenate([lh, rh, pose])   # (225,)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Body-relative normalization for avatar rendering.

    Steps:
    1. Extract hip midpoint from pose landmarks (indices 23 and 24)
    2. Extract shoulder width for scale
    3. Subtract hip center from all x,y coords
    4. Divide by shoulder width → scale-invariant

    This makes all signers map to the same body-space coordinates
    so the avatar pose is consistent regardless of camera distance.

    Pose landmark layout (each landmark = x,y,z at indices i*3, i*3+1, i*3+2):
      index 11 = left shoulder
      index 12 = right shoulder
      index 23 = left hip
      index 24 = right hip
    """
    seq = seq.copy()
    T = seq.shape[0]

    # Pose block starts at dim 126
    pose_block = seq[:, 126:]   # (T, 99) = 33 landmarks × 3

    # Hip midpoint (landmarks 23 and 24)
    l_hip_x = pose_block[:, 23*3]
    l_hip_y = pose_block[:, 23*3+1]
    r_hip_x = pose_block[:, 24*3]
    r_hip_y = pose_block[:, 24*3+1]

    # Only compute where both hips detected
    hip_detected = (l_hip_x != 0) & (r_hip_x != 0)

    hip_cx = np.where(hip_detected, (l_hip_x + r_hip_x) / 2, 0.5)
    hip_cy = np.where(hip_detected, (l_hip_y + r_hip_y) / 2, 0.8)

    # Shoulder width for scale
    l_sh_x = pose_block[:, 11*3]
    r_sh_x = pose_block[:, 12*3]
    sh_detected = (l_sh_x != 0) & (r_sh_x != 0)
    sh_width = np.where(sh_detected, np.abs(r_sh_x - l_sh_x), 0.3)
    sh_width = np.clip(sh_width, 0.1, 1.0)   # prevent division by zero

    # Normalize each block
    for t in range(T):
        cx, cy = hip_cx[t], hip_cy[t]
        sc = sh_width[t] * 3.0   # scale factor (3 shoulder-widths = full range)

        # Left hand block (dims 0:63) — 21 landmarks
        for i in range(21):
            if seq[t, i*3] != 0:   # only if detected
                seq[t, i*3]   = (seq[t, i*3]   - cx) / sc + 0.5
                seq[t, i*3+1] = (seq[t, i*3+1] - cy) / sc + 0.5

        # Right hand block (dims 63:126)
        for i in range(21):
            base = 63 + i*3
            if seq[t, base] != 0:
                seq[t, base]   = (seq[t, base]   - cx) / sc + 0.5
                seq[t, base+1] = (seq[t, base+1] - cy) / sc + 0.5

        # Pose block (dims 126:225)
        for i in range(33):
            base = 126 + i*3
            if seq[t, base] != 0:
                seq[t, base]   = (seq[t, base]   - cx) / sc + 0.5
                seq[t, base+1] = (seq[t, base+1] - cy) / sc + 0.5

    return seq


def smooth_sequence(seq: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Gaussian smooth per channel. Only smooth non-zero channels."""
    result = seq.copy()
    for ch in range(seq.shape[1]):
        col = seq[:, ch]
        if (col != 0).sum() < 3:
            continue
        result[:, ch] = gaussian_filter1d(col, sigma)
    return result


def extract_video(video_path: str) -> np.ndarray:
    """Extract, normalize, smooth keypoints from one video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_kp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_frame(frame, _holistic)
        frames_kp.append(kp)
    cap.release()

    if not frames_kp:
        return None

    seq = np.array(frames_kp, np.float32)   # (T, 225)

    # Resample to NUM_FRAMES
    T = seq.shape[0]
    if T != cfg.NUM_FRAMES:
        idx = np.linspace(0, T-1, cfg.NUM_FRAMES).astype(int)
        seq = seq[idx]

    # Normalize + smooth
    seq = normalize_sequence(seq)
    seq = smooth_sequence(seq, sigma=1.2)

    return seq


def main():
    if not HAS_MP:
        print("[ERROR] MediaPipe required. Run on your Windows machine.")
        return

    with open(cfg.JSON_100) as f:
        nslt = json.load(f)

    # Load missing list
    missing = set()
    if os.path.exists(cfg.MISSING_TXT):
        with open(cfg.MISSING_TXT) as f_m:
            for line in f_m:
                vid = line.strip().split("/")[-1].replace(".mp4","")
                if vid: missing.add(vid)

    video_ids = [
        vid for vid, info in nslt.items()
        if vid not in missing
        and os.path.exists(os.path.join(cfg.VIDEOS_DIR, f"{vid}.mp4"))
    ]

    print(f"[SLP Extract] {len(video_ids)} videos {SLP_KP_DIR}")
    done = skipped = failed = 0

    for i, vid_id in enumerate(video_ids):
        out_path = os.path.join(SLP_KP_DIR, f"{vid_id}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        video_path = os.path.join(cfg.VIDEOS_DIR, f"{vid_id}.mp4")
        t0 = time.time()
        seq = extract_video(video_path)

        if seq is None:
            failed += 1
            print(f"  [{i+1}/{len(video_ids)}] FAILED  {vid_id}")
            continue

        np.save(out_path, seq)
        done += 1

        if (i+1) % 50 == 0 or i < 5:
            print(f"  [{i+1}/{len(video_ids)}] {vid_id}  "
                  f"shape={seq.shape}  {time.time()-t0:.1f}s")

    _holistic.close()
    print(f"\n[SLP Extract] Done: {done} extracted, {skipped} skipped, {failed} failed")
    print(f"[SLP Extract] Output: {SLP_KP_DIR}")
    print(f"\nNext: update slp_dataset.py KEYPOINTS_DIR to use keypoints_slp,")
    print(f"then run: python 9_slp_train.py")


if __name__ == "__main__":
    main()