"""
1_extract_keypoints.py
======================
Step 1: Extract MediaPipe Holistic keypoints from every WLASL-100 video
and save them as .npy files.

Run:
    python 1_extract_keypoints.py

Requirements:
    pip install mediapipe opencv-python tqdm
    (mediapipe >= 0.8 with legacy solutions API, or see note for 0.10+)

Output layout:
    keypoints_100/
        <video_id>.npy    shape: (NUM_FRAMES, 225)

Each .npy file is a float32 array of shape (64, 225):
    [:, 0:63]   = left hand  (21 pts * xyz)
    [:, 63:126] = right hand (21 pts * xyz)
    [:, 126:225]= pose       (33 pts * xyz)

Missing landmark detections are filled with zeros.
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import cv2
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe import - supports both old (<0.10) and new (>=0.10) API
# ──────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    _USE_LEGACY_API = True
    print("[INFO] Using MediaPipe legacy solutions API")
except AttributeError:
    # MediaPipe >= 0.10 dropped mp.solutions; use tasks API or fallback
    _USE_LEGACY_API = False
    print("[INFO] MediaPipe >= 0.10 detected - using tasks-based extraction")
    print("[WARN] For best results install mediapipe==0.9.3:")
    print("       pip install mediapipe==0.9.3")
    # We'll handle this below with cv2-only pose estimation as fallback

import config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_missing(path: str) -> set:
    """Load video IDs listed in missing.txt."""
    missing = set()
    if not os.path.exists(path):
        return missing
    with open(path, "r") as f:
        for line in f:
            vid = line.strip().split("/")[-1].replace(".mp4", "")
            if vid:
                missing.add(vid)
    return missing


def sample_frames(cap: cv2.VideoCapture, n: int) -> list:
    """Uniformly sample n frames from a video capture."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = n
    indices = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frames.append(None)
        else:
            frames.append(frame)
    return frames


def landmarks_to_array(landmarks, n_pts: int) -> np.ndarray:
    """Convert mediapipe landmark list to flat numpy array."""
    if landmarks is None:
        return np.zeros(n_pts * 3, dtype=np.float32)
    arr = np.array([[l.x, l.y, l.z] for l in landmarks.landmark], dtype=np.float32)
    return arr.flatten()


# ──────────────────────────────────────────────────────────────────────────────
# Legacy API extractor (mediapipe < 0.10)
# ──────────────────────────────────────────────────────────────────────────────

class HolisticExtractorLegacy:
    def __init__(self):
        self.holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.holistic.process(rgb)

    def extract_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns float32 array of shape (225,)."""
        result = self.process(bgr_frame)
        lh   = landmarks_to_array(result.left_hand_landmarks,  21)
        rh   = landmarks_to_array(result.right_hand_landmarks, 21)
        pose = landmarks_to_array(result.pose_landmarks,       33)
        return np.concatenate([lh, rh, pose])

    def close(self):
        self.holistic.close()


# ──────────────────────────────────────────────────────────────────────────────
# New tasks API extractor (mediapipe >= 0.10)
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: requires downloading model bundle files manually:
#   hand_landmarker.task  - from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
#   pose_landmarker_lite.task - from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
# Place them in the same folder as this script.

class HolisticExtractorNew:
    def __init__(self):
        import mediapipe as mp
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Hand landmarker
        hand_model = "hand_landmarker.task"
        if not os.path.exists(hand_model):
            raise FileNotFoundError(
                f"'{hand_model}' not found.\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
                "and place it in the script directory."
            )
        hand_opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        self.hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)

        # Pose landmarker
        pose_model = "pose_landmarker_lite.task"
        if not os.path.exists(pose_model):
            raise FileNotFoundError(
                f"'{pose_model}' not found.\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n"
                "and place it in the script directory."
            )
        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
            running_mode=VisionRunningMode.IMAGE,
        )
        self.pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)
        self._mp = mp

    def extract_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        lh_arr = np.zeros(63, dtype=np.float32)
        rh_arr = np.zeros(63, dtype=np.float32)

        hand_result = self.hand_lm.detect(mp_image)
        for i, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            if i < len(hand_result.hand_landmarks):
                pts = np.array([[l.x, l.y, l.z] for l in hand_result.hand_landmarks[i]], dtype=np.float32).flatten()
                if label == "Left":
                    lh_arr = pts
                else:
                    rh_arr = pts

        pose_result = self.pose_lm.detect(mp_image)
        pose_arr = np.zeros(99, dtype=np.float32)
        if pose_result.pose_landmarks:
            pts = np.array([[l.x, l.y, l.z] for l in pose_result.pose_landmarks[0]], dtype=np.float32).flatten()
            pose_arr = pts[:99]

        return np.concatenate([lh_arr, rh_arr, pose_arr])

    def close(self):
        self.hand_lm.close()
        self.pose_lm.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────

def extract_video(video_path: str, extractor, n_frames: int) -> np.ndarray:
    """
    Extract keypoints from a single video.
    Returns float32 array of shape (n_frames, 225).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = sample_frames(cap, n_frames)
    cap.release()

    sequence = []
    for frame in frames:
        if frame is None:
            sequence.append(np.zeros(cfg.FEATURE_DIM, dtype=np.float32))
        else:
            kp = extractor.extract_frame(frame)
            sequence.append(kp)

    return np.array(sequence, dtype=np.float32)   # (n_frames, 225)


def main():
    parser = argparse.ArgumentParser(description="Extract WLASL keypoints")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = single process, safer)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if .npy already exists")
    args = parser.parse_args()

    os.makedirs(cfg.KEYPOINTS_DIR, exist_ok=True)

    # Load the 100-word split
    with open(cfg.JSON_100, "r") as f:
        nslt_100 = json.load(f)   # {video_id: {"action": [class_idx, split], ...}}

    # Load missing list
    missing = load_missing(cfg.MISSING_TXT)
    print(f"[INFO] Loaded {len(nslt_100)} entries from nslt_100.json")
    print(f"[INFO] {len(missing)} videos listed as missing")

    # Build list of (video_id, video_path)
    tasks = []
    for vid_id in nslt_100.keys():
        if vid_id in missing:
            continue
        video_path = os.path.join(cfg.VIDEOS_DIR, f"{vid_id}.mp4")
        if not os.path.exists(video_path):
            # Some datasets use subfolder layout
            video_path = os.path.join(cfg.VIDEOS_DIR, vid_id, f"{vid_id}.mp4")
        if not os.path.exists(video_path):
            continue
        out_path = os.path.join(cfg.KEYPOINTS_DIR, f"{vid_id}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        tasks.append((vid_id, video_path))

    print(f"[INFO] Videos to process: {len(tasks)}")

    # Initialise extractor
    try:
        if _USE_LEGACY_API:
            extractor = HolisticExtractorLegacy()
        else:
            extractor = HolisticExtractorNew()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    errors = []
    for vid_id, video_path in tqdm(tasks, desc="Extracting", unit="video"):
        try:
            kp = extract_video(video_path, extractor, cfg.NUM_FRAMES)
            if kp is None:
                errors.append(vid_id)
                continue
            out_path = os.path.join(cfg.KEYPOINTS_DIR, f"{vid_id}.npy")
            np.save(out_path, kp)
        except Exception:
            errors.append(vid_id)
            traceback.print_exc()

    extractor.close()

    print(f"\n[DONE] Extracted keypoints saved to: {cfg.KEYPOINTS_DIR}")
    print(f"       Errors: {len(errors)}")
    if errors:
        print("       Failed IDs:", errors[:20])


if __name__ == "__main__":
    main()
