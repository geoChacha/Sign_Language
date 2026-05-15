"""
test_openpose_json.py — Test model on raw OpenPose JSON keypoint files.

Maps words_dataset folders to Urdu labels by matching file counts to DB counts.

Usage:
    python test_openpose_json.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from config import MODEL_PATH, LABEL_MAP_PATH, DROPOUT


class PSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x): return self.net(x)


def normalize_hand(pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.copy()
    if not np.any(pts != 0):
        return pts.flatten()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_features(json_path: Path) -> np.ndarray | None:
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    people = data.get('people', [])
    if not people:
        return None
    p = people[0]

    rhand_raw = np.array(p.get('hand_right_keypoints_2d', [0]*63), dtype=np.float32)
    rhand_xy  = rhand_raw.reshape(21, 3)[:, :2]

    lhand_raw = np.array(p.get('hand_left_keypoints_2d', [0]*63), dtype=np.float32)
    lhand_xy  = lhand_raw.reshape(21, 3)[:, :2]

    pose_raw = np.array(p.get('pose_keypoints_2d', [0]*75), dtype=np.float32)
    pose_pts = pose_raw.reshape(25, 3)[:, :2]
    key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 0, 1, 2, 3, 4]
    pose_kp = np.zeros(26, dtype=np.float32)
    for i, idx in enumerate(key_indices):
        if idx < len(pose_pts):
            pose_kp[i*2]   = pose_pts[idx, 0]
            pose_kp[i*2+1] = pose_pts[idx, 1]
    pose_kp /= 1280.0

    return np.concatenate([normalize_hand(rhand_xy), normalize_hand(lhand_xy), pose_kp])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = PSLClassifier(checkpoint["input_dim"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(LABEL_MAP_PATH, encoding='utf-8') as f:
        meta = json.load(f)
    label_map = {int(k): v for k, v in meta["label_map"].items()}
    mean = np.array(meta["mean"], dtype=np.float32)
    std  = np.array(meta["std"],  dtype=np.float32)

    # Map folders to Urdu labels by matching file counts
    words_dir = Path("PSL_dataset/datasets/words_dataset")
    conn = sqlite3.connect('main_dataset.db')
    cursor = conn.cursor()
    cursor.execute("SELECT label, COUNT(*) FROM poseDataset GROUP BY label ORDER BY COUNT(*) DESC")
    db_counts = cursor.fetchall()
    conn.close()

    folders = sorted(
        [f for f in words_dir.iterdir() if f.is_dir()],
        key=lambda f: len(list(f.rglob("*_keypoints.json"))),
        reverse=True
    )

    print("\nFolder → Label mapping:")
    folder_label = {}
    for (label, count), folder in zip(db_counts, folders):
        n = len(list(folder.rglob("*_keypoints.json")))
        folder_label[folder] = label
        print(f"  {n:3d} files → {label}")

    # Collect all samples
    samples = []
    for folder, label in folder_label.items():
        for jf in sorted(folder.rglob("*_keypoints.json")):
            samples.append((jf, label))

    print(f"\nTotal: {len(samples)} samples")

    y_true, y_pred = [], []
    failed = 0

    for json_file, true_label in samples:
        feats = extract_features(json_file)
        if feats is None:
            failed += 1
            continue

        feats_norm = np.clip((feats - mean) / (std + 1e-8), -5.0, 5.0)
        x = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            pred  = label_map[int(np.argmax(probs))]

        y_true.append(true_label)
        y_pred.append(pred)

    print(f"Processed: {len(y_true)}  Failed: {failed}")
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy on OpenPose JSON files: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
