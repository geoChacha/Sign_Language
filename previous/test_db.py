"""
test_db.py — Evaluate the trained model on the database using cross-validation.

Usage:
    python test_db.py
    python test_db.py --table word
"""

import argparse
import json
import sqlite3

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from config import DB_PATH, MODEL_PATH, LABEL_MAP_PATH, DROPOUT


class PSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def load_data(table: str) -> tuple:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    tname = "poseDataset" if table == "pose" else "wordDataset"
    cursor.execute(f"SELECT * FROM {tname}")
    rows = cursor.fetchall()
    cursor.execute(f"PRAGMA table_info({tname})")
    cols = [c[1] for c in cursor.fetchall()]
    conn.close()

    feature_cols = [c for c in cols if c not in ('id', 'label')]
    label_col = cols.index('label')
    feat_indices = [cols.index(c) for c in feature_cols]

    X = np.array([[row[i] for i in feat_indices] for row in rows], dtype=np.float32)
    y = [row[label_col] for row in rows]
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["pose", "word"], default="pose")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading saved model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    input_dim   = checkpoint["input_dim"]
    num_classes = checkpoint["num_classes"]

    model = PSLClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    label_map = {int(k): v for k, v in meta["label_map"].items()}
    mean = np.array(meta["mean"], dtype=np.float32)
    std  = np.array(meta["std"],  dtype=np.float32)

    print(f"Loading {args.table}Dataset...")
    X, y = load_data(args.table)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_norm = (X - mean) / std
    X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    # Map back to string labels
    pred_labels = [label_map[p] for p in preds]
    true_labels = y

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy (on full dataset): {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    # Per-class breakdown
    print("\nPer-class sample counts:")
    from collections import Counter
    counts = Counter(y)
    for cls, cnt in sorted(counts.items()):
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        print(f"  {cls:<20}: {correct:3d}/{cnt:3d}  ({100*correct/cnt:.0f}%)")


if __name__ == "__main__":
    main()
