"""
train_from_json.py — Train PSL classifier using exact original repo feature pipeline.

Uses psl_features.py which implements the identical normalization as the
original PSL repo (scale.scalePoints, normalize.scaleBody, normalize.move_to_wrist).

Usage:
    python train_from_json.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import MODEL_PATH, LABEL_MAP_PATH, DROPOUT
from psl_features import extract_from_openpose_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 110   # matches poseDataset: rhand(42) + lhand(42) + pose_subset(26)


# =========================
# LOAD DATASET FROM JSON FILES
# =========================

def load_dataset() -> tuple:
    """Load all OpenPose JSON files and map to Urdu labels."""
    words_dir = Path("PSL_dataset/datasets/words_dataset")

    # Map folders to Urdu labels by matching file counts to DB counts
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

    print("Folder → Label mapping:")
    folder_label = {}
    for (label, count), folder in zip(db_counts, folders):
        n = len(list(folder.rglob("*_keypoints.json")))
        folder_label[folder] = label
        print(f"  {n:3d} files → {label}")

    X_list, y_list = [], []
    skipped = 0

    for folder, label in folder_label.items():
        for jf in sorted(folder.rglob("*_keypoints.json")):
            try:
                with open(jf, encoding='utf-8') as f:
                    data = json.load(f)
                feats = extract_from_openpose_json(data)
            except Exception:
                feats = None

            if feats is None:
                skipped += 1
                continue
            X_list.append(feats)
            y_list.append(label)

    print(f"\nLoaded: {len(X_list)} samples  Skipped: {skipped}")
    return np.array(X_list, dtype=np.float32), y_list


# =========================
# MODEL
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
# TRAIN
# =========================

def train():
    print(f"Device: {DEVICE}")
    print("Loading dataset from OpenPose JSON files...")
    X, y_str = load_dataset()

    # Encode labels
    classes = sorted(set(y_str))
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y = np.array([label_to_idx[lbl] for lbl in y_str], dtype=np.int64)

    print(f"  Features: {X.shape[1]}  Classes: {len(classes)}")

    # Normalize
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

    model     = PSLClassifier(X.shape[1], len(classes)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    PATIENCE     = 30

    print("\nTraining...")
    for epoch in range(300):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            preds    = model(X_val_t).argmax(dim=1).cpu().numpy()
            val_acc  = accuracy_score(y_val, preds)

        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Val={val_acc:.4f}  Best={best_val_acc:.4f}")

        if no_improve >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Final report
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1).cpu().numpy()
    print(f"\nFinal Val Accuracy: {accuracy_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds, target_names=classes))

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   X.shape[1],
        "num_classes": len(classes),
        "table":       "json",
    }, MODEL_PATH)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "label_map": idx_to_label,
            "mean": mean.tolist(),
            "std":  std.tolist(),
        }, f, ensure_ascii=False)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Label map saved: {LABEL_MAP_PATH}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
