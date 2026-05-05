"""
train.py — Train PSL sign classifier on the Kaggle keypoint dataset.

Uses poseDataset (right hand + left hand + pose keypoints, 12 words).
Architecture: MLP with batch norm — appropriate for static frame keypoints.
Training: standard cross-entropy with train/val split.

Usage:
    python train.py
    python train.py --table word    # use wordDataset (right hand only)
    python train.py --table pose    # use poseDataset (both hands + pose)
"""

import argparse
import json
import sqlite3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from config import DB_PATH, MODEL_PATH, LABEL_MAP_PATH, DROPOUT


# =========================
# DATA LOADING
# =========================

def load_pose_data(db_path: str) -> tuple:
    """Load poseDataset: right hand + left hand + pose keypoints."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM poseDataset")
    rows = cursor.fetchall()
    cursor.execute("PRAGMA table_info(poseDataset)")
    cols = [c[1] for c in cursor.fetchall()]
    conn.close()

    # Columns: id, Rx1..Ry21 (42), Lx1..Ly21 (42), Px1..Py13 (26), label
    feature_cols = [c for c in cols if c not in ('id', 'label')]
    label_col = cols.index('label')
    feat_indices = [cols.index(c) for c in feature_cols]

    X = np.array([[row[i] for i in feat_indices] for row in rows], dtype=np.float32)
    y = [row[label_col] for row in rows]

    return X, y


def load_word_data(db_path: str) -> tuple:
    """Load wordDataset: right hand keypoints only (21 landmarks × 2)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM wordDataset")
    rows = cursor.fetchall()
    cursor.execute("PRAGMA table_info(wordDataset)")
    cols = [c[1] for c in cursor.fetchall()]
    conn.close()

    feature_cols = [c for c in cols if c not in ('id', 'label')]
    label_col = cols.index('label')
    feat_indices = [cols.index(c) for c in feature_cols]

    X = np.array([[row[i] for i in feat_indices] for row in rows], dtype=np.float32)
    y = [row[label_col] for row in rows]

    return X, y


def normalize(X: np.ndarray) -> tuple:
    """Normalize features: subtract mean, divide by std."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


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

    def embed(self, x):
        """Return penultimate layer embedding for nearest-neighbor inference."""
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return nn.functional.normalize(x, p=2, dim=1)


# =========================
# TRAIN
# =========================

def train(table: str = "pose"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"Loading {table}Dataset...")
    if table == "pose":
        X, y = load_pose_data(DB_PATH)
    else:
        X, y = load_word_data(DB_PATH)

    print(f"  Samples: {len(X)}  |  Features: {X.shape[1]}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}")

    # Normalize
    X_norm, mean, std = normalize(X)

    # Train/val split — stratified
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    # Model
    model = PSLClassifier(X.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Dataset + loader
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    best_val_acc = 0.0
    best_state   = None
    patience_count = 0
    PATIENCE = 30

    print("\nTraining...")
    for epoch in range(300):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_acc    = accuracy_score(y_val, val_preds)

        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/len(loader):.4f}  Val={val_acc:.4f}  Best={best_val_acc:.4f}")

        if patience_count >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # Load best
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_vl).argmax(dim=1).cpu().numpy()

    print(f"\nFinal Val Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=le.classes_))

    # Save model + metadata
    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   X.shape[1],
        "num_classes": num_classes,
        "table":       table,
    }, MODEL_PATH)

    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map, "mean": mean.tolist(), "std": std.tolist()}, f, ensure_ascii=False)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["pose", "word"], default="pose",
                        help="Which table to train on (default: pose)")
    args = parser.parse_args()
    train(args.table)
