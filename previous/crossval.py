"""
crossval.py — 5-fold cross-validation for honest accuracy estimate.

Trains and evaluates 5 times on different splits to get a reliable
accuracy number that isn't inflated by testing on training data.
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from config import DB_PATH, DROPOUT


class PSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x): return self.net(x)


def load_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM poseDataset")
    rows = cursor.fetchall()
    cursor.execute("PRAGMA table_info(poseDataset)")
    cols = [c[1] for c in cursor.fetchall()]
    conn.close()

    feature_cols = [c for c in cols if c not in ('id', 'label')]
    label_col = cols.index('label')
    feat_indices = [cols.index(c) for c in feature_cols]

    X = np.array([[row[i] for i in feat_indices] for row in rows], dtype=np.float32)
    y = [row[label_col] for row in rows]
    return X, y


def train_fold(X_tr, y_tr, input_dim, num_classes, device):
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mean) / std

    X_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_tr,   dtype=torch.long).to(device)

    model = PSLClassifier(input_dim, num_classes).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ds    = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(150):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    return model, mean, std


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading poseDataset...")

    X, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  {len(X)} samples, {len(le.classes_)} classes")
    print(f"  Classes: {list(le.classes_)}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    all_true, all_pred = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_enc)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]

        model, mean, std = train_fold(X_tr, y_tr, X.shape[1], len(le.classes_), device)
        model.eval()

        X_val_n = (X_val - mean) / std
        X_vt = torch.tensor(X_val_n, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_vt).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_val, preds)
        fold_accs.append(acc)
        all_true.extend(y_val)
        all_pred.extend(preds)
        print(f"  Fold {fold+1}: Val Acc = {acc:.4f}  ({int(acc*len(y_val))}/{len(y_val)})")

    print(f"\n{'='*50}")
    print(f"5-Fold CV Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"Min: {min(fold_accs):.4f}  Max: {max(fold_accs):.4f}")
    print(f"\nAggregated Classification Report:")
    print(classification_report(all_true, all_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
