"""Verify what features the model was actually trained on."""
import sqlite3, json, numpy as np, torch
from config import MODEL_PATH, LABEL_MAP_PATH, DROPOUT
import torch.nn as nn

class PSLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x): return self.net(x)

def normalize_hand(hand_arr):
    if not np.any(hand_arr != 0): return hand_arr
    pts = hand_arr.reshape(21, 2).copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale > 1e-6: pts /= scale
    return pts.flatten()

# Load model and stats
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
with open(LABEL_MAP_PATH, encoding='utf-8') as f:
    meta = json.load(f)
label_map = {int(k): v for k, v in meta['label_map'].items()}
mean = np.array(meta['mean'], dtype=np.float32)
std  = np.array(meta['std'],  dtype=np.float32)

model = PSLClassifier(checkpoint['input_dim'], checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Load raw DB data and apply train.py normalization
conn = sqlite3.connect('main_dataset.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM poseDataset LIMIT 20")
rows = cursor.fetchall()
cursor.execute("PRAGMA table_info(poseDataset)")
cols = [c[1] for c in cursor.fetchall()]
conn.close()

feat_cols = [c for c in cols if c not in ('id','label')]
label_col = cols.index('label')
feat_indices = [cols.index(c) for c in feat_cols]

correct = 0
for row in rows:
    raw = np.array([row[i] for i in feat_indices], dtype=np.float32)
    true_label = row[label_col]

    # Apply train.py normalization
    raw[:42]   = normalize_hand(raw[:42])
    raw[42:84] = normalize_hand(raw[42:84])
    raw[84:]  /= 1280.0

    norm = (raw - mean) / (std + 1e-8)

    x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
        pred = label_map[int(np.argmax(probs))]

    mark = 'OK' if pred == true_label else 'WRONG'
    print(f"  True: {true_label:<15} Pred: {pred:<15} {mark}")
    if pred == true_label: correct += 1

print(f"\nAccuracy on 20 DB samples: {correct}/20")
