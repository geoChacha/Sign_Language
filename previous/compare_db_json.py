import sqlite3, json, numpy as np
from pathlib import Path

conn = sqlite3.connect('main_dataset.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM poseDataset LIMIT 1")
row = cursor.fetchone()
cursor.execute("PRAGMA table_info(poseDataset)")
cols = [c[1] for c in cursor.fetchall()]
conn.close()

feat_cols = [c for c in cols if c not in ('id','label')]
db_feats = np.array([row[cols.index(c)] for c in feat_cols], dtype=np.float32)
label = row[cols.index('label')]
print(f"DB row label: {label}")
print(f"DB Rx1,Ry1 (right hand wrist): {db_feats[0]:.3f}, {db_feats[1]:.3f}")
print(f"DB Rx2,Ry2: {db_feats[2]:.3f}, {db_feats[3]:.3f}")
print(f"DB Rx3,Ry3: {db_feats[4]:.3f}, {db_feats[5]:.3f}")
print(f"DB full range: [{db_feats.min():.1f}, {db_feats.max():.1f}]")
print(f"DB non-zero: {np.sum(db_feats!=0)}/110")

# Get corresponding JSON folder
words_dir = Path('PSL_dataset/datasets/words_dataset')
folders = sorted([f for f in words_dir.iterdir() if f.is_dir()],
                 key=lambda f: len(list(f.rglob('*.json'))), reverse=True)

# The first DB row is from the largest class (شکریہ = 131 files)
folder = folders[0]
json_files = sorted(folder.rglob('*_keypoints.json'))
print(f"\nFirst JSON: {json_files[0]}")

with open(json_files[0]) as f:
    data = json.load(f)
p = data['people'][0]
rhand = np.array(p['hand_right_keypoints_2d'], dtype=np.float32).reshape(21,3)
lhand = np.array(p['hand_left_keypoints_2d'], dtype=np.float32).reshape(21,3)
print(f"JSON right hand wrist (x,y,conf): {rhand[0]}")
print(f"JSON right hand lm2 (x,y,conf): {rhand[1]}")
print(f"JSON right hand x range: [{rhand[:,0].min():.1f}, {rhand[:,0].max():.1f}]")
print(f"JSON left hand all zero: {np.all(lhand[:,0]==0) and np.all(lhand[:,1]==0)}")

# What if DB stores raw x,y without normalization?
print(f"\nDirect x,y comparison:")
print(f"  DB Rx1={db_feats[0]:.3f}  JSON rhand[0].x={rhand[0,0]:.3f}")
print(f"  DB Ry1={db_feats[1]:.3f}  JSON rhand[0].y={rhand[0,1]:.3f}")
print(f"  DB Rx2={db_feats[2]:.3f}  JSON rhand[1].x={rhand[1,0]:.3f}")
print(f"  DB Ry2={db_feats[3]:.3f}  JSON rhand[1].y={rhand[1,1]:.3f}")
