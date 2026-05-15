"""
Find the correct mapping between OpenPose JSON files and DB rows
by matching keypoint values directly.
"""
import sqlite3, json, numpy as np
from pathlib import Path

conn = sqlite3.connect('main_dataset.db')
cursor = conn.cursor()
cursor.execute("SELECT id, Rx1, Ry1, Rx2, Ry2, label FROM poseDataset LIMIT 5")
rows = cursor.fetchall()
conn.close()

print("First 5 DB rows (id, Rx1, Ry1, Rx2, Ry2, label):")
for r in rows:
    print(f"  id={r[0]}  Rx1={r[1]:.2f}  Ry1={r[2]:.2f}  Rx2={r[3]:.2f}  Ry2={r[4]:.2f}  label={r[5]}")

# Search for matching JSON
words_dir = Path('PSL_dataset/datasets/words_dataset')
target_rx1, target_ry1 = rows[0][1], rows[0][2]
print(f"\nSearching for JSON with right hand wrist ≈ ({target_rx1:.1f}, {target_ry1:.1f})...")

for jf in words_dir.rglob('*_keypoints.json'):
    with open(jf) as f:
        data = json.load(f)
    people = data.get('people', [])
    if not people: continue
    p = people[0]
    rhand = p.get('hand_right_keypoints_2d', [])
    if not rhand: continue
    rx1, ry1 = rhand[0], rhand[1]
    if abs(rx1 - target_rx1) < 1.0 and abs(ry1 - target_ry1) < 1.0:
        print(f"  MATCH: {jf}")
        print(f"  JSON Rx1={rx1:.2f}  Ry1={ry1:.2f}")
        break
