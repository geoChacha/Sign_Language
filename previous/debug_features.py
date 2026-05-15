"""Debug: compare normalized training features vs video features."""
import sqlite3, json, numpy as np, cv2, mediapipe as mp

with open('label_map.json', encoding='utf-8') as f:
    meta = json.load(f)
mean = np.array(meta['mean'])
std  = np.array(meta['std'])

def normalize_hand(hand_arr):
    if not np.any(hand_arr != 0): return hand_arr
    pts = hand_arr.reshape(21, 2).copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale > 1e-6: pts /= scale
    return pts.flatten()

# Training sample
conn = sqlite3.connect('main_dataset.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM poseDataset WHERE label='جانتا' LIMIT 1")
row = cursor.fetchone()
cursor.execute("PRAGMA table_info(poseDataset)")
cols = [c[1] for c in cursor.fetchall()]
conn.close()

feat_cols = [c for c in cols if c not in ('id','label')]
raw = np.array([row[cols.index(c)] for c in feat_cols], dtype=np.float32)
raw[:42] = normalize_hand(raw[:42])
raw[42:84] = normalize_hand(raw[42:84])
raw[84:] /= 1280.0
norm = (raw - mean) / (std + 1e-8)
print(f"Training جانتا normalized: range=[{norm.min():.3f},{norm.max():.3f}] mean={norm.mean():.4f}")
print(f"  rhand wrist: {raw[:2]}")
print(f"  rhand[2:6]: {raw[2:6]}")

# Video frame
mp_pose_obj = mp.solutions.pose.Pose(static_image_mode=False)
mp_hands_obj = mp.solutions.hands.Hands(max_num_hands=2)
cap = cv2.VideoCapture("test_vid/know_psl.mp4")
for _ in range(30): cap.read()
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pose_res = mp_pose_obj.process(rgb)
hand_res = mp_hands_obj.process(rgb)

rhand = np.zeros(42, dtype=np.float32)
lhand = np.zeros(42, dtype=np.float32)
if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
    for hl, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
        lbl = hd.classification[0].label
        coords = np.array([c for lm in hl.landmark for c in [lm.x*w, lm.y*h]], dtype=np.float32)
        print(f"  MediaPipe hand '{lbl}': wrist=({coords[0]:.1f},{coords[1]:.1f})")
        if lbl == "Left": rhand = coords
        else: lhand = coords

rhand = normalize_hand(rhand)
lhand = normalize_hand(lhand)
pose_kp = np.zeros(26, dtype=np.float32)
if pose_res.pose_landmarks:
    for i, idx in enumerate([11,12,13,14,15,16,23,24,0,1,2,3,4]):
        lm = pose_res.pose_landmarks.landmark[idx]
        pose_kp[i*2] = lm.x*w; pose_kp[i*2+1] = lm.y*h
    pose_kp /= 1280.0

video_feats = np.concatenate([rhand, lhand, pose_kp])
video_norm = (video_feats - mean) / (std + 1e-8)
print(f"\nVideo normalized: range=[{video_norm.min():.3f},{video_norm.max():.3f}] mean={video_norm.mean():.4f}")
print(f"  rhand wrist: {rhand[:2]}")
print(f"  rhand[2:6]: {rhand[2:6]}")
print(f"  Non-zero in rhand: {np.sum(rhand!=0)}/42")
print(f"  Non-zero in lhand: {np.sum(lhand!=0)}/42")

mp_pose_obj.close()
mp_hands_obj.close()
