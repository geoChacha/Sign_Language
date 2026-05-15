"""Compare OpenPose JSON hand shape vs MediaPipe hand shape for same sign."""
import json, numpy as np, cv2, mediapipe as mp
from pathlib import Path

def normalize_hand(pts_xy):
    pts = pts_xy.copy()
    if not np.any(pts != 0): return pts.flatten()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale > 1e-6: pts /= scale
    return pts.flatten()

# Load a جانتا (know) OpenPose JSON sample
words_dir = Path('PSL_dataset/datasets/words_dataset')
folders = sorted([f for f in words_dir.iterdir() if f.is_dir()],
                 key=lambda f: len(list(f.rglob('*.json'))), reverse=True)
# جانتا = 21 files = index 9
know_folder = folders[9]
know_files = sorted(know_folder.rglob('*_keypoints.json'))
print(f"جانتا folder: {know_folder.name} ({len(know_files)} files)")

with open(know_files[0]) as f:
    data = json.load(f)
p = data['people'][0]
rhand = np.array(p['hand_right_keypoints_2d'], dtype=np.float32).reshape(21,3)[:,:2]
lhand = np.array(p['hand_left_keypoints_2d'], dtype=np.float32).reshape(21,3)[:,:2]
print(f"OpenPose right hand detected: {np.any(rhand!=0)}")
print(f"OpenPose left hand detected: {np.any(lhand!=0)}")
rn = normalize_hand(rhand)
ln = normalize_hand(lhand)
print(f"OpenPose rhand norm[0:6]: {rn[:6].round(3)}")
print(f"OpenPose lhand norm[0:6]: {ln[:6].round(3)}")

# Now extract from know_psl.mp4 using MediaPipe
print("\n--- MediaPipe from know_psl.mp4 ---")
mp_pose_obj = mp.solutions.pose.Pose(static_image_mode=False)
mp_hands_obj = mp.solutions.hands.Hands(max_num_hands=2)
cap = cv2.VideoCapture("test_vid/know_psl.mp4")
for _ in range(20): cap.read()
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
hand_res = mp_hands_obj.process(rgb)

if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
    for hl, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
        lbl = hd.classification[0].label
        pts = np.array([[lm.x*w, lm.y*h] for lm in hl.landmark], dtype=np.float32)
        norm = normalize_hand(pts)
        print(f"MediaPipe '{lbl}' hand norm[0:6]: {norm[:6].round(3)}")
        print(f"  (wrist at origin, scale normalized)")
else:
    print("No hands detected")

mp_pose_obj.close()
mp_hands_obj.close()
