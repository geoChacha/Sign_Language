"""
psl_features.py — Exact feature extraction matching the original PSL repo pipeline.

Implements the same normalization as:
  PSL/helper/normalize.py  (scaleBody, moveBody, move_to_wrist)
  PSL/helper/scale.py      (scalePoints)
  PSL/helper/helperFunc.py (removePoints)

This ensures MediaPipe-extracted features match the training data distribution.

OpenPose landmark indices used:
  Pose:  0=nose, 1=neck, 2=RShoulder, 5=LShoulder, 8=RWrist, 14=LWrist
  Hand:  0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
         landmark 18 = pinky MCP (used as scale reference in original code)

MediaPipe hand landmarks are identical to OpenPose hand landmarks (same indices).
MediaPipe pose landmarks differ — mapping provided below.
"""

import math
import numpy as np


# =========================
# ORIGINAL REPO HELPERS
# =========================

def remove_confidence(keypoints_3n: list) -> list:
    """Strip confidence values from [x,y,c, x,y,c, ...] → [x,y, x,y, ...]."""
    result = []
    for i in range(0, len(keypoints_3n), 3):
        result.append(keypoints_3n[i])      # x
        result.append(keypoints_3n[i + 1])  # y
    return result


def scale_hand(hand_xy: list, ref: float = 50.0) -> list:
    """Scale hand so distance(wrist, landmark18) = ref, then ×2.
    Matches scale.scalePoints(hand, distance) with ref=50, then ×2.
    """
    xs = hand_xy[0::2]
    ys = hand_xy[1::2]

    # distance between wrist (0) and pinky MCP (18)
    p1 = [xs[0], ys[0]]
    p2 = [xs[18], ys[18]]
    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    if distance == 0:
        return hand_xy

    scale = ref / distance

    xs = [x * scale * 2 for x in xs]
    ys = [y * scale * 2 for y in ys]

    result = []
    for x, y in zip(xs, ys):
        result.append(x)
        result.append(y)
    return result


def scale_body(pose_xy: list, ref: float = 200.0) -> list:
    """Scale body so distance(landmark0, landmark1) = ref.
    Matches normalize.scaleBody(pose, distance) with ref=200.
    """
    xs = pose_xy[0::2]
    ys = pose_xy[1::2]

    p1 = [xs[0], ys[0]]
    p2 = [xs[1], ys[1]]
    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    if distance == 0:
        return pose_xy

    scale = ref / distance
    xs = [x * scale for x in xs]
    ys = [y * scale for y in ys]

    result = []
    for x, y in zip(xs, ys):
        result.append(x)
        result.append(y)
    return result


def move_body(pose_xy: list, ref_x: float = 1000.0, ref_y: float = 400.0) -> list:
    """Translate body so landmark1 (neck) is at (ref_x, ref_y).
    Matches normalize.moveBody with refX=1000, refY=400.
    """
    xs = pose_xy[0::2]
    ys = pose_xy[1::2]

    dx = xs[1] - ref_x
    dy = ys[1] - ref_y

    xs = [x - dx if x != 0 else 0 for x in xs]
    ys = [y - dy if y != 0 else 0 for y in ys]

    result = []
    for x, y in zip(xs, ys):
        result.append(x)
        result.append(y)
    return result


def move_to_wrist(hand_xy: list, wrist_x: float, wrist_y: float) -> list:
    """Translate hand so its wrist aligns to pose wrist position.
    Matches normalize.move_to_wrist(hand, wristX, wristY).
    """
    xs = hand_xy[0::2]
    ys = hand_xy[1::2]

    dx = xs[0] - wrist_x
    dy = ys[0] - wrist_y

    xs = [x - dx for x in xs]
    ys = [y - dy for y in ys]

    result = []
    for x, y in zip(xs, ys):
        result.append(x)
        result.append(y)
    return result


# =========================
# OPENPOSE JSON → FEATURES
# =========================

def extract_from_openpose_json(data: dict) -> np.ndarray | None:
    """Extract features from OpenPose JSON dict using original repo pipeline.

    Returns 110-dim feature vector matching poseDataset columns:
      [rhand(42), lhand(42), pose_subset(26)]
    """
    people = data.get('people', [])
    if not people:
        return None

    p = people[0]
    pose_raw   = p.get('pose_keypoints_2d', [])
    rhand_raw  = p.get('hand_right_keypoints_2d', [])
    lhand_raw  = p.get('hand_left_keypoints_2d', [])

    if not pose_raw or not rhand_raw:
        return None

    # Confidence check (matches original: RightConfidence > 12)
    rconf = sum(rhand_raw[2::3])
    lconf = sum(lhand_raw[2::3]) if lhand_raw else 0

    if rconf <= 12:
        return None

    # Strip confidence values
    pose_xy  = remove_confidence(pose_raw)
    rhand_xy = remove_confidence(rhand_raw)
    lhand_xy = remove_confidence(lhand_raw) if lhand_raw else [0.0] * 42

    # Normalize body
    p1 = [pose_xy[0], pose_xy[1]]
    p2 = [pose_xy[2], pose_xy[3]]
    body_dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    pose_scaled = scale_body(pose_xy, ref=200.0)
    pose_moved  = move_body(pose_scaled, ref_x=1000.0, ref_y=400.0)

    # Normalize right hand
    p1 = [rhand_xy[0], rhand_xy[1]]
    p2 = [rhand_xy[36], rhand_xy[37]]  # landmark 18 = index 36,37
    rhand_dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    rhand_scaled = scale_hand(rhand_xy, ref=50.0)
    # pose_moved[8], pose_moved[9] = right wrist x,y (OpenPose pose landmark 4 = RWrist)
    # In the original: poseResults[8], poseResults[9]
    rhand_final = move_to_wrist(rhand_scaled, pose_moved[8], pose_moved[9])

    # Normalize left hand
    if lconf > 3 and lhand_raw:
        p1 = [lhand_xy[0], lhand_xy[1]]
        p2 = [lhand_xy[36], lhand_xy[37]]
        lhand_dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        if lhand_dist != 0:
            lhand_scaled = scale_hand(lhand_xy, ref=50.0)
        else:
            lhand_scaled = lhand_xy
        # pose_moved[14], pose_moved[15] = left wrist (OpenPose landmark 7 = LWrist)
        lhand_final = move_to_wrist(lhand_scaled, pose_moved[14], pose_moved[15])
    else:
        lhand_final = [0.0] * 42

    # Pose subset: landmarks 0-17 (36 values) + landmarks 15-18 (8 values) = 44 values
    # Original: posePoints 0..17 + 30..37 → indices 0-35 + 30-37 in pose_moved
    pose_subset = []
    for x in range(18):
        pose_subset.append(pose_moved[x])
    for x in range(30, 38):
        pose_subset.append(pose_moved[x])

    return np.array(rhand_final + lhand_final + pose_subset, dtype=np.float32)


# =========================
# MEDIAPIPE → OPENPOSE FORMAT → FEATURES
# =========================

# MediaPipe pose landmark indices → OpenPose pose landmark indices mapping
# OpenPose 25-keypoint model:
#   0=nose, 1=neck, 2=RShoulder, 3=RElbow, 4=RWrist,
#   5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip,
#   9=RHip, 10=RKnee, 11=RAnkle, 12=LHip, 13=LKnee, 14=LAnkle,
#   15=REye, 16=LEye, 17=REar, 18=LEar, 19-24=feet
#
# MediaPipe pose (33 landmarks):
#   0=nose, 11=LShoulder, 12=RShoulder, 13=LElbow, 14=RElbow,
#   15=LWrist, 16=RWrist, 23=LHip, 24=RHip
#
# Mapping: OpenPose_idx → MediaPipe_idx
MP_TO_OP_POSE = {
    0:  0,   # nose
    1:  None,  # neck = midpoint of shoulders (11,12)
    2:  12,  # RShoulder
    3:  14,  # RElbow
    4:  16,  # RWrist
    5:  11,  # LShoulder
    6:  13,  # LElbow
    7:  15,  # LWrist
    8:  None,  # MidHip = midpoint of hips (23,24)
    9:  24,  # RHip
    10: 26,  # RKnee
    11: 28,  # RAnkle
    12: 23,  # LHip
    13: 25,  # LKnee
    14: 27,  # LAnkle
    15: 5,   # REye
    16: 2,   # LEye
    17: 8,   # REar
    18: 7,   # LEar
}


def mediapipe_to_openpose_pose(pose_landmarks, frame_w: int, frame_h: int) -> list:
    """Convert MediaPipe pose landmarks to OpenPose 25-keypoint flat x,y,c list."""
    lms = pose_landmarks.landmark
    result = []

    for op_idx in range(19):  # OpenPose indices 0-18
        mp_idx = MP_TO_OP_POSE.get(op_idx)

        if mp_idx is None:
            if op_idx == 1:  # neck = midpoint of shoulders
                x = (lms[11].x + lms[12].x) / 2 * frame_w
                y = (lms[11].y + lms[12].y) / 2 * frame_h
                c = (lms[11].visibility + lms[12].visibility) / 2
            elif op_idx == 8:  # mid hip
                x = (lms[23].x + lms[24].x) / 2 * frame_w
                y = (lms[23].y + lms[24].y) / 2 * frame_h
                c = (lms[23].visibility + lms[24].visibility) / 2
            else:
                x, y, c = 0.0, 0.0, 0.0
        else:
            lm = lms[mp_idx]
            x = lm.x * frame_w
            y = lm.y * frame_h
            c = lm.visibility

        result.extend([x, y, c])

    # Pad to 25 landmarks (add zeros for feet landmarks 19-24)
    result.extend([0.0] * 6 * 3)
    return result


def mediapipe_hand_to_openpose(hand_landmarks, frame_w: int, frame_h: int) -> list:
    """Convert MediaPipe hand landmarks to OpenPose hand format [x,y,c × 21].

    MediaPipe and OpenPose use identical hand landmark indices (0-20).
    """
    result = []
    for lm in hand_landmarks.landmark:
        result.extend([lm.x * frame_w, lm.y * frame_h, lm.visibility])
    return result


def extract_from_mediapipe(pose_res, hand_res, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Extract 110-dim features from MediaPipe results using original repo pipeline.

    Converts MediaPipe landmarks to OpenPose pixel coordinates, then applies
    the exact same normalization as the original PSL repo (scale + move_to_wrist).
    Bypasses the OpenPose confidence check since MediaPipe uses visibility (0-1).
    """
    if not pose_res.pose_landmarks:
        return None

    # Need at least one hand
    if not hand_res.multi_hand_landmarks:
        return None

    # Convert pose to OpenPose format (x,y,c × 25)
    pose_raw = mediapipe_to_openpose_pose(pose_res.pose_landmarks, frame_w, frame_h)

    # Convert hands
    rhand_raw = [0.0] * 63
    lhand_raw = [0.0] * 63

    for hand_lm, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
        mp_label = handedness.classification[0].label
        op_hand  = mediapipe_hand_to_openpose(hand_lm, frame_w, frame_h)
        # MediaPipe "Left" from camera POV = signer's right hand
        if mp_label == "Left":
            rhand_raw = op_hand
        else:
            lhand_raw = op_hand

    # Check right hand is actually detected (not all zeros)
    rhand_xy = remove_confidence(rhand_raw)
    if not any(v != 0 for v in rhand_xy):
        return None

    # Strip confidence from all keypoints
    pose_xy  = remove_confidence(pose_raw)
    lhand_xy = remove_confidence(lhand_raw)
    lconf    = sum(lhand_raw[2::3])  # sum of visibility scores

    # ── Apply exact original repo normalization ──

    # Normalize body
    pose_scaled = scale_body(pose_xy, ref=200.0)
    pose_moved  = move_body(pose_scaled, ref_x=1000.0, ref_y=400.0)

    # Normalize right hand
    rhand_scaled = scale_hand(rhand_xy, ref=50.0)
    rhand_final  = move_to_wrist(rhand_scaled, pose_moved[8], pose_moved[9])

    # Normalize left hand
    if lconf > 0.5:  # MediaPipe visibility threshold
        lhand_scaled = scale_hand(lhand_xy, ref=50.0)
        lhand_final  = move_to_wrist(lhand_scaled, pose_moved[14], pose_moved[15])
    else:
        lhand_final = [0.0] * 42

    # Pose subset (matches original: indices 0-17 + 30-37)
    pose_subset = []
    for x in range(18):
        pose_subset.append(pose_moved[x])
    for x in range(30, 38):
        pose_subset.append(pose_moved[x])

    return np.array(rhand_final + lhand_final + pose_subset, dtype=np.float32)
