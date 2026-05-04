"""
config.py - Central configuration for WLASL 100-word ASL pipeline.
Edit the paths below to match your Kaggle dataset location.
"""

import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
# Resolve paths relative to this config file's location so the script works
# regardless of which directory you run it from.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../Model Files/
_WORKSPACE = os.path.dirname(_HERE)                          # workspace root

# FIX #4: removed redundant "./" — os.path.join with "./" can behave unexpectedly
DATASET_ROOT = _WORKSPACE

VIDEOS_DIR      = os.path.join(DATASET_ROOT, "videos")
JSON_WLASL      = os.path.join(DATASET_ROOT, "WLASL_v0.3.json")
JSON_100        = os.path.join(DATASET_ROOT, "nslt_100.json")
MISSING_TXT     = os.path.join(DATASET_ROOT, "missing.txt")
SLP_DIR         = os.path.join(DATASET_ROOT, "keypoints_slp")

# ── Keypoints directory — pre-extracted .npy files at workspace root ────────
KEYPOINTS_DIR   = os.path.join(_WORKSPACE, "keypoints_100")

# ── Checkpoint directory — points to Model/best_model.pth ──────────────────
CHECKPOINTS_DIR = os.path.join(_WORKSPACE, "Model")


# KEYPOINT EXTRACTION
NUM_FRAMES  = 64
# Full feature dim stored in .npy files (do not change - no need to re-extract)
FEATURE_DIM_FULL = 63 + 63 + 99   # = 225
# Features actually used by the model - hands only, pose dropped
# Pose coords depend on camera framing (close-up vs waist-up) making them
# harmful for real-world generalisation. Hands are scale/crop invariant.
FEATURE_DIM = 63 + 63              # = 126  (L-hand + R-hand only)

# MODEL - TCN + BiGRU
NUM_CLASSES = 100
D_MODEL     = 192
NHEAD       = 4
NUM_LAYERS  = 2
DIM_FF      = 256
DROPOUT     = 0.4
MAX_SEQ_LEN = NUM_FRAMES

# TRAINING
BATCH_SIZE    = 32
NUM_EPOCHS    = 150
LR            = 3e-4
WEIGHT_DECAY  = 5e-4
LABEL_SMOOTH  = 0.15
WARMUP_EPOCHS = 10
PATIENCE      = 25

# AUGMENTATION
AUG_TEMPORAL_JITTER  = True
AUG_SPATIAL_NOISE    = 0.015
AUG_TEMPORAL_FLIP    = 0.0   # FIX #5: intentionally disabled — set to >0.0 to enable (e.g. 0.5)
AUG_MIRROR           = 0.5
AUG_SPEED_PERTURB    = True
AUG_KEYPOINT_DROPOUT = 0.1

# DEVICE
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MISC
SEED        = 42
NUM_WORKERS = 4