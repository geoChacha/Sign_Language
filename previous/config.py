# =========================
# config.py — PSL Recognition System
# =========================

# Database
DB_PATH = "main_dataset.db"

# Feature dimensions
# poseDataset: right hand (42) + left hand (42) + pose (26) = 110 dims
POSE_INPUT_DIM  = 110   # full pose table features
WORD_INPUT_DIM  = 42    # wordDataset: right hand only (21 landmarks × 2)

# Model
MODEL_DIM  = 128
EMBED_DIM  = 64
NUM_HEADS  = 4
DROPOUT    = 0.3

# Inference
CONFIDENCE_THRESHOLD = 0.60   # min similarity to predict (not Unknown)

# Paths
MODEL_PATH = "psl_model.pt"
LABEL_MAP_PATH = "label_map.json"
