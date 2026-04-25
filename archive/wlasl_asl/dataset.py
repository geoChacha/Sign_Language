"""
dataset.py - PyTorch Dataset for WLASL-100 keypoint sequences.

Reads pre-extracted .npy files from KEYPOINTS_DIR.
Applies augmentation during training.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def temporal_jitter(seq: np.ndarray, n: int) -> np.ndarray:
    """Re-sample n frames with a small random offset (temporal jitter)."""
    T = seq.shape[0]
    offset = random.randint(0, max(T - n, 0))
    sub = seq[offset: offset + T]
    indices = np.linspace(0, len(sub) - 1, n).astype(int)
    return sub[indices]


def speed_perturb(seq: np.ndarray, n: int, lo: float = 0.6, hi: float = 1.4) -> np.ndarray:
    """Randomly stretch or compress the temporal axis.
    Wide range [0.6, 1.4] forces the model to be speed-invariant,
    which is the biggest source of variation between different signers.
    """
    T = seq.shape[0]
    factor = random.uniform(lo, hi)
    new_len = max(int(T * factor), n)
    indices = np.linspace(0, T - 1, new_len).astype(int)
    stretched = seq[indices]
    sample_idx = np.linspace(0, len(stretched) - 1, n).astype(int)
    return stretched[sample_idx]


def spatial_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise to all keypoints."""
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    return seq + noise


def keypoint_dropout(seq: np.ndarray, p: float = 0.2) -> np.ndarray:
    """
    Randomly zero out an entire hand (left or right) for each frame
    independently with probability p. Simulates missed MediaPipe detections
    on real-world phone camera input where hands may be partially occluded.
    """
    seq = seq.copy()
    T = seq.shape[0]
    for t in range(T):
        if random.random() < p:
            seq[t, :63] = 0.0      # drop left hand
        if random.random() < p:
            seq[t, 63:126] = 0.0   # drop right hand
    return seq


def horizontal_mirror(seq: np.ndarray) -> np.ndarray:
    """
    Mirror sign along x-axis and swap left/right hands.
    seq: (T, 225) where [:, 0:63]=LH, [:, 63:126]=RH, [:, 126:225]=pose
    """
    seq = seq.copy()
    # Flip x coordinate (index 0, 3, 6, ... in each 63-dim block)
    lh   = seq[:, :63].copy()
    rh   = seq[:, 63:126].copy()
    pose = seq[:, 126:].copy()

    # Flip x (every 3rd element starting at 0)
    for block in [lh, rh, pose]:
        block[:, 0::3] = 1.0 - block[:, 0::3]

    # Swap L/R hands
    seq[:, :63]    = rh
    seq[:, 63:126] = lh
    seq[:, 126:]   = pose
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    def __init__(self, split: str = "train", augment: bool = True):
        """
        split:   'train' | 'val' | 'test'
        augment: apply augmentation (only for training)
        """
        assert split in ("train", "val", "test")
        self.split   = split
        self.augment = augment and (split == "train")
        self.n_frames = cfg.NUM_FRAMES

        # Load nslt_100 which maps video_id -> {"subset": "train|val|test", "action": [class_idx, ?, total]}
        with open(cfg.JSON_100, "r") as f:
            nslt = json.load(f)

        # Build word-to-index mapping from WLASL_v0.3 for human-readable labels
        with open(cfg.JSON_WLASL, "r") as f:
            wlasl = json.load(f)

        # Collect all gloss names sorted by class index
        # nslt format: {vid_id: {"subset": "train|val|test", "action": [class_idx, ?, total]}}
        # We derive class list from the indices
        class_set = {}
        for entry in wlasl:
            gloss = entry["gloss"]
            for inst in entry.get("instances", []):
                vid_id = str(inst.get("video_id", ""))
                if vid_id in nslt:
                    idx = nslt[vid_id]["action"][0]
                    class_set[idx] = gloss
        self.idx_to_gloss = {k: v for k, v in sorted(class_set.items())}
        self.gloss_to_idx = {v: k for k, v in self.idx_to_gloss.items()}

        # Load missing list
        missing = set()
        if os.path.exists(cfg.MISSING_TXT):
            with open(cfg.MISSING_TXT) as f:
                for line in f:
                    vid = line.strip().split("/")[-1].replace(".mp4", "")
                    if vid:
                        missing.add(vid)

        # Build samples: [(npy_path, class_idx), ...]
        self.samples = []
        split_map = {"train": "train", "val": "val", "test": "test"}
        target_split = split_map[split]

        for vid_id, info in nslt.items():
            class_idx = info["action"][0]
            vid_split = info["subset"]
            if vid_split != target_split:
                continue
            if vid_id in missing:
                continue
            npy_path = os.path.join(cfg.KEYPOINTS_DIR, f"{vid_id}.npy")
            if not os.path.exists(npy_path):
                continue
            self.samples.append((npy_path, class_idx))

        print(f"[Dataset] split={split}  samples={len(self.samples)}  classes={len(self.idx_to_gloss)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, class_idx = self.samples[idx]
        seq = np.load(npy_path).astype(np.float32)   # (T, 225)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            if cfg.AUG_SPEED_PERTURB:
                seq = speed_perturb(seq, self.n_frames)
            elif cfg.AUG_TEMPORAL_JITTER:
                seq = temporal_jitter(seq, self.n_frames)

            if cfg.AUG_SPATIAL_NOISE > 0:
                seq = spatial_noise(seq, cfg.AUG_SPATIAL_NOISE)

            if random.random() < cfg.AUG_MIRROR:
                seq = horizontal_mirror(seq)

            kp_drop = getattr(cfg, "AUG_KEYPOINT_DROPOUT", 0.0)
            if kp_drop > 0:
                seq = keypoint_dropout(seq, kp_drop)
        else:
            # Just uniform re-sample to fixed length
            T = seq.shape[0]
            indices = np.linspace(0, T - 1, self.n_frames).astype(int)
            seq = seq[indices]

        # Ensure exactly n_frames
        if seq.shape[0] != self.n_frames:
            indices = np.linspace(0, seq.shape[0] - 1, self.n_frames).astype(int)
            seq = seq[indices]

        # Normalize: zero-fill frames that were missing (all zeros) stay as-is
        # Per-channel standardisation is NOT applied here because zero
        # padding needs to remain distinguishable. Instead, we let BatchNorm
        # in the TCN handle normalisation.

        # Use hands-only features - drop pose ([:, 126:]) for framing invariance
        seq = seq[:, :126]                 # (T, 126) - L-hand + R-hand only
        x = torch.from_numpy(seq)
        y = torch.tensor(class_idx, dtype=torch.long)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_loaders():
    train_ds = WLASLDataset("train", augment=True)
    val_ds   = WLASLDataset("val",   augment=False)
    test_ds  = WLASLDataset("test",  augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds.idx_to_gloss


def get_trainval_loader():
    """
    Combine train + val into a single training set.
    Use this for the final training run once you know the model
    converges around epoch N - train for that fixed number of epochs,
    no early stopping needed.
    Val samples get augmentation too since they are now training data.
    """
    train_ds = WLASLDataset("train", augment=True)
    val_ds   = WLASLDataset("val",   augment=True)   # augment val too now
    test_ds  = WLASLDataset("test",  augment=False)

    combined = torch.utils.data.ConcatDataset([train_ds, val_ds])
    print(f"[Dataset] train+val combined: {len(combined)} samples")

    trainval_loader = DataLoader(
        combined,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return trainval_loader, test_loader, train_ds.idx_to_gloss


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_loaders()
    x, y = next(iter(train_loader))
    print(f"x: {x.shape}  y: {y.shape}")
    print(f"Vocab sample: {list(vocab.items())[:5]}")