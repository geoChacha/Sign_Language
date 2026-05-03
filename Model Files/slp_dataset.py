"""
7_slp_dataset.py - Dataset for Sign Language Production (SLP).

Loads keypoint .npy files and pairs each with its gloss label.
For production we use the FULL 225-dim keypoints (hands + pose)
because the renderer needs pose for body/arm positioning.

Also builds a gloss vocabulary for the seq2seq model.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import config as cfg


class SLPDataset(Dataset):
    """
    Each sample: (gloss_idx, keypoint_sequence)
    gloss_idx    : int  - index into gloss vocab
    keypoint_seq : (T, 225) float32 - full body keypoints (NOT sliced)
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")

        with open(cfg.JSON_100) as f:
            nslt = json.load(f)
        with open(cfg.JSON_WLASL) as f:
            wlasl = json.load(f)

        # Build class_idx -> gloss
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

        # Load missing
        missing = set()
        if os.path.exists(cfg.MISSING_TXT):
            with open(cfg.MISSING_TXT) as f:
                for line in f:
                    vid = line.strip().split("/")[-1].replace(".mp4", "")
                    if vid:
                        missing.add(vid)

        # Build samples: (npy_path, class_idx)
        self.samples = []
        for vid_id, info in nslt.items():
            if info["subset"] != split:
                continue
            if vid_id in missing:
                continue
            npy_path = os.path.join(cfg.SLP_DIR, f"{vid_id}.npy")
            if not os.path.exists(npy_path):
                continue
            self.samples.append((npy_path, info["action"][0]))

        # Save vocab to disk for renderer / inference
        vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "slp_vocab.json")
        os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump({"idx_to_gloss": {str(k): v for k, v in self.idx_to_gloss.items()},
                       "gloss_to_idx": self.gloss_to_idx}, f, indent=2)

        print(f"[SLPDataset] split={split}  samples={len(self.samples)}  "
              f"glosses={len(self.idx_to_gloss)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, class_idx = self.samples[idx]
        seq = np.load(npy_path).astype(np.float32)   # (T, 225) full keypoints

        # Normalise length to NUM_FRAMES
        T = seq.shape[0]
        indices = np.linspace(0, T - 1, cfg.NUM_FRAMES).astype(int)
        seq = seq[indices]                             # (NUM_FRAMES, 225)

        return (
            torch.tensor(class_idx, dtype=torch.long),
            torch.from_numpy(seq),                     # (T, 225)
        )


def slp_collate(batch):
    labels, seqs = zip(*batch)
    return (
        torch.stack(labels),
        torch.stack(seqs),   # (B, T, 225) - all same length after resampling
    )


def get_slp_loaders():
    train_ds = SLPDataset("train")
    val_ds   = SLPDataset("val")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, collate_fn=slp_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, collate_fn=slp_collate)

    return train_loader, val_loader, train_ds.idx_to_gloss, train_ds.gloss_to_idx


if __name__ == "__main__":
    train_loader, val_loader, vocab, g2i = get_slp_loaders()
    labels, seqs = next(iter(train_loader))
    print(f"labels: {labels.shape}  seqs: {seqs.shape}")
    print(f"Vocab sample: {list(vocab.items())[:5]}")