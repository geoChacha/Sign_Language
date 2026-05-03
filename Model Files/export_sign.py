"""
11_export_sign_poses.py - Export sign pose sequences to JSON for the web player.

Run AFTER training (9_slp_train.py):
    python 11_export_sign_poses.py

Outputs:
    checkpoints/sign_poses.json   — pose sequences for every word in vocab
    checkpoints/slp_vocab.json    — already created by training

The web player (sign_avatar_player.html) loads both files.

For Flutter/React Native:
    Copy sign_avatar_player.html, sign_poses.json, slp_vocab.json
    and your avatar.vrm into your app's assets folder.
    Load the HTML in a WebView.
"""

import os
import json
import torch
import numpy as np
import config as cfg
from slp_model import SignProductionModel
from slp_dataset import SLPDataset


def load_model(device):
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "slp_best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No trained model at {ckpt_path}")
        print("        Run python 9_slp_train.py first")
        import sys; sys.exit(1)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = SignProductionModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[Export] Loaded model epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model


def normalize_poses(frames: np.ndarray) -> np.ndarray:
    """
    Normalize per-channel to [0.05, 0.95] for the web renderer.
    Preserves zeros (missing detections).
    """
    result = frames.copy()
    for ch in range(frames.shape[1]):
        col  = frames[:, ch]
        mask = col != 0
        if mask.sum() < 2:
            continue
        lo, hi = col[mask].min(), col[mask].max()
        rng = hi - lo
        if rng < 1e-6:
            continue
        normed         = (col - lo) / rng * 0.90 + 0.05
        result[:, ch]  = np.where(mask, normed, col)
    return result


def smooth_poses(frames: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smooth each channel over time for fluid animation."""
    from scipy.ndimage import gaussian_filter1d
    result = frames.copy()
    for ch in range(frames.shape[1]):
        col  = frames[:, ch]
        mask = col != 0
        if mask.sum() < 3:
            continue
        result[:, ch] = np.where(mask, gaussian_filter1d(col, sigma), col)
    return result


def main():
    device = torch.device(cfg.DEVICE)
    model  = load_model(device)

    # Load vocabulary
    ds = SLPDataset("train")
    vocab = ds.idx_to_gloss   # idx -> gloss

    sign_poses = {}
    total = len(vocab)

    print(f"[Export] Generating poses for {total} words...")

    with torch.no_grad():
        for idx, gloss in vocab.items():
            gloss_tensor = torch.tensor([idx], dtype=torch.long, device=device)
            pose_seq = model.generate(
                gloss_tensor,
                n_frames=cfg.NUM_FRAMES
            )   # (1, T, 225)

            frames = pose_seq[0].cpu().numpy()   # (T, 225)
            frames = normalize_poses(frames)
            frames = smooth_poses(frames, sigma=1.0)

            # Convert to list of lists for JSON
            sign_poses[gloss] = frames.tolist()

            print(f"  [{idx+1:3d}/{total}] {gloss:<20} ({frames.shape[0]} frames)")

    # Save
    out_path = os.path.join(cfg.CHECKPOINTS_DIR, "sign_poses.json")
    with open(out_path, "w") as f:
        json.dump(sign_poses, f)

    size_mb = os.path.getsize(out_path) / (1024*1024)
    print(f"\n[Export] Done  {out_path}  ({size_mb:.1f} MB)")
    print(f"[Export] {len(sign_poses)} words exported")
    print(f"\nNext steps:")
    print(f"  1. Copy to your app assets folder:")
    print(f"       {out_path}")
    print(f"       {os.path.join(cfg.CHECKPOINTS_DIR, 'slp_vocab.json')}")
    print(f"       sign_avatar_player.html")
    print(f"       avatar.vrm  (from VRoid Studio or hub.vroid.com)")
    print(f"  2. Load sign_avatar_player.html in a Flutter WebView or RN WebView")
    print(f"  3. The player loads the JSON files and animates the avatar")


if __name__ == "__main__":
    main()