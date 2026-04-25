"""
utils.py - Miscellaneous helpers for the WLASL pipeline.
"""

import os
import json
import numpy as np


def plot_training_history(history_path: str, out_path: str = None):
    """Plot training/val loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to plot training history")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [r["epoch"] for r in history]
    tr_loss    = [r["train_loss"] for r in history]
    va_loss    = [r["val_loss"] for r in history]
    tr_top1    = [r["train_top1"] for r in history]
    va_top1    = [r["val_top1"] for r in history]
    tr_top5    = [r["train_top5"] for r in history]
    va_top5    = [r["val_top5"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, tr_loss, label="Train", color="royalblue")
    axes[0].plot(epochs, va_loss, label="Val",   color="tomato")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, tr_top1, label="Train Top-1", color="royalblue")
    axes[1].plot(epochs, va_top1, label="Val Top-1",   color="tomato")
    axes[1].plot(epochs, tr_top5, label="Train Top-5", color="royalblue",  linestyle="--", alpha=0.6)
    axes[1].plot(epochs, va_top5, label="Val Top-5",   color="tomato",     linestyle="--", alpha=0.6)
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_path or os.path.join(os.path.dirname(history_path), "training_curves.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Training curves saved to {out}")


def dataset_stats(keypoints_dir: str, nslt_json: str):
    """Print per-class sample counts and keypoint fill-rate."""
    with open(nslt_json) as f:
        nslt = json.load(f)

    class_counts = {}
    for vid_id, info in nslt.items():
        cls   = info["action"][0]
        split = info["subset"]
        npy = os.path.join(keypoints_dir, f"{vid_id}.npy")
        if os.path.exists(npy):
            class_counts.setdefault(cls, {"train": 0, "val": 0, "test": 0})
            class_counts[cls][split] = class_counts[cls].get(split, 0) + 1

    print(f"\nDataset statistics ({len(class_counts)} classes):")
    print(f"{'Class':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}")
    print("-" * 38)
    totals = {"train": 0, "val": 0, "test": 0}
    for cls in sorted(class_counts.keys()):
        c = class_counts[cls]
        t = sum(c.values())
        print(f"{cls:>6}  {c.get('train',0):>6}  {c.get('val',0):>5}  {c.get('test',0):>5}  {t:>6}")
        for k in totals:
            totals[k] += c.get(k, 0)
    print("-" * 38)
    print(f"{'TOTAL':>6}  {totals['train']:>6}  {totals['val']:>5}  {totals['test']:>5}  {sum(totals.values()):>6}")


def check_keypoint_quality(keypoints_dir: str, sample: int = 100):
    """
    Sample random .npy files and report fraction of frames with non-zero
    left-hand, right-hand, and pose detections.
    """
    files = [f for f in os.listdir(keypoints_dir) if f.endswith(".npy")]
    if not files:
        print("No .npy files found")
        return
    sampled = np.random.choice(files, min(sample, len(files)), replace=False)

    lh_fill, rh_fill, pose_fill = [], [], []
    for fname in sampled:
        seq = np.load(os.path.join(keypoints_dir, fname))
        lh_fill.append( (seq[:, :63].sum(axis=1) != 0).mean() )
        rh_fill.append( (seq[:, 63:126].sum(axis=1) != 0).mean() )
        pose_fill.append( (seq[:, 126:].sum(axis=1) != 0).mean() )

    print(f"\nKeypoint detection rate (sampled {len(sampled)} videos):")
    print(f"  Left hand:  {np.mean(lh_fill)*100:.1f}%")
    print(f"  Right hand: {np.mean(rh_fill)*100:.1f}%")
    print(f"  Pose:       {np.mean(pose_fill)*100:.1f}%")


if __name__ == "__main__":
    import config as cfg

    # Plot training curves if history exists
    hist_path = os.path.join(cfg.CHECKPOINTS_DIR, "history.json")
    if os.path.exists(hist_path):
        plot_training_history(hist_path)
    else:
        print(f"No history.json found at {hist_path}")

    # Dataset stats
    if os.path.exists(cfg.KEYPOINTS_DIR) and os.path.exists(cfg.JSON_100):
        dataset_stats(cfg.KEYPOINTS_DIR, cfg.JSON_100)
        check_keypoint_quality(cfg.KEYPOINTS_DIR)