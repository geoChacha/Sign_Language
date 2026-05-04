"""
3_evaluate.py - Evaluate the trained model on the test split.

Outputs:
- Top-1 / Top-5 accuracy
- Per-class accuracy table
- Confusion matrix (saved as PNG)
- Predictions CSV

Run:
    python 3_evaluate.py
"""

import os
import json
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import config as cfg
from model import SignLanguageTransformer
from dataset import WLASLDataset


def load_model(ckpt_path: str, device: torch.device) -> SignLanguageTransformer:
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
    mc = ckpt["config"]
    model = SignLanguageTransformer(
        feature_dim=mc["feature_dim"],
        num_classes=mc["num_classes"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[Eval] Loaded checkpoint from epoch {ckpt['epoch']}  (val@1 during training: {ckpt['val_top1']:.2f}%)")
    return model


def evaluate(model, loader, device, vocab):
    all_preds  = []
    all_labels = []
    all_top5   = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            top5  = torch.topk(probs, 5, dim=1).indices.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_top5.extend(top5.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    top1 = (all_preds == all_labels).mean() * 100
    top5 = np.mean([all_labels[i] in all_top5[i] for i in range(len(all_labels))]) * 100

    print(f"\n[Results] Test Top-1: {top1:.2f}%  |  Test Top-5: {top5:.2f}%")
    print(f"          Total samples: {len(all_labels)}\n")

    # Per-class accuracy
    unique_classes = sorted(set(all_labels.tolist()))
    print(f"{'Class':>5}  {'Gloss':<20}  {'Correct':>7}  {'Total':>5}  {'Acc':>7}")
    print("-" * 52)
    per_class = {}
    for cls in unique_classes:
        mask    = all_labels == cls
        correct = (all_preds[mask] == cls).sum()
        total   = mask.sum()
        acc     = correct / total * 100 if total > 0 else 0
        gloss   = vocab.get(str(cls), vocab.get(cls, f"cls_{cls}"))
        per_class[cls] = {"gloss": gloss, "correct": int(correct), "total": int(total), "acc": acc}
        print(f"{cls:>5}  {gloss:<20}  {correct:>7}  {total:>5}  {acc:>6.1f}%")

    return all_preds, all_labels, per_class, top1, top5


def plot_confusion(all_preds, all_labels, vocab, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        classes = sorted(set(all_labels.tolist()))
        labels  = [vocab.get(str(c), vocab.get(c, str(c))) for c in classes]
        cm = confusion_matrix(all_labels, all_preds, labels=classes)

        fig_size = max(16, len(classes) // 3)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix - WLASL 100", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"[Eval] Confusion matrix saved to {out_path}")
    except ImportError:
        print("[Eval] Install matplotlib, seaborn, sklearn for confusion matrix plot")


def main():
    device = torch.device(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    # Load vocabulary
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)

    model = load_model(ckpt_path, device)

    test_ds = WLASLDataset("test", augment=False)
    loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.NUM_WORKERS, pin_memory=True)

    all_preds, all_labels, per_class, top1, top5 = evaluate(model, loader, device, vocab)

    # Save results
    results_path = os.path.join(cfg.CHECKPOINTS_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({"top1": top1, "top5": top5, "per_class": {str(k): v for k, v in per_class.items()}}, f, indent=2)
    print(f"[Eval] Results saved to {results_path}")

    # Confusion matrix
    cm_path = os.path.join(cfg.CHECKPOINTS_DIR, "confusion_matrix.png")
    plot_confusion(all_preds, all_labels, vocab, cm_path)


if __name__ == "__main__":
    main()
