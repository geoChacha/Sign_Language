"""
2_train.py - Train the Sign Language Transformer on WLASL-100.

Features:
- Mixed precision training (AMP) for RTX 3070
- Cosine LR schedule with linear warm-up
- Label smoothing
- Gradient clipping
- Top-1 and Top-5 accuracy tracking
- Best model checkpoint + full training history JSON
- Early stopping

Run:
    python 2_train.py
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast          # FIX #3: updated import (torch.cuda.amp deprecated)

import argparse
import config as cfg
from model import SignLanguageTransformer, LabelSmoothingCrossEntropy, count_params
from dataset import get_loaders, get_trainval_loader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer) -> float:
    for pg in optimizer.param_groups:
        return pg["lr"]
    return 0.0


def warmup_cosine_schedule(optimizer, epoch: int, warmup: int, total: int, base_lr: float):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total - warmup, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    lr = max(lr, 1e-6)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        B = targets.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.unsqueeze(0).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum().item()
            results.append(correct_k / B * 100.0)
        return results


def run_epoch(model, loader, criterion, scaler, device, train: bool, optimizer=None):
    # FIX #2: optimizer is now an optional kwarg — only required when train=True
    model.train() if train else model.eval()
    total_loss = 0.0
    top1_total = 0.0
    top5_total = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda"):   # FIX #3: explicit device_type
                    logits = model(x)
                    loss   = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(device_type="cuda"):   # FIX #3: explicit device_type
                    logits = model(x)
                    loss   = criterion(logits, y)

            t1, t5 = accuracy(logits, y, topk=(1, 5))
            total_loss += loss.item()
            top1_total += t1
            top5_total += t5
            n_batches  += 1

    return total_loss / n_batches, top1_total / n_batches, top5_total / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval", action="store_true",
                        help="Train on train+val combined for fixed epochs (no early stopping). "
                             "Use after you know convergence epoch from a normal run.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS (useful with --trainval)")
    args = parser.parse_args()

    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    print(f"[Train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.trainval:
        print("[Train] Mode: train+val combined (fixed epochs, no early stopping)")
        train_loader, val_loader, vocab = get_trainval_loader()
        use_early_stopping = False
    else:
        train_loader, val_loader, _, vocab = get_loaders()
        use_early_stopping = True
    print(f"[Train] Batches/epoch: train={len(train_loader)}  val={len(val_loader)}")

    # Save vocabulary for inference
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[Train] Vocabulary saved to {vocab_path}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = SignLanguageTransformer(
        feature_dim=cfg.FEATURE_DIM,
        num_classes=cfg.NUM_CLASSES,
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        num_layers=cfg.NUM_LAYERS,
        dim_feedforward=cfg.DIM_FF,
        dropout=cfg.DROPOUT,
        max_seq_len=cfg.MAX_SEQ_LEN,
    ).to(device)
    print(f"[Train] Model params: {count_params(model)}")

    # ── Optimiser & schedule ───────────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    scaler = GradScaler(device="cuda")   # FIX #3: explicit device arg

    # ── Training loop ──────────────────────────────────────────────────────────
    num_epochs = args.epochs if args.epochs else cfg.NUM_EPOCHS   # FIX #1: use local var
    best_val_top1 = 0.0
    patience_counter = 0
    history = []
    best_ckpt = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")

    for epoch in range(num_epochs):
        warmup_cosine_schedule(optimizer, epoch, cfg.WARMUP_EPOCHS, num_epochs, cfg.LR)
        lr = get_lr(optimizer)

        t0 = time.time()
        tr_loss, tr_top1, tr_top5 = run_epoch(
            model, train_loader, criterion, scaler, device, train=True, optimizer=optimizer
        )
        va_loss, va_top1, va_top5 = run_epoch(
            model, val_loader, criterion, scaler, device, train=False
            # FIX #2: optimizer not passed for validation
        )
        elapsed = time.time() - t0

        # FIX #1: use num_epochs (respects --epochs override) instead of cfg.NUM_EPOCHS
        print(
            f"Epoch {epoch+1:03d}/{num_epochs}  lr={lr:.2e}  "
            f"tr_loss={tr_loss:.4f}  tr@1={tr_top1:.1f}  tr@5={tr_top5:.1f}  "
            f"va_loss={va_loss:.4f}  va@1={va_top1:.1f}  va@5={va_top5:.1f}  "
            f"[{elapsed:.0f}s]"
        )

        record = {
            "epoch": epoch + 1, "lr": lr,
            "train_loss": tr_loss, "train_top1": tr_top1, "train_top5": tr_top5,
            "val_loss": va_loss,   "val_top1":   va_top1, "val_top5":   va_top5,
        }
        history.append(record)

        # Save best
        if va_top1 > best_val_top1:
            best_val_top1 = va_top1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_top1": va_top1,
                "config": {
                    "feature_dim": cfg.FEATURE_DIM,
                    "num_classes": cfg.NUM_CLASSES,
                    "d_model": cfg.D_MODEL,
                    "nhead": cfg.NHEAD,
                    "num_layers": cfg.NUM_LAYERS,
                    "dim_feedforward": cfg.DIM_FF,
                    "dropout": cfg.DROPOUT,
                    "max_seq_len": cfg.MAX_SEQ_LEN,
                },
            }, best_ckpt)
            print(f"  --> Best model saved (val@1={best_val_top1:.2f}%)")
        else:
            patience_counter += 1
            if use_early_stopping and patience_counter >= cfg.PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch+1}")
                break

    # Save training history
    hist_path = os.path.join(cfg.CHECKPOINTS_DIR, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Done.  Best val@1={best_val_top1:.2f}%")
    print(f"[Train] Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()