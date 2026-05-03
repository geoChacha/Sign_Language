"""
9_slp_train.py - Train the Sign Language Production model.

Run:
    python 9_slp_train.py

Trains gloss→pose sequence model. Best checkpoint saved to
checkpoints/slp_best_model.pth

After training, run:
    python 10_slp_render.py --sentence "hello how are you"
"""

import os
import json
import time
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

import config as cfg
from slp_dataset import get_slp_loaders
from slp_model import SignProductionModel, SLPLoss, count_params


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, epoch, warmup, total, base_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total - warmup, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    lr = max(lr, 1e-6)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for gloss_idx, pose_seq in loader:
            gloss_idx = gloss_idx.to(device)
            pose_seq  = pose_seq.to(device)

            with autocast():
                pred = model(gloss_idx, pose_seq)
                loss = criterion(pred, pose_seq)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


def main():
    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE)
    print(f"[SLP Train] Device: {device}")

    train_loader, val_loader, vocab, g2i = get_slp_loaders()

    model = SignProductionModel(
        num_glosses=cfg.NUM_CLASSES,
        pose_dim=225,           # full keypoints including pose
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.2,
        max_seq_len=cfg.NUM_FRAMES,
    ).to(device)
    print(f"[SLP Train] {count_params(model)}")

    criterion = SLPLoss(vel_weight=0.5, acc_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                   weight_decay=1e-4, betas=(0.9, 0.98))
    scaler    = GradScaler()

    NUM_EPOCHS = 120
    PATIENCE   = 20
    best_val   = float("inf")
    patience_counter = 0
    best_ckpt  = os.path.join(cfg.CHECKPOINTS_DIR, "slp_best_model.pth")
    history    = []

    for epoch in range(NUM_EPOCHS):
        warmup_cosine(optimizer, epoch, warmup=10, total=NUM_EPOCHS, base_lr=1e-3)
        lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()

        tr_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va_loss = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS}  lr={lr:.2e}  "
              f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  [{elapsed:.0f}s]")

        history.append({"epoch": epoch+1, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "val_loss": va_loss,
                "config": {
                    "num_glosses": cfg.NUM_CLASSES,
                    "pose_dim": 225,
                    "d_model": 256,
                    "nhead": 4,
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 3,
                    "dim_feedforward": 512,
                    "dropout": 0.0,   # no dropout at inference
                    "max_seq_len": cfg.NUM_FRAMES,
                },
            }, best_ckpt)
            print(f"  --> Best model saved (val_loss={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[SLP Train] Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(cfg.CHECKPOINTS_DIR, "slp_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[SLP Train] Done. Best val_loss={best_val:.4f}")
    print(f"[SLP Train] Run: python 10_slp_render.py --sentence \"I need help\"")


if __name__ == "__main__":
    main()