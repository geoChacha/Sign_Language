"""
Training Module for ANN-Based PSL Alphabet Recognition System

Implements the training loop with on-the-fly data augmentation, early stopping,
and per-epoch logging. The model output size is determined dynamically from the
dataset so all alphabet classes present in the dataset are supported.

Augmentation applied only to training batches (not validation/test):
    - Gaussian noise  (std = 0.02)
    - Random rotation (±15°, around wrist)
    - Random scaling  ([0.9, 1.1], around wrist)
    Applied in random order each sample.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import csv
import logging
from pathlib import Path
from typing import Tuple, Dict

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    MODEL_PATH,
    TRAINING_LOG_PATH,
)
from model import AlphabetClassifier
from preprocessor import augment_coordinates

logger = logging.getLogger(__name__)


def _set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: AlphabetClassifier,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    label_map: Dict[int, str],
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> None:
    """
    Train the AlphabetClassifier with early stopping and logging.

    Args:
        model:      AlphabetClassifier instance (already instantiated with correct num_classes)
        train_data: (X_train, y_train)
        val_data:   (X_val,   y_val)
        test_data:  (X_test,  y_test)
        label_map:  {int → alphabet string} — saved alongside model weights
        epochs:     Maximum training epochs
        batch_size: Mini-batch size
        lr:         Adam learning rate
        patience:   Early-stopping patience (epochs without val_loss improvement)

    Saves:
        MODEL_PATH        — checkpoint dict with weights, num_classes, label_map
        TRAINING_LOG_PATH — CSV with epoch, train_loss, val_loss, val_accuracy
    """
    _set_seeds()

    X_train, y_train = train_data
    X_val,   y_val   = val_data
    X_test,  y_test  = test_data

    # ── PyTorch tensors ──────────────────────────────────────────────────────
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).long()
    X_test_t  = torch.from_numpy(X_test).float()
    y_test_t  = torch.from_numpy(y_test).long()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training log CSV ─────────────────────────────────────────────────────
    log_dir = Path(TRAINING_LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])

    # ── Early stopping state ─────────────────────────────────────────────────
    best_val_loss   = float('inf')
    patience_counter = 0
    best_state      = None
    best_epoch      = 0

    logger.info(f"Training started — {epochs} max epochs, patience={patience}, "
                f"num_classes={model.num_classes}")

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            # Apply augmentation on CPU numpy, then back to tensor
            aug_X = torch.stack([
                torch.from_numpy(augment_coordinates(x.numpy()))
                for x in batch_X
            ]).float()

            optimizer.zero_grad()
            loss = criterion(model(aug_X), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_X)

        train_loss = running_loss / len(X_train)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits  = model(X_val_t)
            val_loss    = criterion(val_logits, y_val_t).item()
            val_acc     = (val_logits.argmax(1) == y_val_t).float().mean().item()

        # ── Log ──────────────────────────────────────────────────────────────
        with open(TRAINING_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1,
                                    f"{train_loss:.4f}",
                                    f"{val_loss:.4f}",
                                    f"{val_acc:.4f}"])

        logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}")

        # ── Early stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch + 1
            patience_counter = 0

            # Save checkpoint (weights + metadata for consistent loading)
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': best_state,
                'num_classes':      model.num_classes,
                'label_map':        label_map,
            }, MODEL_PATH)
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                            f"(best was epoch {best_epoch})")
                break

    # ── Restore best weights ─────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best weights from epoch {best_epoch}")

    # ── Final test accuracy ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    logger.info(f"Final val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Val  accuracy   : {val_acc:.4f}")
    print(f"  Test accuracy   : {test_acc:.4f}")
    print(f"  Model saved to  : {MODEL_PATH}")
    print(f"  Training log    : {TRAINING_LOG_PATH}")
    print(f"{'='*60}\n")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    from dataset_loader import load_dataset
    from preprocessor import normalize_hand_coords

    logger.info("Loading dataset …")
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), label_map = load_dataset()

    # Normalise all splits
    X_tr = np.array([normalize_hand_coords(x) for x in X_tr])
    X_v  = np.array([normalize_hand_coords(x) for x in X_v])
    X_te = np.array([normalize_hand_coords(x) for x in X_te])

    num_classes = len(label_map)
    logger.info(f"Detected {num_classes} classes — building model …")

    model = AlphabetClassifier(num_classes=num_classes)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_model(
        model,
        (X_tr, y_tr),
        (X_v,  y_v),
        (X_te, y_te),
        label_map,
    )
