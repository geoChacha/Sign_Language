"""
model.py - TCN + BiGRU model for WLASL-100.

Why switched from Transformer:
  Transformers need large datasets to learn useful attention patterns.
  WLASL-100 has ~20 samples per class - too small. Self-attention
  memorizes training signers instead of learning sign shapes.

  BiGRU is a better fit:
  - Processes sequences recurrently - naturally captures temporal order
  - Far fewer parameters (~400K vs ~2M) - much less memorization
  - TCN front-end captures local motion (velocity/acceleration)
  - Still exports cleanly to TorchScript and ONNX for mobile

Architecture:
  Input (B, T, 225)
    -> Separate linear projections per body part (L-hand, R-hand, pose)
    -> TCN: 3 residual 1D-conv blocks for local motion features
    -> BiGRU: 2 layers, captures long-range temporal dependencies
    -> Attention pooling over time (better than just last hidden state)
    -> Classifier MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Single residual 1D-conv block."""
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.net(x) + x)


class AttentionPooling(nn.Module):
    """
    Soft attention over time steps - learns which frames matter most
    for each sign. Much better than mean/last pooling for variable-speed signs.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (weights * x).sum(dim=1)                 # (B, H)


class SignLanguageTransformer(nn.Module):
    """
    TCN + BiGRU model for sign language recognition.
    Named SignLanguageTransformer for drop-in compatibility with existing
    training/export/inference scripts.
    """
    def __init__(
        self,
        feature_dim: int = 225,
        num_classes: int = 100,
        d_model: int = 192,
        nhead: int = 4,           # unused, kept for API compatibility
        num_layers: int = 3,      # maps to GRU layers (capped at 2)
        dim_feedforward: int = 384,
        dropout: float = 0.4,
        max_seq_len: int = 64,    # unused, kept for API compatibility
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Number of GRU layers: use min(num_layers, 2) - more than 2 rarely helps
        gru_layers = min(num_layers, 2)

        # Input projections - hands only (pose removed for framing invariance)
        # Each hand gets d_model//2 so concatenation gives d_model
        lh_dim = d_model // 2
        rh_dim = d_model // 2
        self.proj_lhand = nn.Linear(63, lh_dim)
        self.proj_rhand = nn.Linear(63, rh_dim)
        self.proj_merge = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # TCN: 3 residual blocks for local motion features
        self.tcn = nn.Sequential(
            TCNBlock(d_model, kernel_size=3, dropout=dropout * 0.5),
            TCNBlock(d_model, kernel_size=3, dropout=dropout * 0.5),
            TCNBlock(d_model, kernel_size=5, dropout=dropout * 0.5),
        )

        # BiGRU: captures long-range temporal dependencies
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = d_model * 2   # bidirectional

        # Attention pooling over time
        self.attn_pool = AttentionPooling(gru_out_dim)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Linear(gru_out_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 126) -> (B, gru_out_dim)  -- hands only, pose dropped"""
        # Use only first 126 dims (L-hand + R-hand), ignore pose if present
        lh     = self.proj_lhand(x[:, :, :63])
        rh     = self.proj_rhand(x[:, :, 63:126])
        tokens = self.proj_merge(torch.cat([lh, rh], dim=-1))  # (B, T, d_model)

        # TCN: (B, T, C) -> (B, C, T) -> TCN -> (B, T, C)
        tcn_out = self.tcn(tokens.permute(0, 2, 1)).permute(0, 2, 1)

        # BiGRU
        gru_out, _ = self.gru(tcn_out)    # (B, T, d_model*2)

        # Attention pooling
        return self.attn_pool(gru_out)    # (B, d_model*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_dist * log_probs).sum(dim=-1).mean()


def count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"


if __name__ == "__main__":
    import config as cfg
    m = SignLanguageTransformer(
        feature_dim=cfg.FEATURE_DIM,
        num_classes=cfg.NUM_CLASSES,
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        num_layers=cfg.NUM_LAYERS,
        dim_feedforward=cfg.DIM_FF,
        dropout=cfg.DROPOUT,
        max_seq_len=cfg.MAX_SEQ_LEN,
    )
    print(count_params(m))
    dummy = torch.randn(4, cfg.NUM_FRAMES, cfg.FEATURE_DIM)
    out   = m(dummy)
    print(f"Output shape: {out.shape}")