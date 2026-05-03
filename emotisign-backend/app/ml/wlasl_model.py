"""
wlasl_model.py - TCN + BiGRU model for WLASL-100 ASL recognition.

Architecture:
  Input (B, T, 126) - hands only (L-hand + R-hand keypoints)
    -> Separate linear projections per hand
    -> TCN: 3 residual 1D-conv blocks for local motion features
    -> BiGRU: 2 layers, captures long-range temporal dependencies
    -> Attention pooling over time
    -> Classifier MLP head
    -> Output (B, 100) - logits for 100 ASL signs

Why TCN + BiGRU instead of Transformer:
  - WLASL-100 has ~20 samples per class (too small for Transformers)
  - BiGRU naturally captures temporal order with fewer parameters
  - TCN captures local motion (velocity/acceleration)
  - ~400K parameters vs ~2M for Transformer (less memorization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class TCNBlock(nn.Module):
    """Single residual 1D-conv block for temporal feature extraction."""
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - batch, channels, time
        Returns:
            (B, C, T) - residual output
        """
        return self.drop(self.net(x) + x)


class AttentionPooling(nn.Module):
    """
    Soft attention over time steps - learns which frames matter most
    for each sign. Better than mean/last pooling for variable-speed signs.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H) - batch, time, hidden
        Returns:
            (B, H) - pooled representation
        """
        weights = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (weights * x).sum(dim=1)                 # (B, H)


class SignLanguageTransformer(nn.Module):
    """
    TCN + BiGRU model for sign language recognition.
    Named SignLanguageTransformer for compatibility with existing checkpoints.
    """
    
    def __init__(
        self,
        feature_dim: int = 126,        # Hands only (L + R)
        num_classes: int = 100,        # WLASL-100 vocabulary
        d_model: int = 192,            # Hidden dimension
        nhead: int = 4,                # Unused (kept for API compatibility)
        num_layers: int = 2,           # GRU layers
        dim_feedforward: int = 256,    # MLP hidden size
        dropout: float = 0.4,          # Dropout rate
        max_seq_len: int = 64,         # Unused (kept for API compatibility)
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_classes = num_classes

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
        """Initialize model weights."""
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
        """
        Encode keypoint sequence to feature representation.
        
        Args:
            x: (B, T, 126) - batch, time, features (hands only)
        Returns:
            (B, gru_out_dim) - encoded representation
        """
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
        """
        Forward pass.
        
        Args:
            x: (B, T, 126) - batch, time, features
        Returns:
            (B, num_classes) - logits
        """
        return self.head(self.encode(x))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu"
    ) -> "SignLanguageTransformer":
        """
        Load model from checkpoint file.

        Uses weights_only=True (safe context) to prevent arbitrary code
        execution from untrusted checkpoint files. Falls back to
        weights_only=False only if the checkpoint contains non-tensor
        objects (e.g. config dicts saved with older PyTorch versions).

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model on ('cuda' or 'cpu')
        Returns:
            Loaded model in eval mode
        """
        # Try safe load first (weights_only=True — no arbitrary code execution)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except Exception:
            # Fallback for checkpoints that contain non-tensor objects (config dicts)
            # Only use with checkpoints you trust (i.e. your own trained weights)
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

        # Extract model config from checkpoint
        config = checkpoint.get("config", {})

        # Create model with checkpoint config
        model = cls(
            feature_dim=config.get("feature_dim", 126),
            num_classes=config.get("num_classes", 100),
            d_model=config.get("d_model", 192),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=0.0,  # Disable dropout for inference
            max_seq_len=config.get("max_seq_len", 64),
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        return model


def count_params(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    Returns:
        Dictionary with 'total' and 'trainable' parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Test model creation
    model = SignLanguageTransformer(
        feature_dim=126,
        num_classes=100,
        d_model=192,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.4,
    )
    
    params = count_params(model)
    print(f"Model parameters: Total={params['total']:,}, Trainable={params['trainable']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 64, 126)  # (batch=4, time=64, features=126)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 100)")
