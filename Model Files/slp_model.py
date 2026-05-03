"""
8_slp_model.py - Sign Language Production model.

Architecture: Embedding → Transformer Decoder → Pose sequence

Given a gloss index (word), generates a sequence of 225-dim pose
vectors representing how that word is signed.

Why Transformer Decoder (not encoder):
  - Pose generation is autoregressive - each frame depends on the previous
  - Decoder with causal masking naturally models this
  - Teacher forcing during training: feed real pose frames as input,
    predict next frame as output

Input  (training): (B,) gloss indices  +  (B, T, 225) real pose sequence
Output (training): (B, T, 225) predicted pose sequence (shifted by 1)
Output (inference): (B, T, 225) generated pose sequence (autoregressive)

Loss: MSE on pose coordinates + velocity loss (smooth motion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignProductionModel(nn.Module):
    """
    Gloss → Pose sequence model.

    Training (teacher forcing):
        - Gloss embedding becomes memory for cross-attention
        - Real pose frames (shifted right) are the decoder input
        - Model predicts next frame at each position

    Inference (autoregressive):
        - Start from zero frame
        - Generate one frame at a time
        - Feed generated frame back as next input
    """

    def __init__(
        self,
        num_glosses: int,        # vocabulary size (100 for WLASL-100)
        pose_dim: int = 225,     # keypoint feature dim per frame
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.pose_dim    = pose_dim
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # ── Gloss encoder ──────────────────────────────────────────────────
        # Embed the gloss, then expand to a short sequence so the decoder
        # has rich cross-attention context (not just a single vector)
        self.gloss_embed = nn.Embedding(num_glosses, d_model)
        # Expand single embedding → sequence of context_len vectors
        self.context_len = 8
        self.gloss_expand = nn.Sequential(
            nn.Linear(d_model, d_model * self.context_len),
            nn.GELU(),
        )
        self.gloss_pos = PositionalEncoding(d_model, max_len=self.context_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.gloss_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers,
                                                    enable_nested_tensor=False)

        # ── Pose input projection ──────────────────────────────────────────
        self.pose_in  = nn.Linear(pose_dim, d_model)
        self.pose_pos = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # ── Transformer decoder ────────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # ── Pose output projection ─────────────────────────────────────────
        self.pose_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pose_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def encode_gloss(self, gloss_idx: torch.Tensor) -> torch.Tensor:
        """
        gloss_idx: (B,) -> memory: (B, context_len, d_model)
        """
        B = gloss_idx.size(0)
        emb = self.gloss_embed(gloss_idx)                        # (B, d_model)
        expanded = self.gloss_expand(emb)                        # (B, d_model * context_len)
        expanded = expanded.view(B, self.context_len, self.d_model)  # (B, CL, d_model)
        expanded = self.gloss_pos(expanded)
        memory   = self.gloss_encoder(expanded)                  # (B, CL, d_model)
        return memory

    def forward(self, gloss_idx: torch.Tensor, pose_seq: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.

        gloss_idx : (B,)
        pose_seq  : (B, T, 225) - target pose sequence
        returns   : (B, T, 225) - predicted pose at each step
        """
        B, T, _ = pose_seq.shape

        # Gloss → memory
        memory = self.encode_gloss(gloss_idx)          # (B, CL, d_model)

        # Shift pose right: prepend zero frame as start token
        start  = torch.zeros(B, 1, self.pose_dim, device=pose_seq.device)
        dec_in = torch.cat([start, pose_seq[:, :-1]], dim=1)   # (B, T, 225)

        # Project poses to d_model
        dec_tokens = self.pose_pos(self.pose_in(dec_in))       # (B, T, d_model)

        # Causal mask - each position can only attend to previous positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=pose_seq.device)

        # Decode
        out = self.decoder(dec_tokens, memory, tgt_mask=causal_mask,
                           tgt_is_causal=True)                 # (B, T, d_model)

        return self.pose_out(out)                              # (B, T, 225)

    @torch.no_grad()
    def generate(self, gloss_idx: torch.Tensor, n_frames: int = 64) -> torch.Tensor:
        """
        Autoregressive generation at inference time.

        gloss_idx : (B,) or scalar int
        returns   : (B, n_frames, 225)
        """
        if isinstance(gloss_idx, int):
            gloss_idx = torch.tensor([gloss_idx])
        gloss_idx = gloss_idx.to(next(self.parameters()).device)
        B = gloss_idx.size(0)

        memory   = self.encode_gloss(gloss_idx)
        # Start with zero frame
        generated = torch.zeros(B, 1, self.pose_dim, device=gloss_idx.device)

        for t in range(n_frames):
            dec_tokens  = self.pose_pos(self.pose_in(generated))
            T_cur       = generated.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T_cur, device=gloss_idx.device)
            out  = self.decoder(dec_tokens, memory, tgt_mask=causal_mask, tgt_is_causal=True)
            next_frame = self.pose_out(out[:, -1:])            # (B, 1, 225)
            generated  = torch.cat([generated, next_frame], dim=1)

        return generated[:, 1:]   # drop the initial zero start frame  -> (B, T, 225)

    @torch.no_grad()
    def generate_sentence(self, gloss_indices: list, n_frames_per_word: int = 64,
                          transition_frames: int = 8) -> torch.Tensor:
        """
        Generate a full sentence by concatenating word animations with
        smooth linear interpolation transitions between words.

        gloss_indices : list of int - one per word in sentence
        returns       : (T_total, 225) pose sequence
        """
        word_seqs = []
        for gidx in gloss_indices:
            seq = self.generate(gidx, n_frames=n_frames_per_word)  # (1, T, 225)
            word_seqs.append(seq[0])   # (T, 225)

        if len(word_seqs) == 1:
            return word_seqs[0]

        # Stitch with linear interpolation transitions
        result = [word_seqs[0]]
        for i in range(1, len(word_seqs)):
            prev_last  = word_seqs[i-1][-1:]    # (1, 225)
            next_first = word_seqs[i][:1]        # (1, 225)
            # Linear blend over transition_frames
            alphas = torch.linspace(0, 1, transition_frames + 2)[1:-1]
            transition = torch.stack([
                (1 - a) * prev_last[0] + a * next_first[0]
                for a in alphas
            ])   # (transition_frames, 225)
            result.append(transition)
            result.append(word_seqs[i])

        return torch.cat(result, dim=0)   # (T_total, 225)


class SLPLoss(nn.Module):
    """
    Combined loss for pose generation:
    - MSE on absolute pose coordinates
    - MSE on frame-to-frame velocity (smooth motion)
    - MSE on acceleration (prevents jerky transitions)
    """
    def __init__(self, vel_weight: float = 0.5, acc_weight: float = 0.2):
        super().__init__()
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pose loss
        pose_loss = F.mse_loss(pred, target)

        # Velocity loss - encourage smooth motion
        pred_vel   = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss   = F.mse_loss(pred_vel, target_vel)

        # Acceleration loss - penalise jerky changes
        pred_acc   = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        acc_loss   = F.mse_loss(pred_acc, target_acc)

        return pose_loss + self.vel_weight * vel_loss + self.acc_weight * acc_loss


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return f"Total: {total:,}"


if __name__ == "__main__":
    m = SignProductionModel(num_glosses=100)
    print(count_params(m))
    gloss = torch.randint(0, 100, (4,))
    pose  = torch.randn(4, 64, 225)
    out   = m(gloss, pose)
    print(f"Train output: {out.shape}")
    gen   = m.generate(gloss, n_frames=64)
    print(f"Generated:    {gen.shape}")
    sent  = m.generate_sentence([0, 5, 12], n_frames_per_word=64)
    print(f"Sentence:     {sent.shape}")