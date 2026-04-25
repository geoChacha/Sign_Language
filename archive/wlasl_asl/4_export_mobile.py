"""
4_export_mobile.py - Export trained model for mobile deployment.

Exports:
1. TorchScript (.pt)     - for Android (via PyTorch Mobile) or iOS (via LibTorch)
2. ONNX (.onnx)          - universal, works with ONNX Runtime Mobile, TFLite converter
3. Quantized TorchScript - INT8 dynamic quantization for smaller size & faster CPU

Run:
    python 4_export_mobile.py

Output files:
    checkpoints/
        model_scripted.pt          - full precision TorchScript
        model_quantized.pt         - INT8 dynamic quantized TorchScript
        model.onnx                 - ONNX float32
        vocab.json                 - word list
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import config as cfg
from model import SignLanguageTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Inference wrapper: takes raw numpy-equivalent tensor, returns top-k results
# ──────────────────────────────────────────────────────────────────────────────

class InferenceWrapper(nn.Module):
    """
    Wraps the model so mobile apps only need to pass a single float tensor
    and get back predicted class indices and probabilities.

    Input:  (1, T, 225)  float32
    Output: (top_k,) int64 indices, (top_k,) float32 probs
    """
    def __init__(self, model: SignLanguageTransformer, top_k: int = 5):
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        # x: (1, T, 225)
        logits = self.model(x)                     # (1, num_classes)
        probs  = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        return topk_idx[0], topk_probs[0]


def load_model(ckpt_path: str) -> SignLanguageTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mc = ckpt["config"]
    model = SignLanguageTransformer(
        feature_dim=mc["feature_dim"],
        num_classes=mc["num_classes"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=0.0,    # disable dropout at inference
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def export_torchscript(wrapper: InferenceWrapper, out_path: str, dummy_input: torch.Tensor):
    scripted = torch.jit.trace(wrapper, dummy_input)
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Export] TorchScript (mobile lite): {out_path}  ({size_mb:.1f} MB)")


def export_torchscript_quantized(wrapper: InferenceWrapper, out_path: str, dummy_input: torch.Tensor):
    # Dynamic INT8 quantization (no calibration data needed)
    quantized = torch.quantization.quantize_dynamic(
        wrapper.model,
        {nn.Linear, nn.TransformerEncoderLayer},
        dtype=torch.qint8,
    )
    wrapper_q = InferenceWrapper(quantized)
    wrapper_q.eval()
    scripted = torch.jit.trace(wrapper_q, dummy_input)
    scripted.save(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Export] Quantized TorchScript: {out_path}  ({size_mb:.1f} MB)")


def export_onnx(model: SignLanguageTransformer, out_path: str, dummy_input: torch.Tensor, n_classes: int):
    # ONNX export from base model (no wrapper, cleaner graph)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={
            "keypoints": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Export] ONNX: {out_path}  ({size_mb:.1f} MB)")

    # Validate
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"keypoints": dummy_input.numpy()})
        assert out[0].shape == (1, n_classes), f"Unexpected shape: {out[0].shape}"
        print("[Export] ONNX validation passed")
    except ImportError:
        print("[Export] onnxruntime not installed, skipping ONNX validation (pip install onnxruntime)")


def export_vocab_for_mobile(vocab: dict, out_path: str):
    """
    Export vocab as a simple list (index -> gloss) for mobile apps.
    The integer keys are sorted so index i maps to vocab_list[i].
    """
    max_idx = max(int(k) for k in vocab.keys())
    vocab_list = [""] * (max_idx + 1)
    for k, v in vocab.items():
        vocab_list[int(k)] = v
    with open(out_path, "w") as f:
        json.dump(vocab_list, f, indent=2)
    print(f"[Export] Vocab list: {out_path}")


def main():
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    model = load_model(ckpt_path)
    wrapper = InferenceWrapper(model, top_k=5)
    wrapper.eval()

    dummy = torch.randn(1, cfg.NUM_FRAMES, cfg.FEATURE_DIM)

    # TorchScript (PyTorch Mobile lite interpreter)
    ts_path = os.path.join(cfg.CHECKPOINTS_DIR, "model_scripted.ptl")
    export_torchscript(wrapper, ts_path, dummy)

    # Quantized TorchScript
    ts_q_path = os.path.join(cfg.CHECKPOINTS_DIR, "model_quantized.pt")
    try:
        export_torchscript_quantized(wrapper, ts_q_path, dummy)
    except Exception as e:
        print(f"[Export] Quantized export failed: {e}")

    # ONNX
    onnx_path = os.path.join(cfg.CHECKPOINTS_DIR, "model.onnx")
    export_onnx(model, onnx_path, dummy, cfg.NUM_CLASSES)

    # Vocab for mobile
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "vocab.json")
    out_vocab   = os.path.join(cfg.CHECKPOINTS_DIR, "vocab_list.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        export_vocab_for_mobile(vocab, out_vocab)

    print("\n[Export] Done. Files ready for mobile integration:")
    print(f"  Android (PyTorch Mobile):  {ts_path}")
    print(f"  Cross-platform (ONNX):     {onnx_path}")
    print(f"  Quantized (smaller):       {ts_q_path}")
    print(f"  Word list:                 {out_vocab}")


if __name__ == "__main__":
    main()
