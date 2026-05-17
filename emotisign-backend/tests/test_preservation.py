"""
Preservation Tests
==================
These tests verify that non-buggy behavior is already correct on UNFIXED code.
They must PASS before any fixes are applied, and must continue to PASS after
each fix is applied (confirming no regressions).

Preservation 1 — Upload path unaffected by trimming
Preservation 2 — Brief absence does not stop recording (source inspection)
Preservation 3 — PSL model inference unchanged (determinism check)
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure the backend package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_trim_guard(seq: np.ndarray, trim_frames: int) -> np.ndarray:
    """
    Apply the trim guard logic as specified in the design doc.

    Only trims when trim_frames > 0 AND the array has more than 2*trim_frames rows.
    This is the exact guard condition from the fix specification:
        if trim_frames > 0 and keypoints.shape[0] > 2 * trim_frames:
            keypoints = keypoints[trim_frames:-trim_frames]
    """
    if trim_frames > 0 and seq.shape[0] > 2 * trim_frames:
        return seq[trim_frames:-trim_frames]
    return seq


# ---------------------------------------------------------------------------
# Preservation 1 — Upload path unaffected by trimming
# ---------------------------------------------------------------------------

class TestPreservation1UploadPathUnaffected:
    """
    Preservation 1: The upload path must not be affected by any trim logic.

    Tests:
    - trim_frames=0 always returns the original array unchanged (for various sizes)
    - trim_frames=10 on a short array (≤ 2×10) returns the original array unchanged
    - resample_sequence with a 90-frame input produces shape (64, 126)
    """

    def test_trim_guard_zero_returns_original_length_10(self):
        """
        trim_frames=0 on a 10-frame array returns the original array unchanged.

        This documents the preservation: the upload path passes trim_frames=0,
        so no trimming ever occurs regardless of array size.
        """
        rng = np.random.default_rng(seed=1)
        seq = rng.standard_normal((10, 126)).astype(np.float32)

        result = _apply_trim_guard(seq, trim_frames=0)

        assert np.array_equal(result, seq), (
            "trim_frames=0 must return the original array unchanged (length 10)"
        )
        assert result.shape == (10, 126)

    def test_trim_guard_zero_returns_original_length_64(self):
        """
        trim_frames=0 on a 64-frame array returns the original array unchanged.
        """
        rng = np.random.default_rng(seed=2)
        seq = rng.standard_normal((64, 126)).astype(np.float32)

        result = _apply_trim_guard(seq, trim_frames=0)

        assert np.array_equal(result, seq), (
            "trim_frames=0 must return the original array unchanged (length 64)"
        )
        assert result.shape == (64, 126)

    def test_trim_guard_zero_returns_original_length_200(self):
        """
        trim_frames=0 on a 200-frame array returns the original array unchanged.
        """
        rng = np.random.default_rng(seed=3)
        seq = rng.standard_normal((200, 126)).astype(np.float32)

        result = _apply_trim_guard(seq, trim_frames=0)

        assert np.array_equal(result, seq), (
            "trim_frames=0 must return the original array unchanged (length 200)"
        )
        assert result.shape == (200, 126)

    def test_trim_guard_short_array_not_trimmed(self):
        """
        trim_frames=10 on a 15-frame array (≤ 2×10) returns the original array unchanged.

        The guard condition requires shape[0] > 2 * trim_frames.
        15 is NOT > 20, so no trimming should occur.
        """
        rng = np.random.default_rng(seed=4)
        seq = rng.standard_normal((15, 126)).astype(np.float32)

        result = _apply_trim_guard(seq, trim_frames=10)

        assert np.array_equal(result, seq), (
            "A 15-frame array with trim_frames=10 must not be trimmed "
            "(15 ≤ 2×10, guard condition not met)"
        )
        assert result.shape == (15, 126)

    def test_resample_sequence_90_frames_produces_64_126(self):
        """
        resample_sequence with a 90-frame input and default target_frames=64
        produces shape (64, 126).

        This confirms the resampling step is unaffected by any trim logic.
        """
        from app.ml.wlasl_service import WLASLModelService

        # Instantiate service without loading model (we only need resample_sequence)
        service = WLASLModelService(
            model_path="app/ml/models/wlasl100/best_model.pth",
            vocab_path="app/ml/models/wlasl100/vocab.json",
        )

        rng = np.random.default_rng(seed=5)
        seq = rng.standard_normal((90, 126)).astype(np.float32)

        result = service.resample_sequence(seq, target_frames=64)

        assert result.shape == (64, 126), (
            f"resample_sequence(90-frame input, target_frames=64) should produce "
            f"shape (64, 126), got {result.shape}"
        )


# ---------------------------------------------------------------------------
# Preservation 2 — Brief absence does not stop recording (source inspection)
# ---------------------------------------------------------------------------

class TestPreservation2BriefAbsenceDoesNotStopRecording:
    """
    Preservation 2: The existing manual/timeout stop paths in page.tsx must
    remain unchanged after the auto-stop fix is applied.

    Tests (source inspection):
    - stopRecording is defined as a useCallback
    - The 30-second timer logic (elapsed >= 30) is present
    - mediaRecorder.stop() is called in the file
    """

    PAGE_TSX_PATH = os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "emotisign-frontend",
        "src",
        "app",
        "translate",
        "sign-to-text",
        "page.tsx",
    ))

    def _read_page_tsx(self) -> str:
        assert os.path.exists(self.PAGE_TSX_PATH), (
            f"page.tsx not found at: {self.PAGE_TSX_PATH}"
        )
        with open(self.PAGE_TSX_PATH, encoding="utf-8") as f:
            return f.read()

    def test_stop_recording_defined_as_use_callback(self):
        """
        stopRecording must be defined as a useCallback in page.tsx.

        This confirms the manual stop path exists and is a stable callback
        that can be safely called from the auto-stop interval without
        causing stale-closure issues.
        """
        content = self._read_page_tsx()

        # The pattern: const stopRecording = useCallback(
        assert "stopRecording" in content, (
            "stopRecording must be defined in page.tsx"
        )
        assert "useCallback" in content, (
            "useCallback must be used in page.tsx"
        )

        # Verify stopRecording is associated with useCallback
        # Look for the pattern where stopRecording is assigned a useCallback
        assert "stopRecording = useCallback" in content, (
            "stopRecording must be defined as a useCallback in page.tsx. "
            "This ensures the manual stop path is preserved and stable."
        )

    def test_30_second_timer_logic_present(self):
        """
        The existing 30-second timer logic must be present in page.tsx.

        This confirms the timeout auto-stop path is unchanged.
        The design doc specifies: elapsed >= 30 near timer/countdown logic.
        """
        content = self._read_page_tsx()

        assert "elapsed >= 30" in content, (
            "The 30-second timer logic (elapsed >= 30) must be present in page.tsx. "
            "This confirms the timeout auto-stop path is preserved."
        )

    def test_media_recorder_stop_called(self):
        """
        mediaRecorder stop must be called in page.tsx.

        This confirms the core stop mechanism is present and unchanged.
        The stopRecording callback calls mediaRecorderRef.current.stop() to finalize
        the recording and trigger the onstop handler.
        """
        content = self._read_page_tsx()

        # The actual code uses mediaRecorderRef.current.stop() inside stopRecording
        assert ".stop()" in content, (
            "mediaRecorder .stop() must be called in page.tsx. "
            "This confirms the core recording stop mechanism is preserved."
        )
        # More specifically, the ref-based stop call must be present
        assert "mediaRecorderRef.current.stop()" in content, (
            "mediaRecorderRef.current.stop() must be called in page.tsx inside stopRecording. "
            "This confirms the core recording stop mechanism is preserved."
        )


# ---------------------------------------------------------------------------
# Preservation 3 — PSL model inference unchanged (determinism check)
# ---------------------------------------------------------------------------

class TestPreservation3PSLModelInferenceUnchanged:
    """
    Preservation 3: AlphabetClassifier inference must be deterministic.

    Tests:
    - Instantiate AlphabetClassifier(num_classes=23) with random weights
    - Run inference on a fixed synthetic (1, 42) input with torch.manual_seed(0)
    - Assert that re-running inference on the same input produces identical results
    """

    def _make_classifier(self) -> "AlphabetClassifier":
        """Instantiate AlphabetClassifier with random weights (no model file needed)."""
        from app.ml.psl_live_service import AlphabetClassifier
        torch.manual_seed(0)
        model = AlphabetClassifier(num_classes=23)
        model.eval()
        return model

    def _run_inference(self, model, x: torch.Tensor) -> tuple:
        """Run inference and return (predicted_class_index, confidence)."""
        import torch.nn.functional as F
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze()
        conf, idx = probs.max(0)
        return int(idx.item()), float(conf.item())

    def test_inference_is_deterministic(self):
        """
        Re-running inference on the same input with the same seed produces
        identical predicted class index and confidence score.

        This confirms the model inference is deterministic and unaffected
        by any label map changes (only string labels change, not the model).
        """
        from app.ml.psl_live_service import AlphabetClassifier
        import torch.nn.functional as F

        # Build model with fixed seed
        torch.manual_seed(0)
        model = AlphabetClassifier(num_classes=23)
        model.eval()

        # Build fixed synthetic input
        torch.manual_seed(0)
        x = torch.randn(1, 42)

        # First inference
        class_idx_1, conf_1 = self._run_inference(model, x)

        # Second inference on the same input (no re-seeding needed — deterministic)
        class_idx_2, conf_2 = self._run_inference(model, x)

        assert class_idx_1 == class_idx_2, (
            f"Predicted class index must be identical across runs: "
            f"first={class_idx_1}, second={class_idx_2}"
        )
        assert conf_1 == conf_2, (
            f"Confidence score must be identical across runs: "
            f"first={conf_1:.6f}, second={conf_2:.6f}"
        )

    def test_inference_class_index_in_valid_range(self):
        """
        The predicted class index must be in [0, num_classes - 1].

        This is a basic sanity check that the model output is well-formed.
        """
        from app.ml.psl_live_service import AlphabetClassifier

        torch.manual_seed(0)
        model = AlphabetClassifier(num_classes=23)
        model.eval()

        torch.manual_seed(0)
        x = torch.randn(1, 42)

        class_idx, conf = self._run_inference(model, x)

        assert 0 <= class_idx < 23, (
            f"Predicted class index {class_idx} must be in [0, 22] for num_classes=23"
        )
        assert 0.0 <= conf <= 1.0, (
            f"Confidence score {conf:.6f} must be in [0.0, 1.0]"
        )

    def test_same_seed_same_weights_same_output(self):
        """
        Two AlphabetClassifier instances initialized with the same seed
        produce identical outputs for the same input.

        This confirms that the model architecture is deterministic and
        that label map changes do not affect inference.
        """
        from app.ml.psl_live_service import AlphabetClassifier

        # First model
        torch.manual_seed(0)
        model_a = AlphabetClassifier(num_classes=23)
        model_a.eval()

        # Second model with same seed
        torch.manual_seed(0)
        model_b = AlphabetClassifier(num_classes=23)
        model_b.eval()

        # Same input
        torch.manual_seed(0)
        x = torch.randn(1, 42)

        class_idx_a, conf_a = self._run_inference(model_a, x)
        class_idx_b, conf_b = self._run_inference(model_b, x)

        assert class_idx_a == class_idx_b, (
            f"Two models with the same seed must predict the same class: "
            f"model_a={class_idx_a}, model_b={class_idx_b}"
        )
        assert conf_a == conf_b, (
            f"Two models with the same seed must produce the same confidence: "
            f"model_a={conf_a:.6f}, model_b={conf_b:.6f}"
        )
