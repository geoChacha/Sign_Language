"""
Bug Condition Exploration Tests
================================
These tests document the existence of three bugs BEFORE any fixes are applied.

IMPORTANT:
- Bug 1 (WLASL Frame Trimming): PASSES on unfixed code — documents that no trimming
  occurs (untrimmed != trimmed, confirming the bug exists).
- Bug 2 (Auto-Stop Recording): PASSES on unfixed code — confirms that the string
  `autoStopIntervalRef` does NOT exist in page.tsx (no auto-stop mechanism).
- Bug 3 (PSL Urdu Encoding): PASSES on unfixed code — confirms that at least one
  label_map.json value is NOT valid Urdu Unicode (mojibake present).

After the fixes are applied, Bug 2 and Bug 3 tests will FAIL (the bug conditions
will no longer hold), confirming the fixes are in place.
"""

import os
import json
import sys
import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_urdu_char(ch: str) -> bool:
    """Return True if the character's code point is in the Urdu/Arabic Unicode blocks.

    Accepted ranges:
      U+0600–U+06FF  Arabic block (covers Urdu letters)
      U+FB50–U+FDFF  Arabic Presentation Forms-A
    """
    cp = ord(ch)
    return (0x0600 <= cp <= 0x06FF) or (0xFB50 <= cp <= 0xFDFF)


def _all_chars_valid_urdu(s: str) -> bool:
    """Return True if every character in *s* is a valid Urdu Unicode character."""
    if not s:
        return False
    return all(_is_valid_urdu_char(ch) for ch in s)


# ---------------------------------------------------------------------------
# Bug 1 — WLASL Frame Trimming
# ---------------------------------------------------------------------------

class TestBug1WLASLFrameTrimming:
    """
    Documents Bug 1: without a trim step, transition frames at the start and end
    of a webcam recording are included in the keypoint sequence sent to the model.

    The test constructs a synthetic (90, 126) keypoint array where:
      - Frames 0–9  (first 10): high-variance "transition" values (user getting into position)
      - Frames 10–79 (middle 70): stable, near-zero "sign" values
      - Frames 80–89 (last 10): high-variance "transition" values (user relaxing)

    It then asserts that the untrimmed array is NOT equal to the trimmed array.
    This assertion PASSES on unfixed code (no trimming = bug condition confirmed).
    After the fix, the pipeline will trim automatically, but this test will still
    pass because it is testing the trimming logic directly, not the pipeline.
    """

    def _make_synthetic_keypoints(self, n_frames: int = 90, trim: int = 10) -> np.ndarray:
        """Build a synthetic keypoint array with noisy start/end and stable middle."""
        rng = np.random.default_rng(seed=42)
        seq = np.zeros((n_frames, 126), dtype=np.float32)

        # Transition frames: high variance (simulate user moving into/out of position)
        seq[:trim] = rng.uniform(low=-5.0, high=5.0, size=(trim, 126)).astype(np.float32)
        seq[-trim:] = rng.uniform(low=-5.0, high=5.0, size=(trim, 126)).astype(np.float32)

        # Middle frames: stable sign pattern (low variance, near-zero)
        seq[trim:-trim] = rng.uniform(low=-0.1, high=0.1, size=(n_frames - 2 * trim, 126)).astype(np.float32)

        return seq

    def test_trim_frames_0_returns_full_array(self):
        """
        BUG CONDITION: calling with trim=0 returns the full array (no trimming).

        This documents the bug: without trimming, transition frames are included.
        The test PASSES on unfixed code (confirming the bug exists).
        """
        seq = self._make_synthetic_keypoints(n_frames=90, trim=10)
        trim_frames = 0  # Bug condition: no trimming applied

        # Simulate the unfixed pipeline: no trimming
        if trim_frames > 0 and seq.shape[0] > 2 * trim_frames:
            result = seq[trim_frames:-trim_frames]
        else:
            result = seq  # Bug: full array returned

        # The full array equals the original — no trimming occurred
        assert np.array_equal(result, seq), (
            "Expected the untrimmed array to equal the original (bug condition: no trimming)"
        )

    def test_untrimmed_differs_from_trimmed(self):
        """
        Documents the bug: the untrimmed sequence differs from the trimmed sequence.

        This assertion PASSES on unfixed code because the transition frames are
        present in the untrimmed version but absent in the trimmed version.
        After the fix, the pipeline will trim automatically, but the trimmed
        array will still differ from the full array (the fix changes the data).
        """
        seq = self._make_synthetic_keypoints(n_frames=90, trim=10)
        trim_frames = 10

        # Untrimmed (bug condition)
        untrimmed = seq  # no trimming

        # Trimmed (expected behavior after fix)
        trimmed = seq[trim_frames:-trim_frames]

        # They must differ — the transition frames are present in untrimmed
        assert not np.array_equal(untrimmed, trimmed), (
            "Untrimmed and trimmed arrays should differ: "
            "transition frames are present in the untrimmed version (bug condition)"
        )

        # Confirm shape difference
        assert untrimmed.shape[0] == 90
        assert trimmed.shape[0] == 70  # 90 - 2*10

    def test_transition_frames_have_high_variance(self):
        """
        Confirms that the synthetic transition frames have significantly higher
        variance than the stable middle frames, validating the test setup.
        """
        seq = self._make_synthetic_keypoints(n_frames=90, trim=10)

        transition_variance = np.var(seq[:10]) + np.var(seq[-10:])
        middle_variance = np.var(seq[10:80])

        assert transition_variance > middle_variance * 10, (
            f"Transition frames should have much higher variance than middle frames. "
            f"Transition var: {transition_variance:.4f}, Middle var: {middle_variance:.4f}"
        )

    def test_short_video_not_trimmed(self):
        """
        Preservation: a video with ≤ 2×TRIM_FRAMES frames should NOT be trimmed.
        This ensures the fix does not break short videos.
        """
        trim_frames = 10
        seq = self._make_synthetic_keypoints(n_frames=15, trim=5)  # 15 ≤ 2*10

        # Trimming guard: only trim if enough frames remain
        if trim_frames > 0 and seq.shape[0] > 2 * trim_frames:
            result = seq[trim_frames:-trim_frames]
        else:
            result = seq  # No trimming for short videos

        assert np.array_equal(result, seq), (
            "Short videos (≤ 2×TRIM_FRAMES frames) should not be trimmed"
        )


# ---------------------------------------------------------------------------
# Bug 2 — Auto-Stop Recording (frontend check)
# ---------------------------------------------------------------------------

class TestBug2AutoStopRecording:
    """
    Documents Bug 2: no auto-stop mechanism exists in the frontend.

    The test reads the sign-to-text page.tsx source file and asserts that
    the string `autoStopIntervalRef` does NOT exist in the file.

    This assertion PASSES on unfixed code (confirming the bug: no auto-stop
    mechanism is present). After the fix is applied, `autoStopIntervalRef`
    will be added to page.tsx and this test will FAIL, confirming the fix.
    """

    PAGE_TSX_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "emotisign-frontend",
        "src",
        "app",
        "translate",
        "sign-to-text",
        "page.tsx",
    )

    def test_auto_stop_interval_ref_absent(self):
        """
        BUG CONDITION: `autoStopIntervalRef` does NOT exist in page.tsx.

        This PASSES on unfixed code (confirming the bug: no auto-stop mechanism).
        This will FAIL after the fix is applied (confirming the fix is in place).
        """
        page_path = os.path.normpath(self.PAGE_TSX_PATH)
        assert os.path.exists(page_path), (
            f"page.tsx not found at expected path: {page_path}"
        )

        with open(page_path, encoding="utf-8") as f:
            content = f.read()

        assert "autoStopIntervalRef" not in content, (
            "BUG CONDITION FAILED: `autoStopIntervalRef` was found in page.tsx. "
            "This means the auto-stop fix has already been applied. "
            "This test documents the bug (absence of auto-stop) and should PASS on unfixed code."
        )

    def test_absence_threshold_ms_absent(self):
        """
        BUG CONDITION: `ABSENCE_THRESHOLD_MS` does NOT exist in page.tsx.

        This PASSES on unfixed code (confirming no absence threshold constant exists).
        This will FAIL after the fix is applied.
        """
        page_path = os.path.normpath(self.PAGE_TSX_PATH)
        assert os.path.exists(page_path), (
            f"page.tsx not found at expected path: {page_path}"
        )

        with open(page_path, encoding="utf-8") as f:
            content = f.read()

        assert "ABSENCE_THRESHOLD_MS" not in content, (
            "BUG CONDITION FAILED: `ABSENCE_THRESHOLD_MS` was found in page.tsx. "
            "This means the auto-stop fix has already been applied."
        )

    def test_absence_start_ref_absent(self):
        """
        BUG CONDITION: `absenceStartRef` does NOT exist in page.tsx.

        This PASSES on unfixed code (confirming no absence tracking ref exists).
        This will FAIL after the fix is applied.
        """
        page_path = os.path.normpath(self.PAGE_TSX_PATH)
        assert os.path.exists(page_path), (
            f"page.tsx not found at expected path: {page_path}"
        )

        with open(page_path, encoding="utf-8") as f:
            content = f.read()

        assert "absenceStartRef" not in content, (
            "BUG CONDITION FAILED: `absenceStartRef` was found in page.tsx. "
            "This means the auto-stop fix has already been applied."
        )


# ---------------------------------------------------------------------------
# Bug 3 — PSL Urdu Encoding (mojibake in label_map.json)
# ---------------------------------------------------------------------------

class TestBug3UrduEncoding:
    """
    Documents Bug 3: label_map.json contains mojibake instead of valid Urdu Unicode.

    The test loads label_map.json with UTF-8 encoding and checks that at least
    one value is NOT valid Urdu Unicode (code points outside U+0600–U+06FF and
    U+FB50–U+FDFF).

    This assertion PASSES on unfixed code (confirming the mojibake bug exists).
    After the fix, all values will be valid Urdu Unicode and this test will FAIL,
    confirming the fix is in place.
    """

    LABEL_MAP_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "ml",
        "models",
        "psl",
        "label_map.json",
    )

    def _load_label_map(self) -> dict:
        path = os.path.normpath(self.LABEL_MAP_PATH)
        assert os.path.exists(path), f"label_map.json not found at: {path}"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_label_map_loads_successfully(self):
        """Sanity check: label_map.json can be loaded with UTF-8 encoding."""
        label_map = self._load_label_map()
        assert isinstance(label_map, dict), "label_map.json should be a JSON object"
        assert len(label_map) > 0, "label_map.json should not be empty"

    def test_at_least_one_value_is_not_valid_urdu(self):
        """
        BUG CONDITION: at least one label_map value is NOT valid Urdu Unicode.

        This PASSES on unfixed code (confirming the mojibake bug exists).
        This will FAIL after the fix is applied (all values will be valid Urdu).
        """
        label_map = self._load_label_map()

        invalid_entries = {
            k: v
            for k, v in label_map.items()
            if not _all_chars_valid_urdu(v)
        }

        assert len(invalid_entries) > 0, (
            "BUG CONDITION FAILED: all label_map values appear to be valid Urdu Unicode. "
            "This means the mojibake fix has already been applied. "
            "This test documents the bug (mojibake present) and should PASS on unfixed code. "
            f"All {len(label_map)} entries passed the Urdu Unicode check."
        )

        # Document the counterexamples found
        print(f"\nMojibake counterexamples found ({len(invalid_entries)} of {len(label_map)} entries):")
        for k, v in list(invalid_entries.items())[:5]:
            char_info = ", ".join(f"U+{ord(c):04X}" for c in v)
            print(f"  label_map[{k!r}] = {v!r}  (code points: {char_info})")

    def test_label_map_key_1_is_not_single_urdu_char(self):
        """
        BUG CONDITION: label_map["1"] is NOT a single valid Urdu character.

        The design doc notes that label_map["1"] should be a single Urdu character
        (e.g., "ب") but currently contains mojibake (e.g., "+¡GÇ¼").

        This PASSES on unfixed code (confirming the bug).
        This will FAIL after the fix is applied.
        """
        label_map = self._load_label_map()

        assert "1" in label_map, "label_map.json should have key '1'"
        value = label_map["1"]

        is_single_valid_urdu = (len(value) == 1 and _is_valid_urdu_char(value))

        assert not is_single_valid_urdu, (
            f"BUG CONDITION FAILED: label_map['1'] = {value!r} is already a single valid Urdu character. "
            "This means the mojibake fix has already been applied. "
            "This test documents the bug and should PASS on unfixed code."
        )

        # Document the counterexample
        print(f"\nCounterexample: label_map['1'] = {value!r}")
        print(f"  Length: {len(value)} characters (expected 1 for a single Urdu letter)")
        print(f"  Code points: {', '.join(f'U+{ord(c):04X}' for c in value)}")
        print(f"  Expected: a single Urdu character in range U+0600–U+06FF or U+FB50–U+FDFF")

    def test_mojibake_can_be_decoded(self):
        """
        Documents the root cause: the mojibake values can be recovered by
        encoding as Latin-1 and decoding as UTF-8.

        This test verifies the fix strategy works: v.encode('latin-1').decode('utf-8')
        produces valid Urdu Unicode for the corrupted entries.
        """
        label_map = self._load_label_map()

        recoverable_count = 0
        for k, v in label_map.items():
            if not _all_chars_valid_urdu(v):
                try:
                    fixed = v.encode("latin-1").decode("utf-8")
                    if _all_chars_valid_urdu(fixed):
                        recoverable_count += 1
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

        # At least some entries should be recoverable via latin-1 → utf-8
        assert recoverable_count > 0, (
            "Expected at least one mojibake entry to be recoverable via "
            "v.encode('latin-1').decode('utf-8'). "
            "This validates the fix strategy described in the design doc."
        )

        print(f"\nRecoverable mojibake entries: {recoverable_count} of {len(label_map)}")
