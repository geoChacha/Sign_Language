"""
sign_generator.py — ASL Sign Generation Service

Loads pre-recorded MediaPipe Holistic keypoint sequences from .npy files
and serves them as raw (N_frames, 75, 2) float arrays for canvas rendering.

Architecture:
  - All .npy files are loaded once at application startup via load()
  - Keypoint arrays are cached in process memory (no disk I/O per request)
  - The frontend renders each frame onto an HTML5 Canvas element
  - Words not in the vocabulary fall back to fingerspelling

Landmark layout (75 total):
  Indices  0–32  → MediaPipe Pose (33 body landmarks)
  Indices 33–53  → MediaPipe Left Hand (21 landmarks)
  Indices 54–74  → MediaPipe Right Hand (21 landmarks)
"""

import logging
import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default path — relative to the backend working directory (emotisign-backend/)
# The keypoints folder lives one level up at the workspace root
_DEFAULT_KEYPOINTS_DIR = "../Featrure_sign_generation/keypoints_best"

# Expected number of landmarks per frame
_EXPECTED_LANDMARKS = 75


class SignGenerator:
    """
    Loads ASL keypoint .npy files at startup and serves them on demand.

    Thread-safe for read access; the cache is populated once during load()
    and never mutated afterwards.
    """

    def __init__(self) -> None:
        # word → list of frames, each frame is a list of 75 [x, y] pairs
        self._cache: Dict[str, List[List[List[float]]]] = {}
        self._vocabulary: set = set()
        self._frame_rate_ms: int = self._read_frame_rate()

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def load(self, keypoints_dir: Optional[str] = None) -> None:
        """
        Load all .npy files from keypoints_dir into the in-memory cache.
        Called once during application lifespan startup.

        Args:
            keypoints_dir: Override path (used in tests). If None, reads
                           KEYPOINTS_DIR env var, falling back to the default.
        """
        resolved_dir = self._resolve_dir(keypoints_dir)
        logger.info(f"SignGenerator: loading from {resolved_dir.resolve()}")

        if not resolved_dir.exists():
            logger.error(
                f"SignGenerator: keypoints directory not found: {resolved_dir.resolve()}. "
                "All words will be fingerspelled."
            )
            print(f"❌ SignGenerator: keypoints dir not found: {resolved_dir.resolve()}")
            return

        npy_files = list(resolved_dir.glob("*.npy"))
        if not npy_files:
            logger.error(
                f"SignGenerator: no .npy files found in {resolved_dir}. "
                "All words will be fingerspelled."
            )
            return

        loaded = 0
        for npy_path in npy_files:
            word = npy_path.stem.lower()
            try:
                raw = np.load(str(npy_path), allow_pickle=True)

                # Validate shape: must be (N, 75, D) where D >= 2
                if raw.ndim != 3:
                    logger.warning(
                        f"SignGenerator: skipping '{word}' — unexpected ndim "
                        f"{raw.ndim} (expected 3), shape={raw.shape}"
                    )
                    continue

                n_frames, n_landmarks, n_dims = raw.shape
                if n_landmarks != _EXPECTED_LANDMARKS:
                    logger.warning(
                        f"SignGenerator: skipping '{word}' — expected "
                        f"{_EXPECTED_LANDMARKS} landmarks, got {n_landmarks}, "
                        f"shape={raw.shape}"
                    )
                    continue

                if n_dims < 2:
                    logger.warning(
                        f"SignGenerator: skipping '{word}' — need at least 2 "
                        f"coordinate dims, got {n_dims}, shape={raw.shape}"
                    )
                    continue

                # Strip Z / visibility channels — keep only X and Y
                xy_only = raw[:, :, :2].astype(np.float32)

                # Convert to plain Python list for JSON serialization
                self._cache[word] = xy_only.tolist()
                self._vocabulary.add(word)
                loaded += 1

            except Exception as exc:
                logger.warning(
                    f"SignGenerator: failed to load '{npy_path.name}': {exc}"
                )

        logger.info(
            f"SignGenerator: loaded {loaded}/{len(npy_files)} signs "
            f"(frame_rate={self._frame_rate_ms}ms)"
        )

    def get_sign(self, word: str) -> Optional[List[List[List[float]]]]:
        """
        Return cached keypoint frames for a vocabulary word.

        Args:
            word: Lowercase word string.
        Returns:
            List of frames, each frame being a list of 75 [x, y] pairs.
            None if word is not in vocabulary.
        """
        return self._cache.get(word)

    async def text_to_sign(
        self, text: str, sign_language: str = "ASL"
    ) -> dict:
        """
        Tokenize text and assemble sign data for the API response.

        Tokenization: split on whitespace, strip non-alpha characters, lowercase.
        Vocabulary hit  → keypoints from cache, frames=[], fingerspelled=False
        Vocabulary miss → fingerspelling chars in frames, keypoints=None, fingerspelled=True

        Returns a dict matching the existing ml_service.text_to_sign() contract.
        """
        tokens = self._tokenize(text)
        signs = []
        fingerspelled_words: List[str] = []
        total_frames = 0

        for token in tokens:
            keypoints = self.get_sign(token)

            if keypoints is not None:
                # Vocabulary word — return raw keypoints for canvas rendering
                signs.append({
                    "word": token,
                    "keypoints": keypoints,
                    "frames": [],
                    "gif_url": None,
                    "fingerspelled": False,
                })
                total_frames += len(keypoints)
            else:
                # Out-of-vocabulary — fingerspell character by character
                signs.append({
                    "word": token,
                    "keypoints": None,
                    "frames": list(token),
                    "gif_url": None,
                    "fingerspelled": True,
                })
                fingerspelled_words.append(token)
                total_frames += len(token)

        total_duration_ms = total_frames * self._frame_rate_ms

        return {
            "words": tokens,
            "signs": signs,
            "total_duration_ms": total_duration_ms,
            "sign_language": sign_language,
            "fingerspelled_words": fingerspelled_words,
        }

    @property
    def vocabulary(self) -> FrozenSet[str]:
        """Read-only view of the loaded word set."""
        return frozenset(self._vocabulary)

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Split text into lowercase alphabetic tokens.
        Strips punctuation, digits, and whitespace from each token.
        Returns an empty list if no valid tokens remain.
        """
        tokens = []
        for raw_token in text.split():
            clean = "".join(ch for ch in raw_token if ch.isalpha()).lower()
            if clean:
                tokens.append(clean)
        return tokens

    @staticmethod
    def _resolve_dir(override: Optional[str]) -> Path:
        """Resolve the keypoints directory path, trying multiple fallbacks."""
        # 1. Explicit override (used in tests)
        if override is not None:
            return Path(override)

        # 2. Environment variable (set in Docker via KEYPOINTS_DIR)
        env_val = os.environ.get("KEYPOINTS_DIR")
        if env_val:
            return Path(env_val)

        # 3. Inside the backend directory (Docker build copies it here)
        this_file = Path(__file__).resolve()
        # sign_generator.py → ml → app → emotisign-backend
        backend_root = this_file.parent.parent.parent
        candidate = backend_root / "keypoints_best"
        if candidate.exists():
            return candidate

        # 4. Workspace root sibling (native dev setup)
        workspace_root = backend_root.parent
        candidate2 = workspace_root / "Featrure_sign_generation" / "keypoints_best"
        if candidate2.exists():
            return candidate2

        # 5. Relative to cwd (last resort)
        return Path("../Featrure_sign_generation/keypoints_best")

    @staticmethod
    def _read_frame_rate() -> int:
        """Read SIGN_FRAME_RATE_MS env var, default 50."""
        raw = os.environ.get("SIGN_FRAME_RATE_MS", "50")
        try:
            val = int(raw)
            return val if val > 0 else 50
        except (ValueError, TypeError):
            logger.warning(
                f"SignGenerator: invalid SIGN_FRAME_RATE_MS='{raw}', using 50ms"
            )
            return 50


# ──────────────────────────────────────────────────────────────
#  Module-level singleton (set by main.py lifespan)
# ──────────────────────────────────────────────────────────────

_sign_generator_instance: Optional[SignGenerator] = None


def get_sign_generator() -> Optional[SignGenerator]:
    """Return the global SignGenerator instance (may be None before startup)."""
    return _sign_generator_instance


def set_sign_generator(sg: SignGenerator) -> None:
    """Set the global SignGenerator instance. Called once during lifespan."""
    global _sign_generator_instance
    _sign_generator_instance = sg
