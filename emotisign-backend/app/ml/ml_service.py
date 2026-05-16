"""
EmotiSign ML Service

Delegates text_to_sign() to SignGenerator (real keypoint-based implementation).
Delegates sign_to_text() to WLASLModelService (real TCN+BiGRU model).
Other functions (analyze_sentiment, detect_emotion_from_video) remain as
placeholders ready for future model integration.
"""

import asyncio
import random
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.sign_generator import SignGenerator


# ──────────────────────────────────────────────────────────────
#  EMOTION LABELS
# ──────────────────────────────────────────────────────────────

EMOTIONS = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]


# ──────────────────────────────────────────────────────────────
#  TEXT → SIGN  (delegates to SignGenerator)
# ──────────────────────────────────────────────────────────────

# Module-level singleton — set by main.py lifespan via set_sign_generator()
_sign_generator: Optional["SignGenerator"] = None


def set_sign_generator(sg: "SignGenerator") -> None:
    """Register the SignGenerator instance. Called once during app startup."""
    global _sign_generator
    _sign_generator = sg


async def text_to_sign(text: str, sign_language: str = "ASL") -> dict:
    """
    Convert input text to sign language keypoint representation.

    Delegates to SignGenerator which loads pre-recorded ASL keypoint data
    from .npy files and returns raw (N_frames, 75, 2) arrays for canvas rendering.

    Falls back to an empty result if SignGenerator is not initialized.
    """
    if _sign_generator is None:
        # SignGenerator not yet initialized (e.g. startup failure)
        return {
            "words": [],
            "signs": [],
            "total_duration_ms": 0,
            "sign_language": sign_language,
            "fingerspelled_words": [],
        }
    return await _sign_generator.text_to_sign(text, sign_language)


# ──────────────────────────────────────────────────────────────
#  SIGN → TEXT  (video input — delegates to WLASLModelService)
# ──────────────────────────────────────────────────────────────

async def sign_to_text(video_path: str, sign_language: str = "ASL") -> dict:
    """
    Recognize sign language from an uploaded video file and convert to text.

    Delegates to WLASLModelService which runs:
      1. Frame-by-frame MediaPipe keypoint extraction (in thread executor)
      2. Nearest-neighbor sequence resampling to 64 frames
      3. TCN + BiGRU inference with TTA (in thread executor)
      4. Temperature-scaled confidence scores

    Falls back to a safe error response if the WLASL service is unavailable.

    Returns:
      {
        "recognized_text": str,
        "glosses": [...],
        "confidence": float,
        "sign_language": str,
        "frame_count": int,
        "processing_time_ms": int
      }
    """
    try:
        from app.ml.wlasl_service import get_wlasl_service
        wlasl_service = await get_wlasl_service()
        return await wlasl_service.sign_to_text(video_path, use_tta=True)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"sign_to_text failed: {e}")
        return {
            "recognized_text": "",
            "glosses": [],
            "confidence": 0.0,
            "sign_language": sign_language,
            "frame_count": 0,
            "video_path": str(video_path),
            "processing_time_ms": 0,
            "error": str(e),
        }


# ──────────────────────────────────────────────────────────────
#  SENTIMENT ANALYSIS (text / speech input)
# ──────────────────────────────────────────────────────────────

async def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment and emotion from text input (for hearing users).

    PLACEHOLDER — returns random-ish sentiment based on keywords.

    REAL IMPLEMENTATION:
      Option A: Rule-based — VADER (good for short social-style text)
      Option B: Transformer — cardiffnlp/twitter-roberta-base-sentiment
                              or distilbert-base-uncased-finetuned-sst-2-english
      Option C: Multi-label emotion: SamLowe/roberta-base-go_emotions

    Returns:
      {
        "sentiment_label": "positive" | "neutral" | "negative",
        "sentiment_score": float,   # -1.0 to +1.0
        "emotion": str,             # dominant detected emotion
        "emotion_scores": {emotion: prob, ...},
        "processing_time_ms": int
      }
    """
    start = time.time()
    await asyncio.sleep(0.02)

    text_lower = text.lower()
    positive_words = {"happy", "good", "great", "love", "thank", "wonderful", "nice", "joy", "excited", "pleased"}
    negative_words = {"sad", "bad", "hate", "angry", "sorry", "terrible", "awful", "hurt", "fear", "disgusted"}

    words = set(text_lower.split())
    pos_hits = len(words & positive_words)
    neg_hits = len(words & negative_words)

    if pos_hits > neg_hits:
        label = "positive"
        score = round(random.uniform(0.3, 0.95), 3)
        emotion = random.choice(["happy", "surprised"])
    elif neg_hits > pos_hits:
        label = "negative"
        score = round(random.uniform(-0.95, -0.3), 3)
        emotion = random.choice(["sad", "angry", "fearful", "disgusted"])
    else:
        label = "neutral"
        score = round(random.uniform(-0.2, 0.2), 3)
        emotion = "neutral"

    emotion_scores = {e: round(random.uniform(0.01, 0.15), 3) for e in EMOTIONS}
    emotion_scores[emotion] = round(random.uniform(0.5, 0.9), 3)

    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "emotion": emotion,
        "emotion_scores": emotion_scores,
        "processing_time_ms": int((time.time() - start) * 1000),
        "note": "PLACEHOLDER — integrate VADER/BERT sentiment model here"
    }


# ──────────────────────────────────────────────────────────────
#  EMOTION DETECTION (from video / facial expressions)
# ──────────────────────────────────────────────────────────────

async def detect_emotion_from_video(video_path: str) -> dict:
    """
    Detect happy / sad / neutral from the signer's face across a video.

    Uses MediaPipe Face Mesh (see face_emotion_service.py).
    """
    from app.ml.face_emotion_service import detect_emotion_from_video_async

    return await detect_emotion_from_video_async(video_path)
