"""
EmotiSign ML Service

Delegates text_to_sign() to SignGenerator (real keypoint-based implementation).
Other functions (sign_to_text, analyze_sentiment, detect_emotion_from_video)
remain as placeholders ready for future model integration.
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
#  SIGN → TEXT  (video input)
# ──────────────────────────────────────────────────────────────

async def sign_to_text(video_path: str, sign_language: str = "ASL") -> dict:
    """
    Recognize sign language from an uploaded video file and convert to text.

    PLACEHOLDER — returns a mock transcription.

    REAL IMPLEMENTATION:
      1. Load video frames (OpenCV)
      2. Extract hand/body keypoints per frame (MediaPipe Holistic)
      3. Normalize & window keypoint sequences
      4. Pass through trained gesture classification model:
         - Option A: LSTM / BiLSTM on keypoint sequences
         - Option B: Transformer (like SignBERT or SPOTER)
         - Option C: CNN on pose heatmaps
      5. Post-process predicted sign glosses → English/Urdu text
         (sign language gloss order ≠ spoken language word order)
      6. Return text + confidence + detected signs list

    Returns:
      {
        "recognized_text": str,
        "glosses": [...],        # raw sign glosses before translation
        "confidence": float,
        "sign_language": str,
        "frame_count": int,
        "processing_time_ms": int
      }
    """
    start = time.time()
    await asyncio.sleep(0.2)  # Simulate processing

    mock_responses = [
        "Hello, how are you?",
        "Thank you very much.",
        "Please help me.",
        "I am happy to meet you.",
        "Good morning, nice to see you.",
        "Can you understand me?",
        "I love sign language.",
    ]

    path = Path(video_path)
    frame_count = random.randint(30, 300)  # Mock

    processing_time = int((time.time() - start) * 1000 + frame_count * 2)

    return {
        "recognized_text": random.choice(mock_responses),
        "glosses": ["HELLO", "HOW", "YOU"],  # Mock ASL glosses
        "confidence": round(random.uniform(0.72, 0.97), 3),
        "sign_language": sign_language,
        "frame_count": frame_count,
        "video_path": str(path),
        "processing_time_ms": processing_time,
        "note": "PLACEHOLDER — integrate real CV/pose estimation model here"
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
    Detect emotion from signer's facial expressions in the video.

    PLACEHOLDER — returns mock emotion.

    REAL IMPLEMENTATION:
      1. Sample frames from the video (every N frames)
      2. Detect face in each frame (MTCNN or OpenCV Haar cascades)
      3. Crop face region
      4. Run emotion classifier on face crop:
         - Option A: DeepFace library (wraps multiple models)
         - Option B: FER (Facial Expression Recognition) library
         - Option C: Custom CNN trained on AffectNet / RAF-DB
      5. Aggregate per-frame predictions (majority vote or average)
      6. Return dominant emotion + per-frame breakdown

    Returns:
      {
        "dominant_emotion": str,
        "emotion_scores": {emotion: avg_prob, ...},
        "frame_samples": int,
        "processing_time_ms": int
      }
    """
    start = time.time()
    await asyncio.sleep(0.15)

    dominant = random.choice(EMOTIONS)
    emotion_scores = {e: round(random.uniform(0.01, 0.15), 3) for e in EMOTIONS}
    emotion_scores[dominant] = round(random.uniform(0.45, 0.88), 3)

    return {
        "dominant_emotion": dominant,
        "emotion_scores": emotion_scores,
        "frame_samples": random.randint(5, 30),
        "processing_time_ms": int((time.time() - start) * 1000),
        "note": "PLACEHOLDER — integrate DeepFace/FER model here"
    }
