"""
Translation Router — /api/translate
Endpoints:
  POST /text-to-sign        → convert text → sign data
  POST /sign-to-text        → upload video → recognized text
  GET  /history             → paginated translation history
  GET  /history/{id}        → single translation detail
  DELETE /history/{id}      → delete a translation
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user, get_optional_user
from app.models import TranslationHistory, TranslationMode, SignLanguage, Emotion, User
from app.schemas import (
    TextToSignRequest, TextToSignResponse,
    SignToTextResponse, TranslationHistoryItem, PaginatedTranslations,
    SuccessResponse
)
from app.ml.ml_service import text_to_sign, sign_to_text, analyze_sentiment, detect_emotion_from_video

router = APIRouter(prefix="/api/translate", tags=["Translation"])

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/avi", "video/quicktime"}
MAX_VIDEO_BYTES = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024


# ─────────────────────────── Text → Sign ───────────────────────────

@router.post("/text-to-sign", response_model=TextToSignResponse)
async def translate_text_to_sign(
    payload: TextToSignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Translate input text into sign language visual representation.
    Works for both authenticated users and guests.
    Emotion/sentiment analysis is performed on the input text.
    """
    start = time.time()

    # Run ML tasks concurrently
    import asyncio
    sign_task = asyncio.create_task(text_to_sign(payload.text, payload.sign_language.value))
    sentiment_task = asyncio.create_task(analyze_sentiment(payload.text)) if payload.include_emotion else None

    sign_result = await sign_task
    sentiment_result = await sentiment_task if sentiment_task else None

    processing_time = int((time.time() - start) * 1000)

    # Save to DB
    emotion_val = Emotion.UNKNOWN
    sentiment_score = None
    sentiment_label = None

    if sentiment_result:
        emotion_val = Emotion(sentiment_result["emotion"]) if sentiment_result["emotion"] in Emotion._value2member_map_ else Emotion.UNKNOWN
        sentiment_score = sentiment_result["sentiment_score"]
        sentiment_label = sentiment_result["sentiment_label"]

    history = TranslationHistory(
        user_id=current_user.id if current_user else None,
        mode=TranslationMode.TEXT_TO_SIGN,
        sign_language=payload.sign_language,
        input_text=payload.text,
        output_sign_data=str(sign_result.get("signs")),
        detected_emotion=emotion_val,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        processing_time_ms=processing_time,
        is_guest=current_user is None
    )
    db.add(history)
    await db.commit()
    await db.refresh(history)

    emotion_data = None
    if sentiment_result:
        from app.schemas import EmotionData
        emotion_data = EmotionData(
            emotion=sentiment_result["emotion"],
            sentiment_label=sentiment_result["sentiment_label"],
            sentiment_score=sentiment_result["sentiment_score"],
            emotion_scores=sentiment_result["emotion_scores"]
        )

    return TextToSignResponse(
        translation_id=history.id,
        input_text=payload.text,
        words=sign_result["words"],
        signs=sign_result["signs"],
        total_duration_ms=sign_result["total_duration_ms"],
        sign_language=sign_result["sign_language"],
        fingerspelled_words=sign_result["fingerspelled_words"],
        emotion_analysis=emotion_data,
        processing_time_ms=processing_time
    )


# ─────────────────────────── Sign → Text ───────────────────────────

@router.post("/sign-to-text", response_model=SignToTextResponse)
async def translate_sign_to_text(
    sign_language: str = Form("ASL"),
    detect_emotion: bool = Form(True),
    video: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Upload a sign language video and receive text translation.
    Also performs facial emotion detection from the video.
    Accepts: mp4, webm, avi, mov
    """
    start = time.time()

    # Validate content type
    if video.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )

    # Read and validate size
    content = await video.read()
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Max size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )

    # Save video
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(video.filename).suffix or ".mp4"
    filename = f"{uuid.uuid4()}{ext}"
    video_path = upload_dir / filename

    async with aiofiles.open(video_path, "wb") as f:
        await f.write(content)

    # Run ML tasks
    import asyncio
    sign_lang_enum = SignLanguage.ASL if sign_language.upper() == "ASL" else SignLanguage.PSL

    sign_task = asyncio.create_task(sign_to_text(str(video_path), sign_language))
    emotion_task = asyncio.create_task(detect_emotion_from_video(str(video_path))) if detect_emotion else None

    sign_result = await sign_task
    emotion_result = await emotion_task if emotion_task else None

    processing_time = int((time.time() - start) * 1000)

    emotion_val = Emotion.UNKNOWN
    if emotion_result:
        dom = emotion_result.get("dominant_emotion", "unknown")
        emotion_val = Emotion(dom) if dom in Emotion._value2member_map_ else Emotion.UNKNOWN

    # Save to DB
    history = TranslationHistory(
        user_id=current_user.id if current_user else None,
        mode=TranslationMode.SIGN_TO_TEXT,
        sign_language=sign_lang_enum,
        input_video_path=str(video_path),
        output_text=sign_result["recognized_text"],
        detected_emotion=emotion_val,
        confidence_score=sign_result["confidence"],
        processing_time_ms=processing_time,
        is_guest=current_user is None
    )
    db.add(history)
    await db.commit()
    await db.refresh(history)

    emotion_data = None
    if emotion_result:
        from app.schemas import EmotionData
        emotion_data = EmotionData(
            emotion=emotion_result["dominant_emotion"],
            sentiment_label="visual",
            sentiment_score=emotion_result["emotion_scores"].get(emotion_result["dominant_emotion"], 0.0),
            emotion_scores=emotion_result["emotion_scores"]
        )

    return SignToTextResponse(
        translation_id=history.id,
        recognized_text=sign_result["recognized_text"],
        glosses=sign_result["glosses"],
        confidence=sign_result["confidence"],
        sign_language=sign_result["sign_language"],
        frame_count=sign_result["frame_count"],
        processing_time_ms=processing_time,
        emotion_from_video=emotion_data
    )


# ─────────────────────────── History ───────────────────────────

@router.get("/history", response_model=PaginatedTranslations)
async def get_translation_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = select(TranslationHistory).where(TranslationHistory.user_id == current_user.id)
    if mode:
        try:
            mode_enum = TranslationMode(mode)
            query = query.where(TranslationHistory.mode == mode_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    query = query.order_by(TranslationHistory.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = result.scalars().all()

    return PaginatedTranslations(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size
    )


@router.get("/history/{translation_id}", response_model=TranslationHistoryItem)
async def get_translation_detail(
    translation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(TranslationHistory).where(
            TranslationHistory.id == translation_id,
            TranslationHistory.user_id == current_user.id
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Translation not found")
    return item


@router.delete("/history/{translation_id}", response_model=SuccessResponse)
async def delete_translation(
    translation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(TranslationHistory).where(
            TranslationHistory.id == translation_id,
            TranslationHistory.user_id == current_user.id
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Translation not found")

    await db.delete(item)
    await db.commit()
    return SuccessResponse(message="Translation deleted successfully")
