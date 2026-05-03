"""
Real-Time Translation Router
=============================
WebSocket endpoints for live webcam sign language recognition
and real-time text-to-sign streaming.

Endpoints:
  WS /ws/translate/sign-to-text   — stream webcam frames → receive live text + emotion
  WS /ws/translate/text-to-sign   — send text → receive sign frames streamed back

Frame Protocol (sign-to-text):
  Client → Server:
    { "type": "frame",  "data": "<base64 image>", "mime": "image/jpeg" }
    { "type": "config", "sign_language": "ASL", "detect_emotion": true }
    { "type": "stop" }
    { "type": "ping" }

  Server → Client:
    { "event": "result",    "data": { recognized_text, glosses, confidence, emotion, ... } }
    { "event": "partial",   "data": { recognized_text, confidence, is_partial: true } }
    { "event": "status",    "data": { "message": "...", "frame_count": N } }
    { "event": "error",     "data": { "detail": "..." } }
    { "event": "pong",      "data": {} }
    { "event": "stopped",   "data": { "total_frames": N, "session_duration_s": N } }

Text-to-Sign Protocol:
  Client → Server:
    { "type": "translate", "text": "Hello world", "sign_language": "ASL" }
    { "type": "ping" }

  Server → Client:
    { "event": "sign_start",  "data": { "word_count": N, "total_signs": N } }
    { "event": "sign_frame",  "data": { "word": "hello", "frame_index": 0, "frame": "h", "gif_url": "..." } }
    { "event": "sign_done",   "data": { "total_signs": N, "duration_ms": N } }
    { "event": "emotion",     "data": { sentiment, emotion, scores } }
    { "event": "error",       "data": { "detail": "..." } }
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from jose import JWTError

from app.database import AsyncSessionLocal
from app.models import TranslationHistory, TranslationMode, SignLanguage, Emotion, User
from app.services.auth_service import decode_token
from app.services.frame_processor import FrameBuffer, process_frame_buffer, decode_base64_frame
from app.ml.ml_service import text_to_sign, analyze_sentiment

router = APIRouter(prefix="/ws/translate", tags=["Real-Time Translation"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _event(name: str, data: dict) -> dict:
    return {"event": name, "data": data, "timestamp": _now()}


async def _get_user_from_token(token: Optional[str]) -> Optional[User]:
    """Resolve optional JWT to User (guests allowed)."""
    if not token:
        return None
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        async with AsyncSessionLocal() as db:
            return await db.get(User, int(user_id))
    except JWTError:
        return None


async def _save_translation(
    user_id: Optional[int],
    mode: TranslationMode,
    sign_language: SignLanguage,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    emotion: str = "unknown",
    confidence: float = 0.0,
    processing_time_ms: int = 0,
):
    """Persist a completed real-time translation session to DB."""
    try:
        emotion_val = Emotion(emotion) if emotion in Emotion._value2member_map_ else Emotion.UNKNOWN
        async with AsyncSessionLocal() as db:
            record = TranslationHistory(
                user_id=user_id,
                mode=mode,
                sign_language=sign_language,
                input_text=input_text,
                output_text=output_text,
                detected_emotion=emotion_val,
                confidence_score=confidence,
                processing_time_ms=processing_time_ms,
                is_guest=(user_id is None),
            )
            db.add(record)
            await db.commit()
    except Exception as e:
        print(f"[DB] Failed to save translation: {e}")


# ── WebSocket: Sign-to-Text (Live Webcam) ────────────────────────────────────

@router.websocket("/sign-to-text")
async def ws_sign_to_text(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    """
    Real-time sign language recognition from webcam frames.

    Connect: ws://localhost:8000/ws/translate/sign-to-text?token=<JWT>
    Token is optional — guests can use this endpoint too.

    Send frames as fast as your webcam captures them (30fps is fine).
    The backend processes every 10th frame to avoid overload.
    Partial results are sent immediately; confident results replace them.
    """
    await websocket.accept()

    user = await _get_user_from_token(token)
    user_id = user.id if user else None

    # Session state
    buffer = FrameBuffer()
    sign_language = SignLanguage.ASL
    detect_emotion = True
    session_start = time.time()
    last_recognized_text = ""
    last_emotion = "unknown"
    last_confidence = 0.0
    is_running = True

    await websocket.send_json(_event("status", {
        "message": "Connected. Start sending frames.",
        "user": user.username if user else "guest",
        "frame_buffer_size": 30,
        "processes_every_n_frames": 10,
    }))

    try:
        while is_running:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json(_event("status", {"message": "Waiting for frames..."}))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(_event("error", {"detail": "Invalid JSON"}))
                continue

            msg_type = msg.get("type", "")

            # ── PING ──
            if msg_type == "ping":
                await websocket.send_json(_event("pong", {}))
                continue

            # ── CONFIG ──
            if msg_type == "config":
                sl = msg.get("sign_language", "ASL").upper()
                sign_language = SignLanguage.PSL if sl == "PSL" else SignLanguage.ASL
                detect_emotion = msg.get("detect_emotion", True)
                await websocket.send_json(_event("status", {
                    "message": f"Config updated: {sign_language.value}, emotion={detect_emotion}"
                }))
                continue

            # ── STOP ──
            if msg_type == "stop":
                is_running = False
                duration = round(time.time() - session_start, 2)
                await websocket.send_json(_event("stopped", {
                    "total_frames": buffer.total_received,
                    "session_duration_s": duration,
                    "last_recognized_text": last_recognized_text,
                }))
                # Save session to DB
                await _save_translation(
                    user_id=user_id,
                    mode=TranslationMode.SIGN_TO_TEXT,
                    sign_language=sign_language,
                    output_text=last_recognized_text,
                    emotion=last_emotion,
                    confidence=last_confidence,
                    processing_time_ms=int((time.time() - session_start) * 1000),
                )
                break

            # ── FRAME ──
            if msg_type == "frame":
                frame_data = msg.get("data", "")
                if not frame_data:
                    continue

                # Decode and validate frame
                decoded = await decode_base64_frame(frame_data)
                if decoded is None:
                    await websocket.send_json(_event("error", {"detail": "Invalid frame encoding"}))
                    continue

                buffer.add_frame(frame_data)

                # Send frame acknowledgement every 10 frames
                if buffer.total_received % 10 == 0:
                    await websocket.send_json(_event("status", {
                        "message": "Receiving frames...",
                        "frame_count": buffer.total_received,
                        "buffer_size": len(buffer.frames),
                    }))

                # ── Run inference when buffer is ready ──
                if buffer.should_process():
                    result = await process_frame_buffer(buffer)

                    last_recognized_text = result.recognized_text
                    last_emotion = result.emotion
                    last_confidence = result.confidence

                    if result.is_partial:
                        # Low confidence — send as partial
                        await websocket.send_json(_event("partial", {
                            "recognized_text": result.recognized_text,
                            "glosses": result.glosses,
                            "confidence": result.confidence,
                            "is_partial": True,
                            "frame_count": result.frame_count,
                        }))
                    else:
                        # High confidence — send as final result
                        payload = {
                            "recognized_text": result.recognized_text,
                            "glosses": result.glosses,
                            "confidence": result.confidence,
                            "is_partial": False,
                            "frame_count": result.frame_count,
                            "processing_time_ms": result.processing_time_ms,
                        }
                        if detect_emotion:
                            payload["emotion"] = result.emotion
                            payload["emotion_scores"] = result.emotion_scores

                        await websocket.send_json(_event("result", payload))
                continue

            # Unknown type
            await websocket.send_json(_event("error", {"detail": f"Unknown type: {msg_type}"}))

    except WebSocketDisconnect:
        # Auto-save on disconnect if we have results
        if last_recognized_text:
            await _save_translation(
                user_id=user_id,
                mode=TranslationMode.SIGN_TO_TEXT,
                sign_language=sign_language,
                output_text=last_recognized_text,
                emotion=last_emotion,
                confidence=last_confidence,
                processing_time_ms=int((time.time() - session_start) * 1000),
            )


# ── WebSocket: Text-to-Sign (Streamed Sign Frames) ───────────────────────────

@router.websocket("/text-to-sign")
async def ws_text_to_sign(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    """
    Real-time text-to-sign translation with streamed sign frames.

    Connect: ws://localhost:8000/ws/translate/text-to-sign?token=<JWT>
    Token is optional — guests allowed.

    Instead of getting the entire sign list at once (like the REST endpoint),
    sign frames are streamed word-by-word so the frontend can animate them
    in real-time as they arrive — like a live interpreter.

    The client can send multiple translate messages in one session
    (e.g. for a live conversation where the hearing user keeps typing).
    """
    await websocket.accept()

    user = await _get_user_from_token(token)
    user_id = user.id if user else None

    await websocket.send_json(_event("status", {
        "message": "Connected. Send { type: 'translate', text: '...' } to begin.",
        "user": user.username if user else "guest",
    }))

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json(_event("status", {"message": "Idle. Waiting for input..."}))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(_event("error", {"detail": "Invalid JSON"}))
                continue

            msg_type = msg.get("type", "")

            # ── PING ──
            if msg_type == "ping":
                await websocket.send_json(_event("pong", {}))
                continue

            # ── TRANSLATE ──
            if msg_type == "translate":
                text = msg.get("text", "").strip()
                if not text:
                    await websocket.send_json(_event("error", {"detail": "text cannot be empty"}))
                    continue

                sl_str = msg.get("sign_language", "ASL").upper()
                sign_language = SignLanguage.PSL if sl_str == "PSL" else SignLanguage.ASL
                include_emotion = msg.get("include_emotion", True)

                session_start = time.time()

                # Run sentiment in parallel while we prepare sign data
                sentiment_task = asyncio.create_task(analyze_sentiment(text)) if include_emotion else None
                sign_result = await text_to_sign(text, sign_language.value)

                signs = sign_result.get("signs", [])
                total_signs = sum(len(s.get("frames", [])) for s in signs)

                # Tell client how many signs are coming
                await websocket.send_json(_event("sign_start", {
                    "text": text,
                    "word_count": len(signs),
                    "total_frames": total_signs,
                    "sign_language": sign_language.value,
                }))

                # Stream each word's frames one by one
                frame_index = 0
                for sign in signs:
                    word = sign.get("word", "")
                    frames = sign.get("frames", [])
                    gif_url = sign.get("gif_url")
                    fingerspelled = sign.get("fingerspelled", False)

                    for frame_char in frames:
                        await websocket.send_json(_event("sign_frame", {
                            "word": word,
                            "frame": frame_char,
                            "frame_index": frame_index,
                            "gif_url": gif_url,
                            "fingerspelled": fingerspelled,
                        }))
                        frame_index += 1
                        # Small delay between frames so frontend can animate smoothly
                        # In production this can be driven by the frontend's animation speed
                        await asyncio.sleep(0.08)

                processing_time = int((time.time() - session_start) * 1000)

                # Send done event
                await websocket.send_json(_event("sign_done", {
                    "total_signs": frame_index,
                    "fingerspelled_words": sign_result.get("fingerspelled_words", []),
                    "processing_time_ms": processing_time,
                }))

                # Send emotion analysis result
                if sentiment_task:
                    sentiment = await sentiment_task
                    await websocket.send_json(_event("emotion", {
                        "emotion": sentiment["emotion"],
                        "sentiment_label": sentiment["sentiment_label"],
                        "sentiment_score": sentiment["sentiment_score"],
                        "emotion_scores": sentiment["emotion_scores"],
                    }))

                # Save to DB
                await _save_translation(
                    user_id=user_id,
                    mode=TranslationMode.TEXT_TO_SIGN,
                    sign_language=sign_language,
                    input_text=text,
                    processing_time_ms=processing_time,
                )
                continue

            # Unknown type
            await websocket.send_json(_event("error", {"detail": f"Unknown type: {msg_type}"}))

    except WebSocketDisconnect:
        pass
