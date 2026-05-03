"""
Speech Router — /api/speech
Endpoints:
  POST /api/speech/speech-to-text   — upload audio file → recognized text
  POST /api/speech/text-to-speech   — send text → receive audio file (mp3/wav)
  WS   /ws/speech/live-stt          — stream mic audio chunks → live transcription

Speech-to-Text REAL INTEGRATION OPTIONS:
  Option A: OpenAI Whisper (local, free)
    pip install openai-whisper
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe("audio.wav")

  Option B: Azure Speech API (cloud)
    pip install azure-cognitiveservices-speech
    See: https://learn.microsoft.com/azure/cognitive-services/speech-service/

  Option C: Google Cloud Speech-to-Text (cloud)
    pip install google-cloud-speech

Text-to-Speech REAL INTEGRATION OPTIONS:
  Option A: gTTS (Google Text-to-Speech, free, simple)
    pip install gTTS
    from gtts import gTTS
    tts = gTTS(text="Hello", lang="en")
    tts.save("output.mp3")

  Option B: Azure TTS (cloud, better quality)
  Option C: pyttsx3 (offline, no API key)
    pip install pyttsx3
"""

import asyncio
import base64
import io
import json
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, AsyncSessionLocal
from app.dependencies import get_current_user, get_optional_user
from app.models import TranslationHistory, TranslationMode, SignLanguage, Emotion, User
from app.ml.ml_service import analyze_sentiment
from app.services.auth_service import decode_token

router = APIRouter(tags=["Speech"])

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
    "audio/webm", "audio/ogg", "audio/x-wav", "audio/mp4"
}
UPLOAD_AUDIO_DIR = "uploads/audio"
OUTPUT_AUDIO_DIR = "uploads/tts_output"


# ── Placeholder ML functions ──────────────────────────────────────────────────

async def _speech_to_text_mock(audio_path: str, language: str = "en") -> dict:
    """
    PLACEHOLDER — Transcribe audio file to text.

    REAL IMPLEMENTATION (Whisper — recommended for FYP):
        import whisper
        model = whisper.load_model("base")  # load once at startup, not per request
        result = model.transcribe(audio_path, language=language)
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"],   # word-level timestamps
            "confidence": 0.95
        }

    REAL IMPLEMENTATION (Azure):
        import azure.cognitiveservices.speech as speechsdk
        config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
        audio_cfg = speechsdk.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio_cfg)
        result = recognizer.recognize_once()
        return {"text": result.text, "confidence": result.confidence}
    """
    await asyncio.sleep(0.1)  # simulate processing

    mock_transcriptions = [
        "Hello, how are you doing today?",
        "Can you please help me find the nearest hospital?",
        "I would like to order some food please.",
        "Thank you very much for your assistance.",
        "Could you repeat that more slowly please?",
        "I need help communicating with someone.",
        "Nice to meet you, my name is Hamza.",
        "What time does the store close today?",
    ]

    return {
        "text": random.choice(mock_transcriptions),
        "language": language,
        "confidence": round(random.uniform(0.82, 0.98), 3),
        "duration_seconds": round(random.uniform(1.5, 8.0), 2),
        "note": "PLACEHOLDER — integrate Whisper or Azure Speech here"
    }


async def _text_to_speech_mock(text: str, language: str = "en", voice: str = "female") -> dict:
    """
    PLACEHOLDER — Convert text to speech audio.

    REAL IMPLEMENTATION (gTTS — simplest, free):
        from gtts import gTTS
        import uuid, os
        filename = f"{uuid.uuid4()}.mp3"
        output_path = f"uploads/tts_output/{filename}"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        return {"audio_path": output_path, "format": "mp3", "duration_seconds": len(text) * 0.07}

    REAL IMPLEMENTATION (pyttsx3 — offline):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, output_path)
        engine.runAndWait()

    REAL IMPLEMENTATION (Azure TTS):
        import azure.cognitiveservices.speech as speechsdk
        config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
        config.speech_synthesis_voice_name = "en-US-JennyNeural"
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config)
        result = synthesizer.speak_text_async(text).get()
        with open(output_path, "wb") as f:
            f.write(result.audio_data)
    """
    await asyncio.sleep(0.08)

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join(OUTPUT_AUDIO_DIR, filename)

    # Write a tiny placeholder WAV (44 bytes header, silent)
    # Real implementation replaces this with actual TTS audio
    silent_wav = bytes([
        0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,  # RIFF header
        0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,  # WAVE fmt
        0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,  # PCM mono
        0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,  # 44100 Hz
        0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61,  # 16-bit
        0x00, 0x00, 0x00, 0x00                             # empty data
    ])
    output_path_wav = output_path.replace(".mp3", ".wav")
    async with aiofiles.open(output_path_wav, "wb") as f:
        await f.write(silent_wav)

    estimated_duration = round(len(text.split()) * 0.4, 2)

    return {
        "audio_path": output_path_wav,
        "audio_url": f"/api/speech/audio/{Path(output_path_wav).name}",
        "format": "wav",
        "duration_seconds": estimated_duration,
        "text_length": len(text),
        "voice": voice,
        "language": language,
        "note": "PLACEHOLDER — integrate gTTS/Azure/pyttsx3 here"
    }


async def _live_stt_chunk_mock(chunk_b64: str, chunk_index: int) -> dict:
    """
    PLACEHOLDER — Process a single audio chunk for live STT.

    REAL IMPLEMENTATION:
      Accumulate chunks into a buffer, run VAD (Voice Activity Detection),
      transcribe complete utterances using Whisper streaming or Azure STT streaming.

      Whisper doesn't natively stream but you can use:
        - faster-whisper with chunked inference
        - Azure Speech SDK streaming recognizer (best for live)
        - Deepgram API (excellent streaming STT)
    """
    await asyncio.sleep(0.05)

    # Only return partial results every few chunks to simulate streaming
    if chunk_index % 5 == 0 and chunk_index > 0:
        mock_partials = [
            "Hello...", "How are...", "Can you help...",
            "Thank you...", "I need...", "Nice to...",
        ]
        return {
            "type": "partial",
            "text": random.choice(mock_partials),
            "chunk_index": chunk_index,
            "is_final": False,
        }
    elif chunk_index % 15 == 0 and chunk_index > 0:
        mock_finals = [
            "Hello, how are you?",
            "Can you help me please?",
            "Thank you very much.",
            "I need assistance.",
        ]
        return {
            "type": "final",
            "text": random.choice(mock_finals),
            "chunk_index": chunk_index,
            "confidence": round(random.uniform(0.85, 0.97), 3),
            "is_final": True,
        }
    return {"type": "buffering", "chunk_index": chunk_index, "is_final": False}


# ── REST: Speech-to-Text ──────────────────────────────────────────────────────

@router.post("/api/speech/speech-to-text")
async def speech_to_text(
    language: str = Form("en"),
    include_sentiment: bool = Form(True),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Upload an audio file and receive the transcribed text.
    Also runs sentiment/emotion analysis on the transcribed text.

    Accepted formats: mp3, wav, webm, ogg, mp4 audio
    Max size: 50MB
    """
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Allowed: mp3, wav, webm, ogg"
        )

    content = await audio.read()
    if len(content) > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 50MB)")

    # Save audio file
    os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
    ext = Path(audio.filename or "audio.wav").suffix or ".wav"
    filename = f"{uuid.uuid4()}{ext}"
    audio_path = os.path.join(UPLOAD_AUDIO_DIR, filename)
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(content)

    start = time.time()

    # Run STT + sentiment concurrently
    stt_task = asyncio.create_task(_speech_to_text_mock(audio_path, language))
    stt_result = await stt_task

    sentiment_result = None
    if include_sentiment and stt_result.get("text"):
        sentiment_result = await analyze_sentiment(stt_result["text"])

    processing_time = int((time.time() - start) * 1000)

    # Save to history
    emotion_val = Emotion.UNKNOWN
    if sentiment_result:
        em = sentiment_result.get("emotion", "unknown")
        emotion_val = Emotion(em) if em in Emotion._value2member_map_ else Emotion.UNKNOWN

    history = TranslationHistory(
        user_id=current_user.id if current_user else None,
        mode=TranslationMode.SIGN_TO_TEXT,
        sign_language=SignLanguage.ASL,
        input_video_path=audio_path,
        output_text=stt_result.get("text"),
        detected_emotion=emotion_val,
        confidence_score=stt_result.get("confidence"),
        processing_time_ms=processing_time,
        is_guest=(current_user is None),
    )
    db.add(history)
    await db.commit()

    response = {
        "translation_id": history.id,
        "recognized_text": stt_result["text"],
        "language": stt_result["language"],
        "confidence": stt_result["confidence"],
        "duration_seconds": stt_result["duration_seconds"],
        "processing_time_ms": processing_time,
    }

    if sentiment_result:
        response["sentiment"] = {
            "emotion": sentiment_result["emotion"],
            "sentiment_label": sentiment_result["sentiment_label"],
            "sentiment_score": sentiment_result["sentiment_score"],
            "emotion_scores": sentiment_result["emotion_scores"],
        }

    return response


# ── REST: Text-to-Speech ──────────────────────────────────────────────────────

@router.post("/api/speech/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form("en"),
    voice: str = Form("female"),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Convert text to speech and receive a downloadable audio file.

    Parameters:
      text     — the text to speak (max 2000 chars)
      language — language code e.g. 'en', 'ur' for Urdu (PSL users)
      voice    — 'male' or 'female'

    Returns audio URL to download/play the generated speech.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 characters)")

    start = time.time()
    tts_result = await _text_to_speech_mock(text, language, voice)
    processing_time = int((time.time() - start) * 1000)

    return {
        "audio_url": tts_result["audio_url"],
        "format": tts_result["format"],
        "duration_seconds": tts_result["duration_seconds"],
        "text_length": tts_result["text_length"],
        "voice": tts_result["voice"],
        "language": tts_result["language"],
        "processing_time_ms": processing_time,
    }


# ── REST: Serve generated audio files ────────────────────────────────────────

@router.get("/api/speech/audio/{filename}")
async def get_audio_file(filename: str):
    """Download/stream a generated TTS audio file."""
    # Sanitize filename — no path traversal
    filename = Path(filename).name
    file_path = os.path.join(OUTPUT_AUDIO_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    ext = Path(filename).suffix.lower()
    media_type = "audio/wav" if ext == ".wav" else "audio/mpeg"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


# ── WebSocket: Live Speech-to-Text ────────────────────────────────────────────

@router.websocket("/ws/speech/live-stt")
async def ws_live_speech_to_text(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
    language: str = Query(default="en"),
):
    """
    Real-time Speech-to-Text via WebSocket.
    Stream microphone audio chunks from the browser → receive live transcription.

    Connect: ws://localhost:8000/ws/speech/live-stt?token=<JWT>&language=en
    Token optional — guests allowed.

    Client → Server:
      { "type": "chunk",  "data": "<base64 audio chunk>", "mime": "audio/webm" }
      { "type": "config", "language": "en", "include_sentiment": true }
      { "type": "stop" }
      { "type": "ping" }

    Server → Client:
      { "event": "partial",   "data": { "text": "Hello...", "is_final": false } }
      { "event": "final",     "data": { "text": "Hello, how are you?", "confidence": 0.95, "is_final": true } }
      { "event": "sentiment", "data": { "emotion": "happy", "sentiment_label": "positive", ... } }
      { "event": "stopped",   "data": { "total_chunks": N, "session_duration_s": N } }
      { "event": "status",    "data": { "message": "..." } }
      { "event": "pong",      "data": {} }
      { "event": "error",     "data": { "detail": "..." } }

    Browser usage:
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorder.ondataavailable = async (e) => {
        const buffer = await e.data.arrayBuffer();
        const b64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
        ws.send(JSON.stringify({ type: 'chunk', data: b64, mime: 'audio/webm' }));
      };
      mediaRecorder.start(250); // send chunk every 250ms
    """
    await websocket.accept()

    # Resolve optional user
    user = None
    if token:
        try:
            payload = decode_token(token)
            uid = payload.get("sub")
            if uid:
                async with AsyncSessionLocal() as db:
                    user = await db.get(User, int(uid))
        except JWTError:
            pass

    chunk_index = 0
    session_start = time.time()
    include_sentiment = True
    last_final_text = ""

    await websocket.send_json({
        "event": "status",
        "data": {
            "message": "Connected. Start streaming audio chunks.",
            "user": user.username if user else "guest",
            "language": language,
        },
        "timestamp": datetime.utcnow().isoformat()
    })

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "event": "status",
                    "data": {"message": "Waiting for audio..."},
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "event": "error",
                    "data": {"detail": "Invalid JSON"},
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue

            msg_type = msg.get("type", "")

            # ── PING ──
            if msg_type == "ping":
                await websocket.send_json({
                    "event": "pong", "data": {},
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue

            # ── CONFIG ──
            if msg_type == "config":
                language = msg.get("language", language)
                include_sentiment = msg.get("include_sentiment", True)
                await websocket.send_json({
                    "event": "status",
                    "data": {"message": f"Config updated: language={language}"},
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue

            # ── STOP ──
            if msg_type == "stop":
                duration = round(time.time() - session_start, 2)

                # Run sentiment on last final transcription
                if last_final_text and include_sentiment:
                    sentiment = await analyze_sentiment(last_final_text)
                    await websocket.send_json({
                        "event": "sentiment",
                        "data": {
                            "text": last_final_text,
                            "emotion": sentiment["emotion"],
                            "sentiment_label": sentiment["sentiment_label"],
                            "sentiment_score": sentiment["sentiment_score"],
                            "emotion_scores": sentiment["emotion_scores"],
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })

                await websocket.send_json({
                    "event": "stopped",
                    "data": {
                        "total_chunks": chunk_index,
                        "session_duration_s": duration,
                        "last_transcription": last_final_text,
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Save to DB
                if last_final_text:
                    async with AsyncSessionLocal() as db:
                        record = TranslationHistory(
                            user_id=user.id if user else None,
                            mode=TranslationMode.SIGN_TO_TEXT,
                            sign_language=SignLanguage.ASL,
                            output_text=last_final_text,
                            processing_time_ms=int((time.time() - session_start) * 1000),
                            is_guest=(user is None),
                        )
                        db.add(record)
                        await db.commit()
                break

            # ── AUDIO CHUNK ──
            if msg_type == "chunk":
                chunk_data = msg.get("data", "")
                if not chunk_data:
                    continue

                chunk_index += 1
                result = await _live_stt_chunk_mock(chunk_data, chunk_index)

                if result["type"] == "buffering":
                    continue  # Don't send anything, just buffering

                if result["type"] == "partial":
                    await websocket.send_json({
                        "event": "partial",
                        "data": {
                            "text": result["text"],
                            "is_final": False,
                            "chunk_index": chunk_index,
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif result["type"] == "final":
                    last_final_text = result["text"]
                    await websocket.send_json({
                        "event": "final",
                        "data": {
                            "text": result["text"],
                            "confidence": result.get("confidence", 0.9),
                            "is_final": True,
                            "chunk_index": chunk_index,
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Immediately run sentiment on each final chunk
                    if include_sentiment:
                        sentiment = await analyze_sentiment(result["text"])
                        await websocket.send_json({
                            "event": "sentiment",
                            "data": {
                                "text": result["text"],
                                "emotion": sentiment["emotion"],
                                "sentiment_label": sentiment["sentiment_label"],
                                "sentiment_score": sentiment["sentiment_score"],
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                continue

            await websocket.send_json({
                "event": "error",
                "data": {"detail": f"Unknown type: {msg_type}"},
                "timestamp": datetime.utcnow().isoformat()
            })

    except WebSocketDisconnect:
        pass
