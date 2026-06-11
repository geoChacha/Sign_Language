"""
EmotiSign FastAPI Backend v3.0
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os

from app.database import init_db
from app.routers import auth, translation, chat, realtime
from app.routers import speech
from app.routers import ml_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("uploads/chat_videos", exist_ok=True)
    os.makedirs("uploads/audio", exist_ok=True)
    os.makedirs("uploads/tts_output", exist_ok=True)
    os.makedirs("static/signs", exist_ok=True)
    await init_db()

    # ── Initialize SignGenerator (text-to-sign keypoint service) ──
    try:
        from app.ml.sign_generator import SignGenerator, set_sign_generator
        from app.ml.ml_service import set_sign_generator as ml_set_sign_generator
        sg = SignGenerator()
        sg.load()
        set_sign_generator(sg)
        ml_set_sign_generator(sg)
        print(f"✅ SignGenerator loaded {len(sg.vocabulary)} ASL signs")
    except Exception as e:
        import traceback
        print(f"❌ SignGenerator initialization failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"   Text-to-sign will fall back to fingerspelling for all words")

    # Initialize ML service (sign-to-text WLASL model)
    try:
        from app.ml.wlasl_service import get_wlasl_service
        ml_service = await get_wlasl_service()
        print("✅ WLASL-100 model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: ML service initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Sign-to-text endpoints will not be available")
        import traceback
        traceback.print_exc()
        ml_service = None

    # Initialize PSL live service (AlphabetClassifier — validates model file exists)
    try:
        from app.ml.psl_live_service import get_psl_live_service
        psl_live = get_psl_live_service()
        await psl_live.load_model()
        psl_live.close()  # release MediaPipe — per-session instances used at runtime
        print("✅ PSL live service (AlphabetClassifier) loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: PSL live service initialization failed: {e}")
        print(f"   PSL alphabet WebSocket endpoint will not be available")
        import traceback
        traceback.print_exc()

    print("✅ EmotiSign backend started.")
    print("🧪 Tester UI: http://127.0.0.1:8000/tester")
    print("📖 API Docs:  http://127.0.0.1:8000/docs")
    yield
    
    # Cleanup ML service
    try:
        if ml_service:
            ml_service.close()
            print("✅ ML service closed")
    except:
        pass


app = FastAPI(
    title="EmotiSign API",
    version="3.0.0",
    description="Bidirectional Sign Language Communication System",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# allow_credentials=True requires an explicit origin list — browsers reject "*"
# when credentials (cookies / Authorization headers) are included in the request.
_allowed_origins = settings.get_cors_origins()
print(f"🌐 CORS allowed origins: {_allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin",
                   "X-Requested-With", "Cache-Control"],
    expose_headers=["Content-Length", "Content-Range"],
)

# ── Static files ──
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ── Tester UI route ──
@app.get("/tester", include_in_schema=False)
async def tester():
    return FileResponse("static/tester.html")

# ── Routers ──
app.include_router(auth.router)
app.include_router(translation.router)
app.include_router(chat.router)
app.include_router(realtime.router)
app.include_router(speech.router)
app.include_router(ml_router.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "EmotiSign API",
        "version": "3.0.0",
        "tester_ui": "http://127.0.0.1:8000/tester",
        "docs": "http://127.0.0.1:8000/docs",
        "websockets": {
            "live_sign_to_text": "ws://127.0.0.1:8000/ws/translate/sign-to-text?token=<JWT>",
            "live_text_to_sign": "ws://127.0.0.1:8000/ws/translate/text-to-sign?token=<JWT>",
            "live_speech_to_text": "ws://127.0.0.1:8000/ws/speech/live-stt?token=<JWT>",
            "chat": "ws://127.0.0.1:8000/api/chat/ws/chat/{room_id}?token=<JWT>",
            "psl_live_alphabet": "ws://127.0.0.1:8000/ws/translate/psl-live",
        }
    }


@app.get("/health", tags=["Root"])
async def health():
    return JSONResponse({"status": "ok", "service": "emotisign-backend"})
