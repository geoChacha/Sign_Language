"""
Pydantic Schemas for EmotiSign API
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator
from app.models import UserRole, SignLanguage, Emotion, TranslationMode, MessageType


# ──────────────────────────────────────────────────────────────
#  AUTH SCHEMAS
# ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.HEARING
    preferred_sign_language: SignLanguage = SignLanguage.ASL

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: "UserResponse"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: UserRole
    preferred_sign_language: SignLanguage
    is_active: bool
    is_verified: bool
    profile_picture_url: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    preferred_sign_language: Optional[SignLanguage] = None
    profile_picture_url: Optional[str] = None


# ──────────────────────────────────────────────────────────────
#  TEXT-TO-SIGN SCHEMAS
# ──────────────────────────────────────────────────────────────

class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    sign_language: SignLanguage = SignLanguage.ASL
    include_emotion: bool = True


class SignUnit(BaseModel):
    word: str
    keypoints: Optional[List[List[List[float]]]] = None  # (N_frames, 75, 2) — canvas rendering
    frames: List[str] = []
    gif_url: Optional[str] = None
    fingerspelled: bool = False


class EmotionData(BaseModel):
    emotion: str
    sentiment_label: str
    sentiment_score: float
    emotion_scores: Dict[str, float]


class TextToSignResponse(BaseModel):
    translation_id: int
    input_text: str
    words: List[str]
    signs: List[Dict[str, Any]]
    total_duration_ms: int
    sign_language: str
    fingerspelled_words: List[str]
    emotion_analysis: Optional[EmotionData] = None
    processing_time_ms: int


# ──────────────────────────────────────────────────────────────
#  SIGN-TO-TEXT SCHEMAS
# ──────────────────────────────────────────────────────────────

class SignToTextResponse(BaseModel):
    translation_id: int
    recognized_text: str
    glosses: List[str]
    confidence: float
    sign_language: str
    frame_count: int
    processing_time_ms: int
    emotion_from_video: Optional[EmotionData] = None


# ──────────────────────────────────────────────────────────────
#  TRANSLATION HISTORY
# ──────────────────────────────────────────────────────────────

class TranslationHistoryItem(BaseModel):
    id: int
    mode: TranslationMode
    sign_language: SignLanguage
    input_text: Optional[str]
    output_text: Optional[str]
    detected_emotion: Optional[Emotion]
    sentiment_score: Optional[float]
    confidence_score: Optional[float]
    processing_time_ms: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedTranslations(BaseModel):
    items: List[TranslationHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ──────────────────────────────────────────────────────────────
#  CHAT SCHEMAS
# ──────────────────────────────────────────────────────────────

class CreateChatRoomRequest(BaseModel):
    name: Optional[str] = None
    member_ids: List[int] = Field(..., min_length=1)
    is_direct: bool = True


class ChatRoomResponse(BaseModel):
    id: int
    name: Optional[str]
    is_direct: bool
    created_at: datetime
    member_count: int

    model_config = {"from_attributes": True}


class ChatMessageResponse(BaseModel):
    id: int
    room_id: int
    sender_id: Optional[int]
    sender_username: Optional[str]
    message_type: MessageType
    text_content: Optional[str]
    translated_text: Optional[str]
    sign_data: Optional[str]
    video_path: Optional[str]
    sign_language: Optional[str] = None  # ASL | PSL for sign_video messages
    emotion: Optional[Emotion]
    sentiment_label: Optional[str]
    is_read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedMessages(BaseModel):
    items: List[ChatMessageResponse]
    total: int
    page: int
    page_size: int


# ──────────────────────────────────────────────────────────────
#  WEBSOCKET PAYLOADS
# ──────────────────────────────────────────────────────────────

class WSMessageIn(BaseModel):
    """Incoming WebSocket message from client"""
    type: str  # "text" | "sign_video_url" | "typing" | "read"
    room_id: int
    content: Optional[str] = None
    video_url: Optional[str] = None
    auto_translate: bool = True


class WSMessageOut(BaseModel):
    """Outgoing WebSocket message to client(s)"""
    event: str  # "message" | "typing" | "user_joined" | "user_left" | "read" | "error"
    data: Dict[str, Any]
    timestamp: str


# ──────────────────────────────────────────────────────────────
#  GENERIC
# ──────────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


TokenResponse.model_rebuild()
