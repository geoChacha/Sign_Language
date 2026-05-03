"""
EmotiSign Database Models
Covers: Users, Translation History, Chat Rooms, Chat Messages, Sessions
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, Float, Enum as SAEnum
)
from sqlalchemy.orm import relationship
import enum

from app.database import Base


# ─────────────────────────── Enums ───────────────────────────

class UserRole(str, enum.Enum):
    DEAF = "deaf"
    HEARING = "hearing"
    ADMIN = "admin"


class TranslationMode(str, enum.Enum):
    TEXT_TO_SIGN = "text_to_sign"
    SIGN_TO_TEXT = "sign_to_text"


class SignLanguage(str, enum.Enum):
    ASL = "ASL"
    PSL = "PSL"


class Emotion(str, enum.Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class MessageType(str, enum.Enum):
    TEXT = "text"
    SIGN_VIDEO = "sign_video"
    TRANSLATION = "translation"
    SYSTEM = "system"


# ─────────────────────────── User ───────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.HEARING, nullable=False)
    preferred_sign_language = Column(SAEnum(SignLanguage), default=SignLanguage.ASL)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    profile_picture_url = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    translations = relationship("TranslationHistory", back_populates="user", cascade="all, delete-orphan")
    sent_messages = relationship("ChatMessage", foreign_keys="ChatMessage.sender_id", back_populates="sender")
    chat_rooms_as_member = relationship("ChatRoomMember", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")


# ─────────────────────────── Translation History ───────────────────────────

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    mode = Column(SAEnum(TranslationMode), nullable=False)
    sign_language = Column(SAEnum(SignLanguage), default=SignLanguage.ASL)

    # Text-to-Sign fields
    input_text = Column(Text, nullable=True)
    output_sign_data = Column(Text, nullable=True)  # JSON: list of sign frames / GIF URLs

    # Sign-to-Text fields
    input_video_path = Column(String(255), nullable=True)
    output_text = Column(Text, nullable=True)

    # Emotion / Sentiment
    detected_emotion = Column(SAEnum(Emotion), default=Emotion.UNKNOWN)
    sentiment_score = Column(Float, nullable=True)   # -1.0 (negative) to +1.0 (positive)
    sentiment_label = Column(String(20), nullable=True)  # positive / neutral / negative

    # Metadata
    processing_time_ms = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    is_guest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="translations")


# ─────────────────────────── Chat Room ───────────────────────────

class ChatRoom(Base):
    __tablename__ = "chat_rooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    is_direct = Column(Boolean, default=True)  # True = 1-on-1 DM, False = group
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    members = relationship("ChatRoomMember", back_populates="room", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="room", cascade="all, delete-orphan")


class ChatRoomMember(Base):
    __tablename__ = "chat_room_members"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("chat_rooms.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    joined_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    room = relationship("ChatRoom", back_populates="members")
    user = relationship("User", back_populates="chat_rooms_as_member")


# ─────────────────────────── Chat Message ───────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("chat_rooms.id", ondelete="CASCADE"))
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    message_type = Column(SAEnum(MessageType), default=MessageType.TEXT)

    # Content
    text_content = Column(Text, nullable=True)
    translated_text = Column(Text, nullable=True)       # auto-translated output
    sign_data = Column(Text, nullable=True)             # JSON sign frames / URL for sign video
    video_path = Column(String(255), nullable=True)     # uploaded sign video path

    # Emotion context attached to the message
    emotion = Column(SAEnum(Emotion), default=Emotion.UNKNOWN)
    sentiment_label = Column(String(20), nullable=True)

    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    room = relationship("ChatRoom", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")


# ─────────────────────────── User Session ───────────────────────────

class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    token_jti = Column(String(100), unique=True, index=True)  # JWT unique ID for revocation
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="sessions")
