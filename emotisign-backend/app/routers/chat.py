"""
Chat Router — /api/chat
REST:
  POST   /rooms                → create a chat room
  GET    /rooms                → list user's rooms
  GET    /rooms/{id}/messages  → paginated message history
  POST   /rooms/{id}/message   → send text/sign message (REST fallback)
  PATCH  /rooms/{id}/read      → mark messages as read

WebSocket:
  WS /ws/chat/{room_id}?token=<JWT>
    Events in:  text | sign_video_url | typing | read
    Events out: message | typing | user_joined | user_left | read | error | online_users
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter, Depends, HTTPException, Query,
    WebSocket, WebSocketDisconnect
)
from jose import JWTError
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, AsyncSessionLocal
from app.dependencies import get_current_user
from app.models import (
    ChatRoom, ChatRoomMember, ChatMessage, User,
    MessageType, Emotion, SignLanguage
)
from app.schemas import (
    CreateChatRoomRequest, ChatRoomResponse,
    ChatMessageResponse, PaginatedMessages, SuccessResponse
)
from app.services.auth_service import decode_token
from app.services.ws_manager import manager
from app.ml.ml_service import analyze_sentiment, text_to_sign, sign_to_text

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ─────────────────────────── Helpers ───────────────────────────

async def _assert_member(db: AsyncSession, room_id: int, user_id: int):
    res = await db.execute(
        select(ChatRoomMember).where(
            ChatRoomMember.room_id == room_id,
            ChatRoomMember.user_id == user_id,
            ChatRoomMember.is_active == True
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="You are not a member of this room")


def _msg_to_dict(msg: ChatMessage, sender_username: str = None) -> dict:
    return {
        "id": msg.id,
        "room_id": msg.room_id,
        "sender_id": msg.sender_id,
        "sender_username": sender_username,
        "message_type": msg.message_type.value,
        "text_content": msg.text_content,
        "translated_text": msg.translated_text,
        "sign_data": msg.sign_data,
        "video_path": msg.video_path,
        "emotion": msg.emotion.value if msg.emotion else None,
        "sentiment_label": msg.sentiment_label,
        "is_read": msg.is_read,
        "created_at": msg.created_at.isoformat()
    }


# ─────────────────────────── REST: Rooms ───────────────────────────

@router.post("/rooms", response_model=ChatRoomResponse, status_code=201)
async def create_room(
    payload: CreateChatRoomRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Ensure all members exist
    all_ids = list(set(payload.member_ids + [current_user.id]))
    result = await db.execute(select(User).where(User.id.in_(all_ids)))
    found_users = result.scalars().all()
    if len(found_users) != len(all_ids):
        raise HTTPException(status_code=404, detail="One or more users not found")

    room = ChatRoom(
        name=payload.name,
        is_direct=payload.is_direct,
        created_by=current_user.id
    )
    db.add(room)
    await db.flush()  # get room.id

    for uid in all_ids:
        db.add(ChatRoomMember(room_id=room.id, user_id=uid))

    await db.commit()
    await db.refresh(room)

    return ChatRoomResponse(
        id=room.id,
        name=room.name,
        is_direct=room.is_direct,
        created_at=room.created_at,
        member_count=len(all_ids)
    )


@router.get("/rooms", response_model=list[ChatRoomResponse])
async def list_my_rooms(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Rooms where current user is a member
    subq = select(ChatRoomMember.room_id).where(
        ChatRoomMember.user_id == current_user.id,
        ChatRoomMember.is_active == True
    )
    result = await db.execute(select(ChatRoom).where(ChatRoom.id.in_(subq)))
    rooms = result.scalars().all()

    out = []
    for room in rooms:
        count_res = await db.execute(
            select(func.count()).where(ChatRoomMember.room_id == room.id, ChatRoomMember.is_active == True)
        )
        out.append(ChatRoomResponse(
            id=room.id,
            name=room.name,
            is_direct=room.is_direct,
            created_at=room.created_at,
            member_count=count_res.scalar()
        ))
    return out


# ─────────────────────────── REST: Messages ───────────────────────────

@router.get("/rooms/{room_id}/messages", response_model=PaginatedMessages)
async def get_messages(
    room_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    await _assert_member(db, room_id, current_user.id)

    count_res = await db.execute(
        select(func.count()).where(ChatMessage.room_id == room_id)
    )
    total = count_res.scalar()

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.room_id == room_id)
        .order_by(ChatMessage.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    messages = result.scalars().all()

    items = []
    for msg in messages:
        sender_name = None
        if msg.sender_id:
            u = await db.get(User, msg.sender_id)
            sender_name = u.username if u else None
        items.append(ChatMessageResponse(
            **_msg_to_dict(msg, sender_name),
            id=msg.id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            sender_username=sender_name,
            message_type=msg.message_type,
            text_content=msg.text_content,
            translated_text=msg.translated_text,
            sign_data=msg.sign_data,
            video_path=msg.video_path,
            emotion=msg.emotion,
            sentiment_label=msg.sentiment_label,
            is_read=msg.is_read,
            created_at=msg.created_at
        ))

    return PaginatedMessages(items=items, total=total, page=page, page_size=page_size)


@router.patch("/rooms/{room_id}/read", response_model=SuccessResponse)
async def mark_as_read(
    room_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    await _assert_member(db, room_id, current_user.id)
    await db.execute(
        update(ChatMessage)
        .where(ChatMessage.room_id == room_id, ChatMessage.sender_id != current_user.id)
        .values(is_read=True)
    )
    await db.commit()
    return SuccessResponse(message="Messages marked as read")


# ─────────────────────────── WebSocket Chat ───────────────────────────

@router.websocket("/ws/chat/{room_id}")
async def websocket_chat(
    websocket: WebSocket,
    room_id: int,
    token: str = Query(...)
):
    """
    Real-time bidirectional chat via WebSocket.

    Connect: ws://<host>/api/chat/ws/chat/{room_id}?token=<JWT>

    Client → Server events (JSON):
      { "type": "text", "content": "Hello!" }
      { "type": "sign_video_url", "video_url": "...", "auto_translate": true }
      { "type": "typing" }
      { "type": "read" }
      { "type": "ping" }

    Server → Client events (JSON):
      { "event": "message", "data": { ...message }, "timestamp": "..." }
      { "event": "typing", "data": { "user_id": ..., "username": ... } }
      { "event": "user_joined", "data": { ... } }
      { "event": "user_left", "data": { ... } }
      { "event": "read", "data": { ... } }
      { "event": "online_users", "data": { "users": [...] } }
      { "event": "error", "data": { "detail": "..." } }
      { "event": "pong", "data": {} }
    """

    # ── Authenticate via token query param ──
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # ── Verify room membership ──
    async with AsyncSessionLocal() as db:
        user = await db.get(User, user_id)
        if not user or not user.is_active:
            await websocket.close(code=4001, reason="User not found or inactive")
            return

        res = await db.execute(
            select(ChatRoomMember).where(
                ChatRoomMember.room_id == room_id,
                ChatRoomMember.user_id == user_id,
                ChatRoomMember.is_active == True
            )
        )
        if not res.scalar_one_or_none():
            await websocket.close(code=4003, reason="Not a member of this room")
            return

    await manager.connect(websocket, room_id, user_id)

    # ── Notify others: user joined ──
    await manager.broadcast_to_room(
        room_id,
        manager.build_event("user_joined", {
            "user_id": user_id,
            "username": user.username,
            "online_users": manager.get_online_users(room_id)
        }),
        exclude_user_id=user_id
    )

    # ── Send online users to the newly connected user ──
    await manager.send_to_user(
        room_id, user_id,
        manager.build_event("online_users", {
            "users": manager.get_online_users(room_id)
        })
    )

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to_user(
                    room_id, user_id,
                    manager.build_event("error", {"detail": "Invalid JSON"})
                )
                continue

            event_type = data.get("type", "")

            # ── PING ──
            if event_type == "ping":
                await manager.send_to_user(
                    room_id, user_id,
                    manager.build_event("pong", {})
                )
                continue

            # ── TYPING INDICATOR ──
            if event_type == "typing":
                await manager.broadcast_to_room(
                    room_id,
                    manager.build_event("typing", {
                        "user_id": user_id,
                        "username": user.username
                    }),
                    exclude_user_id=user_id
                )
                continue

            # ── READ RECEIPT ──
            if event_type == "read":
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(ChatMessage)
                        .where(
                            ChatMessage.room_id == room_id,
                            ChatMessage.sender_id != user_id
                        )
                        .values(is_read=True)
                    )
                    await db.commit()
                await manager.broadcast_to_room(
                    room_id,
                    manager.build_event("read", {"user_id": user_id, "room_id": room_id}),
                    exclude_user_id=user_id
                )
                continue

            # ── TEXT MESSAGE ──
            if event_type == "text":
                content = data.get("content", "").strip()
                if not content:
                    continue

                auto_translate = data.get("auto_translate", True)

                # Sentiment analysis on text
                sentiment = await analyze_sentiment(content)
                emotion_val = Emotion(sentiment["emotion"]) if sentiment["emotion"] in Emotion._value2member_map_ else Emotion.UNKNOWN

                # Optional: auto-translate text → sign for the other side
                sign_data_str = None
                if auto_translate:
                    sign_result = await text_to_sign(content, "ASL")
                    import json as _json
                    sign_data_str = _json.dumps(sign_result["signs"])

                async with AsyncSessionLocal() as db:
                    msg = ChatMessage(
                        room_id=room_id,
                        sender_id=user_id,
                        message_type=MessageType.TEXT,
                        text_content=content,
                        sign_data=sign_data_str,
                        emotion=emotion_val,
                        sentiment_label=sentiment["sentiment_label"]
                    )
                    db.add(msg)
                    await db.commit()
                    await db.refresh(msg)

                msg_dict = _msg_to_dict(msg, user.username)
                msg_dict["emotion_scores"] = sentiment.get("emotion_scores", {})

                await manager.broadcast_to_all_in_room(
                    room_id,
                    manager.build_event("message", msg_dict)
                )
                continue

            # ── SIGN VIDEO URL ──
            if event_type == "sign_video_url":
                video_url = data.get("video_url", "").strip()
                if not video_url:
                    continue

                auto_translate = data.get("auto_translate", True)

                recognized_text = None
                emotion_val = Emotion.UNKNOWN
                if auto_translate:
                    # In real impl: download video or use path, run sign_to_text
                    sign_result = await sign_to_text(video_url, "ASL")
                    recognized_text = sign_result["recognized_text"]

                async with AsyncSessionLocal() as db:
                    msg = ChatMessage(
                        room_id=room_id,
                        sender_id=user_id,
                        message_type=MessageType.SIGN_VIDEO,
                        video_path=video_url,
                        translated_text=recognized_text,
                        emotion=emotion_val
                    )
                    db.add(msg)
                    await db.commit()
                    await db.refresh(msg)

                msg_dict = _msg_to_dict(msg, user.username)
                await manager.broadcast_to_all_in_room(
                    room_id,
                    manager.build_event("message", msg_dict)
                )
                continue

            # ── UNKNOWN EVENT ──
            await manager.send_to_user(
                room_id, user_id,
                manager.build_event("error", {"detail": f"Unknown event type: {event_type}"})
            )

    except WebSocketDisconnect:
        manager.disconnect(room_id, user_id)
        await manager.broadcast_to_room(
            room_id,
            manager.build_event("user_left", {
                "user_id": user_id,
                "username": user.username,
                "online_users": manager.get_online_users(room_id)
            })
        )
