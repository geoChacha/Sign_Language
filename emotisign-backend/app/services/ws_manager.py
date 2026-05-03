"""
WebSocket Connection Manager for EmotiSign Real-Time Chat
"""

import json
from datetime import datetime
from typing import Dict, Set

from fastapi import WebSocket


class ConnectionManager:
    """
    Manages active WebSocket connections per chat room.
    Maps: room_id -> set of (user_id, websocket) tuples
    """

    def __init__(self):
        # room_id -> {user_id: WebSocket}
        self.active_connections: Dict[int, Dict[int, WebSocket]] = {}
        # user_id -> set of room_ids (for multi-room tracking)
        self.user_rooms: Dict[int, Set[int]] = {}

    async def connect(self, websocket: WebSocket, room_id: int, user_id: int):
        await websocket.accept()

        if room_id not in self.active_connections:
            self.active_connections[room_id] = {}
        self.active_connections[room_id][user_id] = websocket

        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = set()
        self.user_rooms[user_id].add(room_id)

    def disconnect(self, room_id: int, user_id: int):
        if room_id in self.active_connections:
            self.active_connections[room_id].pop(user_id, None)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room_id)
            if not self.user_rooms[user_id]:
                del self.user_rooms[user_id]

    async def send_to_user(self, room_id: int, user_id: int, data: dict):
        """Send a message to a specific user in a room."""
        ws = self.active_connections.get(room_id, {}).get(user_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(room_id, user_id)

    async def broadcast_to_room(self, room_id: int, data: dict, exclude_user_id: int = None):
        """Broadcast a message to all connected users in a room."""
        connections = self.active_connections.get(room_id, {})
        disconnected = []

        for uid, ws in connections.items():
            if uid == exclude_user_id:
                continue
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(uid)

        for uid in disconnected:
            self.disconnect(room_id, uid)

    async def broadcast_to_all_in_room(self, room_id: int, data: dict):
        """Broadcast to every user including sender."""
        await self.broadcast_to_room(room_id, data, exclude_user_id=None)

    def get_online_users(self, room_id: int) -> list[int]:
        return list(self.active_connections.get(room_id, {}).keys())

    def is_user_online(self, room_id: int, user_id: int) -> bool:
        return user_id in self.active_connections.get(room_id, {})

    @staticmethod
    def build_event(event: str, data: dict) -> dict:
        return {
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global singleton
manager = ConnectionManager()
