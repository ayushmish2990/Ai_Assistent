from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime

from app.services.collaboration_service import CollaborationService, SessionEventType

router = APIRouter()
collaboration_service = CollaborationService()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # session_id -> [websockets]
        self.user_connections: Dict[str, WebSocket] = {}  # user_id -> websocket
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        self.user_connections[user_id] = websocket
    
    def disconnect(self, websocket: WebSocket, session_id: str, user_id: str):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        if user_id in self.user_connections:
            del self.user_connections[user_id]
    
    async def send_to_session(self, session_id: str, message: dict, exclude_user: Optional[str] = None):
        if session_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[session_id]:
                try:
                    # Skip if this is the user who sent the message
                    if exclude_user and websocket == self.user_connections.get(exclude_user):
                        continue
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                if ws in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(ws)
    
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_text(json.dumps(message))
            except:
                del self.user_connections[user_id]

manager = ConnectionManager()

# Pydantic models
class CreateSessionRequest(BaseModel):
    name: str
    user_name: str
    user_email: str

class JoinSessionRequest(BaseModel):
    session_id: str
    user_name: str
    user_email: str

class OpenFileRequest(BaseModel):
    file_path: str
    content: str

class CodeChangeRequest(BaseModel):
    file_path: str
    change_type: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    old_content: str
    new_content: str
    new_file_content: str

class CursorPositionRequest(BaseModel):
    file_path: str
    line: int
    column: int
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None

class ChatMessageRequest(BaseModel):
    message: str
    message_type: str = "text"

# REST endpoints
@router.post("/sessions")
async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """
    Create a new collaboration session.
    """
    try:
        user_id = f"user_{datetime.now().timestamp()}"
        session = await collaboration_service.create_session(
            name=request.name,
            creator_id=user_id,
            creator_name=request.user_name,
            creator_email=request.user_email
        )
        
        return {
            "session_id": session.id,
            "user_id": user_id,
            "session": await collaboration_service.get_session_state(session.id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.post("/sessions/join")
async def join_session(request: JoinSessionRequest) -> Dict[str, Any]:
    """
    Join an existing collaboration session.
    """
    try:
        user_id = f"user_{datetime.now().timestamp()}"
        session = await collaboration_service.join_session(
            session_id=request.session_id,
            user_id=user_id,
            user_name=request.user_name,
            user_email=request.user_email
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.id,
            "user_id": user_id,
            "session": await collaboration_service.get_session_state(session.id)
        }
    except Exception as e:
        if "Session is full" in str(e):
            raise HTTPException(status_code=400, detail="Session is full")
        raise HTTPException(status_code=500, detail=f"Error joining session: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get session state and information.
    """
    session_state = await collaboration_service.get_session_state(session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_state

@router.get("/sessions/{session_id}/files/{file_path:path}")
async def get_file_content(session_id: str, file_path: str) -> Dict[str, Any]:
    """
    Get the current content of a file in the session.
    """
    file_content = await collaboration_service.get_file_content(session_id, file_path)
    if not file_content:
        raise HTTPException(status_code=404, detail="File not found in session")
    
    return file_content

@router.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all sessions a user is part of.
    """
    return await collaboration_service.get_user_sessions(user_id)

# WebSocket endpoint for real-time collaboration
@router.websocket("/ws/{session_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    await manager.connect(websocket, session_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "open_file":
                await handle_open_file(session_id, user_id, message.get("data", {}))
            
            elif message_type == "close_file":
                await handle_close_file(session_id, user_id, message.get("data", {}))
            
            elif message_type == "code_change":
                await handle_code_change(session_id, user_id, message.get("data", {}))
            
            elif message_type == "cursor_position":
                await handle_cursor_position(session_id, user_id, message.get("data", {}))
            
            elif message_type == "chat_message":
                await handle_chat_message(session_id, user_id, message.get("data", {}))
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id, user_id)
        await collaboration_service.leave_session(user_id)
        
        # Notify other users
        await manager.send_to_session(session_id, {
            "type": "user_left",
            "data": {"user_id": user_id}
        })

# WebSocket message handlers
async def handle_open_file(session_id: str, user_id: str, data: Dict[str, Any]):
    file_path = data.get("file_path")
    content = data.get("content", "")
    
    if file_path:
        success = await collaboration_service.open_file(user_id, file_path, content)
        if success:
            await manager.send_to_session(session_id, {
                "type": "file_opened",
                "data": {
                    "user_id": user_id,
                    "file_path": file_path
                }
            }, exclude_user=user_id)

async def handle_close_file(session_id: str, user_id: str, data: Dict[str, Any]):
    file_path = data.get("file_path")
    
    if file_path:
        success = await collaboration_service.close_file(user_id, file_path)
        if success:
            await manager.send_to_session(session_id, {
                "type": "file_closed",
                "data": {
                    "user_id": user_id,
                    "file_path": file_path
                }
            }, exclude_user=user_id)

async def handle_code_change(session_id: str, user_id: str, data: Dict[str, Any]):
    file_path = data.get("file_path")
    
    if file_path:
        change = await collaboration_service.apply_code_change(user_id, file_path, data)
        if change:
            await manager.send_to_session(session_id, {
                "type": "code_changed",
                "data": {
                    "user_id": user_id,
                    "file_path": file_path,
                    "change": {
                        "id": change.id,
                        "type": change.change_type,
                        "start_line": change.start_line,
                        "start_column": change.start_column,
                        "end_line": change.end_line,
                        "end_column": change.end_column,
                        "old_content": change.old_content,
                        "new_content": change.new_content,
                        "version": change.version
                    }
                }
            }, exclude_user=user_id)

async def handle_cursor_position(session_id: str, user_id: str, data: Dict[str, Any]):
    file_path = data.get("file_path")
    line = data.get("line")
    column = data.get("column")
    selection_start = data.get("selection_start")
    selection_end = data.get("selection_end")
    
    if file_path is not None and line is not None and column is not None:
        success = await collaboration_service.update_cursor_position(
            user_id, file_path, line, column, selection_start, selection_end
        )
        if success:
            await manager.send_to_session(session_id, {
                "type": "cursor_moved",
                "data": {
                    "user_id": user_id,
                    "file_path": file_path,
                    "line": line,
                    "column": column,
                    "selection_start": selection_start,
                    "selection_end": selection_end
                }
            }, exclude_user=user_id)

async def handle_chat_message(session_id: str, user_id: str, data: Dict[str, Any]):
    message = data.get("message")
    message_type = data.get("message_type", "text")
    
    if message:
        success = await collaboration_service.send_chat_message(user_id, message, message_type)
        if success:
            await manager.send_to_session(session_id, {
                "type": "chat_message",
                "data": {
                    "user_id": user_id,
                    "message": message,
                    "message_type": message_type,
                    "timestamp": datetime.now().isoformat()
                }
            })

# Event handler for collaboration service
async def collaboration_event_handler(event):
    """Handle events from the collaboration service and broadcast to WebSocket clients."""
    session_id = event.session_id
    
    message = {
        "type": event.event_type.value,
        "data": {
            "user_id": event.user_id,
            "timestamp": event.timestamp.isoformat(),
            **event.data
        }
    }
    
    await manager.send_to_session(session_id, message, exclude_user=event.user_id)

# Register the event handler
collaboration_service.add_event_handler(collaboration_event_handler)