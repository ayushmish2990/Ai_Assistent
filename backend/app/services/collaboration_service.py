from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
from enum import Enum

class SessionEventType(Enum):
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CODE_CHANGED = "code_changed"
    CURSOR_MOVED = "cursor_moved"
    FILE_OPENED = "file_opened"
    FILE_CLOSED = "file_closed"
    CHAT_MESSAGE = "chat_message"
    AI_SUGGESTION = "ai_suggestion"

@dataclass
class User:
    id: str
    name: str
    email: str
    color: str  # For cursor/selection highlighting
    avatar_url: Optional[str] = None
    is_active: bool = True
    last_seen: datetime = field(default_factory=datetime.now)

@dataclass
class CursorPosition:
    file_path: str
    line: int
    column: int
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None

@dataclass
class CodeChange:
    id: str
    user_id: str
    file_path: str
    timestamp: datetime
    change_type: str  # 'insert', 'delete', 'replace'
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    old_content: str
    new_content: str
    version: int

@dataclass
class SessionEvent:
    id: str
    session_id: str
    user_id: str
    event_type: SessionEventType
    timestamp: datetime
    data: Dict[str, Any]

@dataclass
class FileState:
    path: str
    content: str
    version: int
    last_modified: datetime
    locked_by: Optional[str] = None
    active_users: Set[str] = field(default_factory=set)

@dataclass
class CollaborationSession:
    id: str
    name: str
    created_at: datetime
    created_by: str
    users: Dict[str, User] = field(default_factory=dict)
    files: Dict[str, FileState] = field(default_factory=dict)
    events: List[SessionEvent] = field(default_factory=list)
    cursor_positions: Dict[str, CursorPosition] = field(default_factory=dict)
    is_active: bool = True
    max_users: int = 10

class CollaborationService:
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.event_handlers: List = []
        self.file_watchers: Dict[str, Set[str]] = {}  # file_path -> set of session_ids
    
    async def create_session(self, name: str, creator_id: str, creator_name: str, creator_email: str) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = f"session_{datetime.now().timestamp()}"
        
        creator = User(
            id=creator_id,
            name=creator_name,
            email=creator_email,
            color="#3B82F6"  # Blue color for creator
        )
        
        session = CollaborationSession(
            id=session_id,
            name=name,
            created_at=datetime.now(),
            created_by=creator_id
        )
        
        session.users[creator_id] = creator
        self.sessions[session_id] = session
        self.user_sessions[creator_id] = session_id
        
        # Create join event
        await self._add_event(session_id, creator_id, SessionEventType.USER_JOINED, {
            "user": {
                "id": creator.id,
                "name": creator.name,
                "color": creator.color
            }
        })
        
        return session
    
    async def join_session(self, session_id: str, user_id: str, user_name: str, user_email: str) -> Optional[CollaborationSession]:
        """Join an existing collaboration session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if len(session.users) >= session.max_users:
            raise Exception("Session is full")
        
        # Assign a color to the user
        colors = ["#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4"]
        used_colors = {user.color for user in session.users.values()}
        available_colors = [c for c in colors if c not in used_colors]
        user_color = available_colors[0] if available_colors else "#6B7280"
        
        user = User(
            id=user_id,
            name=user_name,
            email=user_email,
            color=user_color
        )
        
        session.users[user_id] = user
        self.user_sessions[user_id] = session_id
        
        # Create join event
        await self._add_event(session_id, user_id, SessionEventType.USER_JOINED, {
            "user": {
                "id": user.id,
                "name": user.name,
                "color": user.color
            }
        })
        
        return session
    
    async def leave_session(self, user_id: str) -> bool:
        """Leave the current collaboration session."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions.get(session_id)
        
        if not session or user_id not in session.users:
            return False
        
        # Remove user from session
        user = session.users.pop(user_id)
        del self.user_sessions[user_id]
        
        # Remove user from file active users
        for file_state in session.files.values():
            file_state.active_users.discard(user_id)
            if file_state.locked_by == user_id:
                file_state.locked_by = None
        
        # Remove cursor position
        session.cursor_positions.pop(user_id, None)
        
        # Create leave event
        await self._add_event(session_id, user_id, SessionEventType.USER_LEFT, {
            "user": {
                "id": user.id,
                "name": user.name
            }
        })
        
        # Clean up empty session
        if not session.users:
            del self.sessions[session_id]
        
        return True
    
    async def open_file(self, user_id: str, file_path: str, content: str) -> bool:
        """Open a file in the collaboration session."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if file_path not in session.files:
            session.files[file_path] = FileState(
                path=file_path,
                content=content,
                version=1,
                last_modified=datetime.now()
            )
        
        session.files[file_path].active_users.add(user_id)
        
        # Add to file watchers
        if file_path not in self.file_watchers:
            self.file_watchers[file_path] = set()
        self.file_watchers[file_path].add(session_id)
        
        await self._add_event(session_id, user_id, SessionEventType.FILE_OPENED, {
            "file_path": file_path,
            "version": session.files[file_path].version
        })
        
        return True
    
    async def close_file(self, user_id: str, file_path: str) -> bool:
        """Close a file in the collaboration session."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if file_path in session.files:
            session.files[file_path].active_users.discard(user_id)
            
            # Unlock file if user had it locked
            if session.files[file_path].locked_by == user_id:
                session.files[file_path].locked_by = None
        
        await self._add_event(session_id, user_id, SessionEventType.FILE_CLOSED, {
            "file_path": file_path
        })
        
        return True
    
    async def update_cursor_position(self, user_id: str, file_path: str, line: int, column: int, 
                                   selection_start: Optional[int] = None, selection_end: Optional[int] = None) -> bool:
        """Update user's cursor position."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        cursor_pos = CursorPosition(
            file_path=file_path,
            line=line,
            column=column,
            selection_start=selection_start,
            selection_end=selection_end
        )
        
        session.cursor_positions[user_id] = cursor_pos
        
        await self._add_event(session_id, user_id, SessionEventType.CURSOR_MOVED, {
            "file_path": file_path,
            "line": line,
            "column": column,
            "selection_start": selection_start,
            "selection_end": selection_end
        })
        
        return True
    
    async def apply_code_change(self, user_id: str, file_path: str, change: Dict[str, Any]) -> Optional[CodeChange]:
        """Apply a code change to a file."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if file_path not in session.files:
            return None
        
        file_state = session.files[file_path]
        
        # Create change record
        code_change = CodeChange(
            id=f"change_{datetime.now().timestamp()}",
            user_id=user_id,
            file_path=file_path,
            timestamp=datetime.now(),
            change_type=change.get('type', 'replace'),
            start_line=change.get('start_line', 0),
            start_column=change.get('start_column', 0),
            end_line=change.get('end_line', 0),
            end_column=change.get('end_column', 0),
            old_content=change.get('old_content', ''),
            new_content=change.get('new_content', ''),
            version=file_state.version + 1
        )
        
        # Apply the change
        file_state.content = change.get('new_file_content', file_state.content)
        file_state.version += 1
        file_state.last_modified = datetime.now()
        
        await self._add_event(session_id, user_id, SessionEventType.CODE_CHANGED, {
            "file_path": file_path,
            "change": {
                "id": code_change.id,
                "type": code_change.change_type,
                "start_line": code_change.start_line,
                "start_column": code_change.start_column,
                "end_line": code_change.end_line,
                "end_column": code_change.end_column,
                "old_content": code_change.old_content,
                "new_content": code_change.new_content,
                "version": code_change.version
            }
        })
        
        return code_change
    
    async def send_chat_message(self, user_id: str, message: str, message_type: str = "text") -> bool:
        """Send a chat message in the collaboration session."""
        session_id = self.user_sessions.get(user_id)
        if not session_id or session_id not in self.sessions:
            return False
        
        await self._add_event(session_id, user_id, SessionEventType.CHAT_MESSAGE, {
            "message": message,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a collaboration session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            "id": session.id,
            "name": session.name,
            "created_at": session.created_at.isoformat(),
            "users": {
                uid: {
                    "id": user.id,
                    "name": user.name,
                    "color": user.color,
                    "is_active": user.is_active,
                    "last_seen": user.last_seen.isoformat()
                }
                for uid, user in session.users.items()
            },
            "files": {
                path: {
                    "path": file_state.path,
                    "version": file_state.version,
                    "last_modified": file_state.last_modified.isoformat(),
                    "active_users": list(file_state.active_users),
                    "locked_by": file_state.locked_by
                }
                for path, file_state in session.files.items()
            },
            "cursor_positions": {
                uid: {
                    "file_path": pos.file_path,
                    "line": pos.line,
                    "column": pos.column,
                    "selection_start": pos.selection_start,
                    "selection_end": pos.selection_end
                }
                for uid, pos in session.cursor_positions.items()
            },
            "recent_events": [
                {
                    "id": event.id,
                    "user_id": event.user_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data
                }
                for event in session.events[-50:]  # Last 50 events
            ]
        }
    
    async def get_file_content(self, session_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Get the current content of a file in the session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if file_path not in session.files:
            return None
        
        file_state = session.files[file_path]
        
        return {
            "path": file_state.path,
            "content": file_state.content,
            "version": file_state.version,
            "last_modified": file_state.last_modified.isoformat(),
            "active_users": list(file_state.active_users),
            "locked_by": file_state.locked_by
        }
    
    async def _add_event(self, session_id: str, user_id: str, event_type: SessionEventType, data: Dict[str, Any]):
        """Add an event to the session."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        event = SessionEvent(
            id=f"event_{datetime.now().timestamp()}",
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        
        session.events.append(event)
        
        # Keep only last 1000 events to prevent memory issues
        if len(session.events) > 1000:
            session.events = session.events[-1000:]
        
        # Notify event handlers
        for handler in self.event_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
    
    def add_event_handler(self, handler):
        """Add an event handler for real-time notifications."""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler):
        """Remove an event handler."""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions a user is part of."""
        user_sessions = []
        
        for session in self.sessions.values():
            if user_id in session.users:
                user_sessions.append({
                    "id": session.id,
                    "name": session.name,
                    "created_at": session.created_at.isoformat(),
                    "user_count": len(session.users),
                    "file_count": len(session.files),
                    "is_active": session.is_active
                })
        
        return user_sessions