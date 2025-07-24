"""
Authentication data models for DigiPal application.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import json


class AuthStatus(Enum):
    """Authentication status enumeration."""
    SUCCESS = "success"
    INVALID_TOKEN = "invalid_token"
    NETWORK_ERROR = "network_error"
    OFFLINE_MODE = "offline_mode"
    EXPIRED_SESSION = "expired_session"
    USER_NOT_FOUND = "user_not_found"


@dataclass
class User:
    """User model for authenticated users."""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for storage."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        return cls(
            id=data['id'],
            username=data['username'],
            email=data.get('email'),
            full_name=data.get('full_name'),
            avatar_url=data.get('avatar_url'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            is_active=data.get('is_active', True)
        )


@dataclass
class AuthSession:
    """Authentication session model."""
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    is_offline: bool = False
    session_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if session is valid (not expired and has token)."""
        return not self.is_expired and bool(self.token)
    
    def refresh_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def extend_session(self, hours: int = 24) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now() + timedelta(hours=hours)
        self.refresh_access()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage."""
        return {
            'user_id': self.user_id,
            'token': self.token,
            'expires_at': self.expires_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'is_offline': self.is_offline,
            'session_data': json.dumps(self.session_data)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthSession':
        """Create session from dictionary."""
        return cls(
            user_id=data['user_id'],
            token=data['token'],
            expires_at=datetime.fromisoformat(data['expires_at']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            is_offline=data.get('is_offline', False),
            session_data=json.loads(data.get('session_data', '{}'))
        )


@dataclass
class AuthResult:
    """Result of authentication operation."""
    status: AuthStatus
    user: Optional[User] = None
    session: Optional[AuthSession] = None
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if authentication was successful."""
        return self.status == AuthStatus.SUCCESS
    
    @property
    def is_offline(self) -> bool:
        """Check if authentication is in offline mode."""
        return self.status == AuthStatus.OFFLINE_MODE