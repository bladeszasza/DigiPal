"""
Tests for authentication models.
"""

import pytest
from datetime import datetime, timedelta
import json

from digipal.auth.models import User, AuthSession, AuthResult, AuthStatus


class TestUser:
    """Test User model."""
    
    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com"
        )
        
        assert user.id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
    
    def test_user_to_dict(self):
        """Test user serialization to dictionary."""
        now = datetime.now()
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            created_at=now,
            last_login=now
        )
        
        user_dict = user.to_dict()
        
        assert user_dict['id'] == "test_user"
        assert user_dict['username'] == "testuser"
        assert user_dict['email'] == "test@example.com"
        assert user_dict['full_name'] == "Test User"
        assert user_dict['created_at'] == now.isoformat()
        assert user_dict['last_login'] == now.isoformat()
        assert user_dict['is_active'] is True
    
    def test_user_from_dict(self):
        """Test user deserialization from dictionary."""
        now = datetime.now()
        user_dict = {
            'id': "test_user",
            'username': "testuser",
            'email': "test@example.com",
            'full_name': "Test User",
            'created_at': now.isoformat(),
            'last_login': now.isoformat(),
            'is_active': True
        }
        
        user = User.from_dict(user_dict)
        
        assert user.id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.created_at == now
        assert user.last_login == now
        assert user.is_active is True


class TestAuthSession:
    """Test AuthSession model."""
    
    def test_session_creation(self):
        """Test basic session creation."""
        expires_at = datetime.now() + timedelta(hours=24)
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=expires_at
        )
        
        assert session.user_id == "test_user"
        assert session.token == "test_token"
        assert session.expires_at == expires_at
        assert session.is_offline is False
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_accessed, datetime)
    
    def test_session_expiration(self):
        """Test session expiration logic."""
        # Create expired session
        expired_session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        assert expired_session.is_expired is True
        assert expired_session.is_valid is False
        
        # Create valid session
        valid_session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        assert valid_session.is_expired is False
        assert valid_session.is_valid is True
    
    def test_session_refresh(self):
        """Test session refresh functionality."""
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        original_access_time = session.last_accessed
        
        # Wait a bit and refresh
        import time
        time.sleep(0.01)
        session.refresh_access()
        
        assert session.last_accessed > original_access_time
    
    def test_session_extension(self):
        """Test session extension."""
        original_expires = datetime.now() + timedelta(hours=1)
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=original_expires
        )
        
        session.extend_session(hours=24)
        
        assert session.expires_at > original_expires
        # Should be approximately 24 hours from now
        expected_expires = datetime.now() + timedelta(hours=24)
        time_diff = abs((session.expires_at - expected_expires).total_seconds())
        assert time_diff < 60  # Within 1 minute
    
    def test_session_serialization(self):
        """Test session to/from dict conversion."""
        now = datetime.now()
        expires_at = now + timedelta(hours=24)
        
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=expires_at,
            created_at=now,
            last_accessed=now,
            is_offline=True,
            session_data={"key": "value"}
        )
        
        # Test to_dict
        session_dict = session.to_dict()
        assert session_dict['user_id'] == "test_user"
        assert session_dict['token'] == "test_token"
        assert session_dict['expires_at'] == expires_at.isoformat()
        assert session_dict['is_offline'] is True
        assert json.loads(session_dict['session_data']) == {"key": "value"}
        
        # Test from_dict
        restored_session = AuthSession.from_dict(session_dict)
        assert restored_session.user_id == "test_user"
        assert restored_session.token == "test_token"
        assert restored_session.expires_at == expires_at
        assert restored_session.is_offline is True
        assert restored_session.session_data == {"key": "value"}


class TestAuthResult:
    """Test AuthResult model."""
    
    def test_successful_auth_result(self):
        """Test successful authentication result."""
        user = User(id="test_user", username="testuser")
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        result = AuthResult(
            status=AuthStatus.SUCCESS,
            user=user,
            session=session
        )
        
        assert result.is_success is True
        assert result.is_offline is False
        assert result.user == user
        assert result.session == session
        assert result.error_message is None
    
    def test_failed_auth_result(self):
        """Test failed authentication result."""
        result = AuthResult(
            status=AuthStatus.INVALID_TOKEN,
            error_message="Invalid token provided"
        )
        
        assert result.is_success is False
        assert result.is_offline is False
        assert result.user is None
        assert result.session is None
        assert result.error_message == "Invalid token provided"
    
    def test_offline_auth_result(self):
        """Test offline authentication result."""
        user = User(id="test_user", username="testuser")
        session = AuthSession(
            user_id="test_user",
            token="test_token",
            expires_at=datetime.now() + timedelta(hours=24),
            is_offline=True
        )
        
        result = AuthResult(
            status=AuthStatus.OFFLINE_MODE,
            user=user,
            session=session
        )
        
        assert result.is_success is False
        assert result.is_offline is True
        assert result.user == user
        assert result.session == session


class TestAuthStatus:
    """Test AuthStatus enum."""
    
    def test_auth_status_values(self):
        """Test all auth status enum values."""
        assert AuthStatus.SUCCESS.value == "success"
        assert AuthStatus.INVALID_TOKEN.value == "invalid_token"
        assert AuthStatus.NETWORK_ERROR.value == "network_error"
        assert AuthStatus.OFFLINE_MODE.value == "offline_mode"
        assert AuthStatus.EXPIRED_SESSION.value == "expired_session"
        assert AuthStatus.USER_NOT_FOUND.value == "user_not_found"