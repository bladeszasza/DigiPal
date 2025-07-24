"""
Tests for session manager.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from digipal.auth.models import User, AuthSession, AuthStatus
from digipal.auth.session_manager import SessionManager
from digipal.storage.database import DatabaseConnection


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    db_conn = DatabaseConnection(str(db_path))
    
    yield db_conn
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def session_manager(temp_db, temp_cache_dir):
    """Create session manager for testing."""
    return SessionManager(temp_db, temp_cache_dir)


@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        id="test_user_123",
        username="testuser",
        email="test@example.com"
    )


class TestSessionManager:
    """Test SessionManager functionality."""
    
    def test_create_session(self, session_manager, test_user):
        """Test session creation."""
        token = "test_token_123"
        session = session_manager.create_session(test_user, token)
        
        assert session.user_id == test_user.id
        assert session.token == token
        assert session.is_valid is True
        assert session.is_offline is False
        
        # Check if session is cached
        cached_session = session_manager.get_session(test_user.id)
        assert cached_session is not None
        assert cached_session.user_id == test_user.id
    
    def test_create_offline_session(self, session_manager, test_user):
        """Test offline session creation."""
        token = "offline_token_123"
        session = session_manager.create_session(
            test_user, token, expires_hours=168, is_offline=True
        )
        
        assert session.user_id == test_user.id
        assert session.token == token
        assert session.is_offline is True
        assert session.is_valid is True
    
    def test_get_session_from_cache(self, session_manager, test_user):
        """Test getting session from memory cache."""
        token = "cached_token_123"
        original_session = session_manager.create_session(test_user, token)
        
        # Get session should return from cache
        retrieved_session = session_manager.get_session(test_user.id)
        
        assert retrieved_session is not None
        assert retrieved_session.user_id == original_session.user_id
        assert retrieved_session.token == original_session.token
    
    def test_get_expired_session(self, session_manager, test_user):
        """Test handling of expired sessions."""
        token = "expired_token_123"
        
        # Create session that expires immediately
        expired_session = AuthSession(
            user_id=test_user.id,
            token=token,
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        # Manually add to cache
        session_manager._session_cache[test_user.id] = expired_session
        
        # Should return None for expired session
        retrieved_session = session_manager.get_session(test_user.id)
        assert retrieved_session is None
        
        # Should be removed from cache
        assert test_user.id not in session_manager._session_cache
    
    def test_validate_session(self, session_manager, test_user):
        """Test session validation."""
        token = "valid_token_123"
        session_manager.create_session(test_user, token)
        
        # Valid token should pass validation
        assert session_manager.validate_session(test_user.id, token) is True
        
        # Invalid token should fail validation
        assert session_manager.validate_session(test_user.id, "wrong_token") is False
        
        # Non-existent user should fail validation
        assert session_manager.validate_session("non_existent", token) is False
    
    def test_refresh_session(self, session_manager, test_user):
        """Test session refresh."""
        token = "refresh_token_123"
        original_session = session_manager.create_session(test_user, token, expires_hours=1)
        original_expires = original_session.expires_at
        
        # Refresh session
        success = session_manager.refresh_session(test_user.id, extend_hours=24)
        assert success is True
        
        # Check if expiration was extended
        refreshed_session = session_manager.get_session(test_user.id)
        assert refreshed_session.expires_at > original_expires
    
    def test_revoke_session(self, session_manager, test_user):
        """Test session revocation."""
        token = "revoke_token_123"
        session_manager.create_session(test_user, token)
        
        # Verify session exists
        assert session_manager.get_session(test_user.id) is not None
        
        # Revoke session
        success = session_manager.revoke_session(test_user.id)
        assert success is True
        
        # Verify session is gone
        assert session_manager.get_session(test_user.id) is None
    
    def test_cleanup_expired_sessions(self, session_manager, test_user):
        """Test cleanup of expired sessions."""
        # Create expired session
        expired_session = AuthSession(
            user_id=test_user.id,
            token="expired_token",
            expires_at=datetime.now() - timedelta(hours=1)
        )
        session_manager._session_cache[test_user.id] = expired_session
        
        # Create valid session for another user
        valid_user = User(id="valid_user", username="validuser")
        session_manager.create_session(valid_user, "valid_token")
        
        # Cleanup should remove expired session
        cleaned_count = session_manager.cleanup_expired_sessions()
        assert cleaned_count >= 1
        
        # Expired session should be gone
        assert test_user.id not in session_manager._session_cache
        
        # Valid session should remain
        assert session_manager.get_session(valid_user.id) is not None
    
    def test_offline_session_validation(self, session_manager, test_user):
        """Test validation of offline sessions."""
        token = "offline_token_123"
        session = session_manager.create_session(
            test_user, token, is_offline=True
        )
        
        # Offline sessions should validate with token hash comparison
        assert session_manager.validate_session(test_user.id, token) is True
        
        # Wrong token should still fail
        assert session_manager.validate_session(test_user.id, "wrong_token") is False
    
    def test_session_persistence(self, temp_db, temp_cache_dir, test_user):
        """Test session persistence across manager instances."""
        # Create session with first manager instance
        manager1 = SessionManager(temp_db, temp_cache_dir)
        token = "persistent_token_123"
        manager1.create_session(test_user, token)
        
        # Create second manager instance
        manager2 = SessionManager(temp_db, temp_cache_dir)
        
        # Should be able to retrieve session
        retrieved_session = manager2.get_session(test_user.id)
        assert retrieved_session is not None
        assert retrieved_session.user_id == test_user.id
        
        # For database persistence, token should match exactly
        # For cache persistence (offline), token will be hashed
        if not retrieved_session.is_offline:
            assert retrieved_session.token == token
        else:
            # For offline sessions, validate using the session manager's validation
            assert manager2.validate_session(test_user.id, token) is True
    
    def test_cache_file_operations(self, session_manager, test_user, temp_cache_dir):
        """Test cache file creation and loading."""
        token = "cache_token_123"
        session = session_manager.create_session(test_user, token)
        
        # Check if cache file was created
        cache_files = list(Path(temp_cache_dir).glob("session_*.json"))
        assert len(cache_files) > 0
        
        # Clear memory cache and try to load from file cache
        session_manager._session_cache.clear()
        
        # Should load from cache file for offline mode
        cached_session = session_manager._load_session_from_cache(test_user.id)
        assert cached_session is not None
        assert cached_session.user_id == test_user.id
        assert cached_session.is_offline is True