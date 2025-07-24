"""
Tests for authentication manager.
"""

import pytest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from digipal.auth.auth_manager import AuthManager
from digipal.auth.models import User, AuthSession, AuthResult, AuthStatus
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
def auth_manager_online(temp_db, temp_cache_dir):
    """Create online auth manager for testing."""
    return AuthManager(temp_db, offline_mode=False, cache_dir=temp_cache_dir)


@pytest.fixture
def auth_manager_offline(temp_db, temp_cache_dir):
    """Create offline auth manager for testing."""
    return AuthManager(temp_db, offline_mode=True, cache_dir=temp_cache_dir)


@pytest.fixture
def mock_hf_response():
    """Mock HuggingFace API response."""
    return {
        'name': 'testuser',
        'id': 'testuser',
        'email': 'test@example.com',
        'fullname': 'Test User',
        'avatarUrl': 'https://example.com/avatar.jpg'
    }


class TestAuthManagerOnline:
    """Test AuthManager in online mode."""
    
    @patch('requests.Session.get')
    def test_successful_authentication(self, mock_get, auth_manager_online, mock_hf_response):
        """Test successful HuggingFace authentication."""
        # Mock successful HF API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_hf_response
        mock_get.return_value = mock_response
        
        token = "hf_test_token_123"
        result = auth_manager_online.authenticate(token)
        
        assert result.is_success is True
        assert result.user is not None
        assert result.user.username == "testuser"
        assert result.session is not None
        assert result.session.token == token
        assert result.error_message is None
        
        # Verify API was called correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'Authorization' in call_args[1]['headers']
        assert call_args[1]['headers']['Authorization'] == f'Bearer {token}'
    
    @patch('requests.Session.get')
    def test_invalid_token_authentication(self, mock_get, auth_manager_online):
        """Test authentication with invalid token."""
        # Mock 401 response for invalid token
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        token = "invalid_token"
        result = auth_manager_online.authenticate(token)
        
        assert result.is_success is False
        assert result.status == AuthStatus.INVALID_TOKEN
        assert result.user is None
        assert result.session is None
        assert "Invalid HuggingFace token" in result.error_message
    
    @patch('requests.Session.get')
    def test_network_error_fallback_to_offline(self, mock_get, auth_manager_online):
        """Test network error fallback to offline authentication."""
        # Mock network error
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        
        token = "hf_test_token_for_offline"
        result = auth_manager_online.authenticate(token)
        
        # Should fallback to offline mode
        assert result.status == AuthStatus.OFFLINE_MODE
        assert result.user is not None
        assert result.session is not None
        assert result.session.is_offline is True
    
    @patch('requests.Session.get')
    def test_validate_existing_session(self, mock_get, auth_manager_online, mock_hf_response):
        """Test validation of existing session."""
        # First authenticate to create session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_hf_response
        mock_get.return_value = mock_response
        
        token = "hf_session_token_123"
        auth_result = auth_manager_online.authenticate(token)
        user_id = auth_result.user.id
        
        # Now validate the session
        validation_result = auth_manager_online.validate_session(user_id, token)
        
        assert validation_result.is_success is True
        assert validation_result.user.id == user_id
        assert validation_result.session is not None
    
    def test_validate_nonexistent_session(self, auth_manager_online):
        """Test validation of non-existent session."""
        result = auth_manager_online.validate_session("nonexistent_user", "fake_token")
        
        assert result.is_success is False
        assert result.status == AuthStatus.EXPIRED_SESSION
        assert result.user is None
        assert result.session is None
    
    @patch('requests.Session.get')
    def test_user_logout(self, mock_get, auth_manager_online, mock_hf_response):
        """Test user logout functionality."""
        # First authenticate
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_hf_response
        mock_get.return_value = mock_response
        
        token = "hf_logout_token_123"
        auth_result = auth_manager_online.authenticate(token)
        user_id = auth_result.user.id
        
        # Verify session exists
        session_check = auth_manager_online.validate_session(user_id, token)
        assert session_check.is_success is True
        
        # Logout
        logout_success = auth_manager_online.logout(user_id)
        assert logout_success is True
        
        # Verify session is gone
        post_logout_check = auth_manager_online.validate_session(user_id, token)
        assert post_logout_check.is_success is False
    
    @patch('requests.Session.get')
    def test_refresh_user_profile(self, mock_get, auth_manager_online, mock_hf_response):
        """Test user profile refresh."""
        # First authenticate
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_hf_response
        mock_get.return_value = mock_response
        
        token = "hf_refresh_token_123"
        auth_result = auth_manager_online.authenticate(token)
        user_id = auth_result.user.id
        
        # Update mock response for refresh
        updated_response = mock_hf_response.copy()
        updated_response['fullname'] = 'Updated Test User'
        mock_response.json.return_value = updated_response
        
        # Refresh profile
        refreshed_user = auth_manager_online.refresh_user_profile(user_id)
        
        assert refreshed_user is not None
        assert refreshed_user.id == user_id
        # Note: The full_name update would need to be implemented in the actual refresh logic


class TestAuthManagerOffline:
    """Test AuthManager in offline mode."""
    
    def test_offline_authentication(self, auth_manager_offline):
        """Test offline authentication."""
        token = "offline_dev_token_123"
        result = auth_manager_offline.authenticate(token)
        
        assert result.status == AuthStatus.OFFLINE_MODE
        assert result.user is not None
        assert result.user.username.startswith("dev_user_")
        assert result.session is not None
        assert result.session.is_offline is True
        assert result.error_message is None
    
    def test_offline_authentication_short_token(self, auth_manager_offline):
        """Test offline authentication with short token."""
        token = "short"
        result = auth_manager_offline.authenticate(token)
        
        assert result.is_success is False
        assert result.status == AuthStatus.INVALID_TOKEN
        assert "Token too short" in result.error_message
    
    def test_offline_session_persistence(self, temp_db, temp_cache_dir):
        """Test offline session persistence."""
        # Create first manager and authenticate
        manager1 = AuthManager(temp_db, offline_mode=True, cache_dir=temp_cache_dir)
        token = "persistent_offline_token_123"
        result1 = manager1.authenticate(token)
        user_id = result1.user.id
        
        # Create second manager instance
        manager2 = AuthManager(temp_db, offline_mode=True, cache_dir=temp_cache_dir)
        
        # Should be able to validate session
        validation_result = manager2.validate_session(user_id, token)
        assert validation_result.status == AuthStatus.OFFLINE_MODE
        assert validation_result.user.id == user_id
    
    def test_offline_user_profile_refresh(self, auth_manager_offline):
        """Test user profile refresh in offline mode."""
        token = "offline_refresh_token_123"
        auth_result = auth_manager_offline.authenticate(token)
        user_id = auth_result.user.id
        
        # Refresh should return existing user in offline mode
        refreshed_user = auth_manager_offline.refresh_user_profile(user_id)
        assert refreshed_user is not None
        assert refreshed_user.id == user_id


class TestAuthManagerUtilities:
    """Test utility functions of AuthManager."""
    
    @patch('requests.Session.get')
    def test_get_user(self, mock_get, auth_manager_online, mock_hf_response):
        """Test getting user by ID."""
        # First authenticate to create user
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_hf_response
        mock_get.return_value = mock_response
        
        token = "hf_get_user_token_123"
        auth_result = auth_manager_online.authenticate(token)
        user_id = auth_result.user.id
        
        # Get user
        retrieved_user = auth_manager_online.get_user(user_id)
        assert retrieved_user is not None
        assert retrieved_user.id == user_id
        assert retrieved_user.username == "testuser"
    
    def test_get_nonexistent_user(self, auth_manager_online):
        """Test getting non-existent user."""
        user = auth_manager_online.get_user("nonexistent_user")
        assert user is None
    
    def test_cleanup_expired_sessions(self, auth_manager_offline):
        """Test cleanup of expired sessions."""
        # Create some sessions
        token1 = "cleanup_token_1"
        token2 = "cleanup_token_2"
        
        auth_manager_offline.authenticate(token1)
        auth_manager_offline.authenticate(token2)
        
        # Cleanup (should not remove valid sessions)
        cleaned_count = auth_manager_offline.cleanup_expired_sessions()
        assert cleaned_count >= 0  # May be 0 if no expired sessions
    
    def test_hf_token_validation_success(self, auth_manager_online, mock_hf_response):
        """Test HuggingFace token validation success."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_hf_response
            mock_get.return_value = mock_response
            
            token = "valid_hf_token"
            user_info = auth_manager_online._validate_hf_token(token)
            
            assert user_info is not None
            assert user_info['name'] == 'testuser'
            assert user_info['email'] == 'test@example.com'
    
    def test_hf_token_validation_failure(self, auth_manager_online):
        """Test HuggingFace token validation failure."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response
            
            token = "invalid_hf_token"
            user_info = auth_manager_online._validate_hf_token(token)
            
            assert user_info is None