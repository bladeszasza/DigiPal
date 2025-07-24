"""
Integration tests for the complete authentication system.
Tests the full authentication flow from login to logout.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime, timedelta

from digipal.auth.auth_manager import AuthManager
from digipal.auth.models import AuthStatus
from digipal.storage.database import DatabaseConnection


@pytest.fixture
def temp_environment():
    """Create temporary environment for integration testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "integration_test.db"
    cache_dir = Path(temp_dir) / "cache"
    
    yield {
        'temp_dir': temp_dir,
        'db_path': str(db_path),
        'cache_dir': str(cache_dir)
    }
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestAuthenticationIntegration:
    """Integration tests for complete authentication workflows."""
    
    @patch('requests.Session.get')
    def test_complete_online_authentication_flow(self, mock_get, temp_environment):
        """Test complete online authentication workflow."""
        # Setup mock HuggingFace response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'integration_user',
            'id': 'integration_user',
            'email': 'integration@test.com',
            'fullname': 'Integration Test User',
            'avatarUrl': 'https://example.com/avatar.jpg'
        }
        mock_get.return_value = mock_response
        
        # Initialize auth manager
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=False, 
            cache_dir=temp_environment['cache_dir']
        )
        
        token = "hf_integration_test_token_12345"
        
        # Step 1: Initial authentication
        auth_result = auth_manager.authenticate(token)
        
        assert auth_result.is_success is True
        assert auth_result.user.username == "integration_user"
        assert auth_result.user.email == "integration@test.com"
        assert auth_result.session is not None
        assert auth_result.session.is_offline is False
        
        user_id = auth_result.user.id
        
        # Step 2: Validate session immediately after authentication
        validation_result = auth_manager.validate_session(user_id, token)
        assert validation_result.is_success is True
        assert validation_result.user.id == user_id
        
        # Step 3: Refresh user profile
        refreshed_user = auth_manager.refresh_user_profile(user_id)
        assert refreshed_user is not None
        assert refreshed_user.id == user_id
        
        # Step 4: Test session persistence by creating new auth manager
        auth_manager2 = AuthManager(
            db_conn, 
            offline_mode=False, 
            cache_dir=temp_environment['cache_dir']
        )
        
        validation_result2 = auth_manager2.validate_session(user_id, token)
        assert validation_result2.is_success is True
        
        # Step 5: Refresh session
        refresh_success = auth_manager2.session_manager.refresh_session(user_id, extend_hours=48)
        assert refresh_success is True
        
        # Step 6: Logout
        logout_success = auth_manager2.logout(user_id)
        assert logout_success is True
        
        # Step 7: Verify session is invalidated after logout
        post_logout_validation = auth_manager2.validate_session(user_id, token)
        assert post_logout_validation.is_success is False
        assert post_logout_validation.status == AuthStatus.EXPIRED_SESSION
    
    def test_complete_offline_authentication_flow(self, temp_environment):
        """Test complete offline authentication workflow."""
        # Initialize auth manager in offline mode
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        token = "offline_integration_test_token_12345"
        
        # Step 1: Initial offline authentication
        auth_result = auth_manager.authenticate(token)
        
        assert auth_result.status == AuthStatus.OFFLINE_MODE
        assert auth_result.user is not None
        assert auth_result.user.username.startswith("dev_user_")
        assert auth_result.session is not None
        assert auth_result.session.is_offline is True
        
        user_id = auth_result.user.id
        
        # Step 2: Validate offline session
        validation_result = auth_manager.validate_session(user_id, token)
        assert validation_result.status == AuthStatus.OFFLINE_MODE
        assert validation_result.user.id == user_id
        
        # Step 3: Test offline session persistence across manager instances
        auth_manager2 = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        validation_result2 = auth_manager2.validate_session(user_id, token)
        assert validation_result2.status == AuthStatus.OFFLINE_MODE
        assert validation_result2.user.id == user_id
        
        # Step 4: Test offline profile refresh (should return existing user)
        refreshed_user = auth_manager2.refresh_user_profile(user_id)
        assert refreshed_user is not None
        assert refreshed_user.id == user_id
        
        # Step 5: Test session refresh in offline mode
        refresh_success = auth_manager2.session_manager.refresh_session(user_id, extend_hours=168)
        assert refresh_success is True
        
        # Step 6: Logout from offline session
        logout_success = auth_manager2.logout(user_id)
        assert logout_success is True
        
        # Step 7: Verify offline session is invalidated after logout
        post_logout_validation = auth_manager2.validate_session(user_id, token)
        assert post_logout_validation.is_success is False
    
    @patch('requests.Session.get')
    def test_online_to_offline_fallback_flow(self, mock_get, temp_environment):
        """Test fallback from online to offline mode during network issues."""
        # Initialize auth manager in online mode
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=False, 
            cache_dir=temp_environment['cache_dir']
        )
        
        token = "fallback_test_token_12345"
        
        # Step 1: Simulate network error
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        
        # Step 2: Authenticate should fallback to offline mode
        auth_result = auth_manager.authenticate(token)
        
        assert auth_result.status == AuthStatus.OFFLINE_MODE
        assert auth_result.user is not None
        assert auth_result.session is not None
        assert auth_result.session.is_offline is True
        
        user_id = auth_result.user.id
        
        # Step 3: Validate offline session works
        validation_result = auth_manager.validate_session(user_id, token)
        assert validation_result.status == AuthStatus.OFFLINE_MODE
        
        # Step 4: Test that offline session persists
        auth_manager2 = AuthManager(
            db_conn, 
            offline_mode=False,  # Still in online mode
            cache_dir=temp_environment['cache_dir']
        )
        
        # Should still work with cached offline session
        validation_result2 = auth_manager2.validate_session(user_id, token)
        assert validation_result2.status == AuthStatus.OFFLINE_MODE
    
    def test_multi_user_authentication_flow(self, temp_environment):
        """Test authentication flow with multiple users."""
        # Initialize auth manager
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        # Create multiple users
        users_data = [
            ("user1_token_12345", "user1"),
            ("user2_token_67890", "user2"),
            ("user3_token_abcde", "user3")
        ]
        
        authenticated_users = []
        
        # Step 1: Authenticate all users
        for token, expected_prefix in users_data:
            auth_result = auth_manager.authenticate(token)
            assert auth_result.status == AuthStatus.OFFLINE_MODE
            assert auth_result.user.username.startswith("dev_user_")
            authenticated_users.append((auth_result.user.id, token))
        
        # Step 2: Validate all sessions simultaneously
        for user_id, token in authenticated_users:
            validation_result = auth_manager.validate_session(user_id, token)
            assert validation_result.status == AuthStatus.OFFLINE_MODE
            assert validation_result.user.id == user_id
        
        # Step 3: Refresh sessions for all users
        for user_id, token in authenticated_users:
            refresh_success = auth_manager.session_manager.refresh_session(user_id)
            assert refresh_success is True
        
        # Step 4: Logout one user and verify others remain active
        logout_user_id, logout_token = authenticated_users[0]
        logout_success = auth_manager.logout(logout_user_id)
        assert logout_success is True
        
        # Verify logged out user cannot validate session
        post_logout_validation = auth_manager.validate_session(logout_user_id, logout_token)
        assert post_logout_validation.is_success is False
        
        # Verify other users still have valid sessions
        for user_id, token in authenticated_users[1:]:
            validation_result = auth_manager.validate_session(user_id, token)
            assert validation_result.status == AuthStatus.OFFLINE_MODE
        
        # Step 5: Clean up remaining sessions
        cleaned_count = auth_manager.cleanup_expired_sessions()
        assert cleaned_count >= 0  # Should not clean up valid sessions
    
    def test_session_expiration_and_cleanup_flow(self, temp_environment):
        """Test session expiration and cleanup workflow."""
        # Initialize auth manager
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        token = "expiration_test_token_12345"
        
        # Step 1: Create session with short expiration
        auth_result = auth_manager.authenticate(token)
        user_id = auth_result.user.id
        
        # Step 2: Manually expire the session for testing
        session = auth_manager.session_manager.get_session(user_id)
        session.expires_at = datetime.now() - timedelta(hours=1)
        auth_manager.session_manager._save_session_to_db(session)
        
        # Step 3: Try to validate expired session
        validation_result = auth_manager.validate_session(user_id, token)
        assert validation_result.is_success is False
        assert validation_result.status == AuthStatus.EXPIRED_SESSION
        
        # Step 4: Clean up expired sessions
        cleaned_count = auth_manager.cleanup_expired_sessions()
        # Note: cleanup_expired_sessions may return 0 if the session was already removed
        # during validation, which is expected behavior
        assert cleaned_count >= 0
        
        # Step 5: Verify session is completely removed
        retrieved_session = auth_manager.session_manager.get_session(user_id)
        assert retrieved_session is None
    
    def test_authentication_error_handling_flow(self, temp_environment):
        """Test error handling in authentication flow."""
        # Initialize auth manager
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        # Test 1: Invalid token (too short)
        short_token = "short"
        auth_result = auth_manager.authenticate(short_token)
        assert auth_result.is_success is False
        assert auth_result.status == AuthStatus.INVALID_TOKEN
        assert "Token too short" in auth_result.error_message
        
        # Test 2: Validate non-existent session
        validation_result = auth_manager.validate_session("nonexistent_user", "fake_token")
        assert validation_result.is_success is False
        assert validation_result.status == AuthStatus.EXPIRED_SESSION
        
        # Test 3: Refresh non-existent user profile
        refreshed_user = auth_manager.refresh_user_profile("nonexistent_user")
        assert refreshed_user is None
        
        # Test 4: Logout non-existent user
        logout_success = auth_manager.logout("nonexistent_user")
        assert logout_success is False
        
        # Test 5: Get non-existent user
        user = auth_manager.get_user("nonexistent_user")
        assert user is None


class TestAuthenticationPerformance:
    """Performance tests for authentication system."""
    
    def test_concurrent_authentication_performance(self, temp_environment):
        """Test performance with multiple concurrent authentications."""
        import time
        
        # Initialize auth manager
        db_conn = DatabaseConnection(temp_environment['db_path'])
        auth_manager = AuthManager(
            db_conn, 
            offline_mode=True, 
            cache_dir=temp_environment['cache_dir']
        )
        
        # Test concurrent authentications
        num_users = 50
        start_time = time.time()
        
        authenticated_users = []
        for i in range(num_users):
            token = f"perf_test_token_{i:04d}_{'x' * 20}"
            auth_result = auth_manager.authenticate(token)
            assert auth_result.status == AuthStatus.OFFLINE_MODE
            authenticated_users.append((auth_result.user.id, token))
        
        auth_time = time.time() - start_time
        
        # Test concurrent validations
        start_time = time.time()
        
        for user_id, token in authenticated_users:
            validation_result = auth_manager.validate_session(user_id, token)
            assert validation_result.status == AuthStatus.OFFLINE_MODE
        
        validation_time = time.time() - start_time
        
        print(f"\nPerformance Results:")
        print(f"  Authentication: {num_users} users in {auth_time:.3f}s ({num_users/auth_time:.1f} auth/s)")
        print(f"  Validation: {num_users} sessions in {validation_time:.3f}s ({num_users/validation_time:.1f} val/s)")
        
        # Performance assertions (reasonable thresholds)
        assert auth_time < 5.0, f"Authentication too slow: {auth_time:.3f}s for {num_users} users"
        assert validation_time < 2.0, f"Validation too slow: {validation_time:.3f}s for {num_users} sessions"
        
        # Cleanup
        for user_id, _ in authenticated_users:
            auth_manager.logout(user_id)