"""
HuggingFace authentication manager for DigiPal application.
"""

import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from .models import User, AuthSession, AuthResult, AuthStatus
from .session_manager import SessionManager
from ..storage.database import DatabaseConnection

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages HuggingFace authentication with offline support."""
    
    # HuggingFace API endpoints
    HF_API_BASE = "https://huggingface.co/api"
    HF_USER_ENDPOINT = f"{HF_API_BASE}/whoami"
    
    def __init__(self, db_connection: DatabaseConnection, offline_mode: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize authentication manager.
        
        Args:
            db_connection: Database connection for user storage
            offline_mode: Enable offline development mode
            cache_dir: Directory for authentication cache
        """
        self.db = db_connection
        self.offline_mode = offline_mode
        self.session_manager = SessionManager(db_connection, cache_dir)
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.timeout = 10  # 10 second timeout
        
        logger.info(f"AuthManager initialized (offline_mode: {offline_mode})")
    
    def authenticate(self, token: str) -> AuthResult:
        """
        Authenticate user with HuggingFace token.
        
        Args:
            token: HuggingFace authentication token
            
        Returns:
            Authentication result with user and session info
        """
        if self.offline_mode:
            return self._authenticate_offline(token)
        
        try:
            # Validate token with HuggingFace API
            user_info = self._validate_hf_token(token)
            if not user_info:
                return AuthResult(
                    status=AuthStatus.INVALID_TOKEN,
                    error_message="Invalid HuggingFace token"
                )
            
            # Create or update user
            user = self._create_or_update_user(user_info, token)
            if not user:
                return AuthResult(
                    status=AuthStatus.USER_NOT_FOUND,
                    error_message="Failed to create or update user"
                )
            
            # Create session
            session = self.session_manager.create_session(user, token)
            
            logger.info(f"Successfully authenticated user: {user.username}")
            return AuthResult(
                status=AuthStatus.SUCCESS,
                user=user,
                session=session
            )
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error during authentication: {e}")
            # Try offline authentication as fallback
            return self._authenticate_offline(token)
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthResult(
                status=AuthStatus.NETWORK_ERROR,
                error_message=f"Authentication failed: {str(e)}"
            )
    
    def validate_session(self, user_id: str, token: str) -> AuthResult:
        """
        Validate existing session.
        
        Args:
            user_id: User ID
            token: Authentication token
            
        Returns:
            Authentication result
        """
        # Check if session exists and is valid
        if not self.session_manager.validate_session(user_id, token):
            return AuthResult(
                status=AuthStatus.EXPIRED_SESSION,
                error_message="Session expired or invalid"
            )
        
        # Get user and session
        user = self.get_user(user_id)
        session = self.session_manager.get_session(user_id)
        
        if not user or not session:
            return AuthResult(
                status=AuthStatus.USER_NOT_FOUND,
                error_message="User or session not found"
            )
        
        # Refresh session
        self.session_manager.refresh_session(user_id)
        
        status = AuthStatus.OFFLINE_MODE if session.is_offline else AuthStatus.SUCCESS
        return AuthResult(
            status=status,
            user=user,
            session=session
        )
    
    def logout(self, user_id: str) -> bool:
        """
        Logout user and revoke session.
        
        Args:
            user_id: User ID to logout
            
        Returns:
            True if logout successful
        """
        success = self.session_manager.revoke_session(user_id)
        if success:
            logger.info(f"User {user_id} logged out successfully")
        return success
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found
        """
        try:
            rows = self.db.execute_query(
                'SELECT * FROM users WHERE id = ?',
                (user_id,)
            )
            
            if rows:
                row = rows[0]
                return User(
                    id=row['id'],
                    username=row['username'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None
                )
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
        
        return None
    
    def refresh_user_profile(self, user_id: str) -> Optional[User]:
        """
        Refresh user profile from HuggingFace.
        
        Args:
            user_id: User ID
            
        Returns:
            Updated user object
        """
        if self.offline_mode:
            return self.get_user(user_id)
        
        try:
            # Get current session to get token
            session = self.session_manager.get_session(user_id)
            if not session or session.is_offline:
                return self.get_user(user_id)
            
            # Fetch updated user info
            user_info = self._validate_hf_token(session.token)
            if user_info:
                user = self._create_or_update_user(user_info, session.token)
                logger.info(f"Refreshed profile for user: {user_id}")
                return user
                
        except Exception as e:
            logger.error(f"Error refreshing user profile: {e}")
        
        return self.get_user(user_id)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return self.session_manager.cleanup_expired_sessions()
    
    def _authenticate_offline(self, token: str) -> AuthResult:
        """
        Authenticate in offline mode using cached data.
        
        Args:
            token: Authentication token
            
        Returns:
            Authentication result for offline mode
        """
        # In offline mode, we create a development user
        # This is for development purposes only
        
        if not token or len(token) < 10:
            return AuthResult(
                status=AuthStatus.INVALID_TOKEN,
                error_message="Token too short for offline mode"
            )
        
        # Create a deterministic user ID from token
        import hashlib
        user_id = f"offline_{hashlib.md5(token.encode()).hexdigest()[:16]}"
        username = f"dev_user_{user_id[-8:]}"
        
        # Check if offline user exists
        user = self.get_user(user_id)
        if not user:
            # Create offline development user
            user = User(
                id=user_id,
                username=username,
                email=f"{username}@offline.dev",
                full_name=f"Development User {username}",
                created_at=datetime.now()
            )
            
            # Save to database
            try:
                self.db.execute_update(
                    '''INSERT OR REPLACE INTO users 
                       (id, username, huggingface_token, created_at, last_login) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (user.id, user.username, token, 
                     user.created_at.isoformat(), datetime.now().isoformat())
                )
            except Exception as e:
                logger.error(f"Error creating offline user: {e}")
                return AuthResult(
                    status=AuthStatus.NETWORK_ERROR,
                    error_message="Failed to create offline user"
                )
        
        # Create offline session
        session = self.session_manager.create_session(
            user, token, expires_hours=168, is_offline=True  # 1 week for offline
        )
        
        logger.info(f"Offline authentication successful for: {username}")
        return AuthResult(
            status=AuthStatus.OFFLINE_MODE,
            user=user,
            session=session
        )
    
    def _validate_hf_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate token with HuggingFace API.
        
        Args:
            token: HuggingFace token
            
        Returns:
            User info dict if valid, None otherwise
        """
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'User-Agent': 'DigiPal/1.0'
            }
            
            response = self.session.get(self.HF_USER_ENDPOINT, headers=headers)
            
            if response.status_code == 200:
                user_info = response.json()
                logger.debug(f"HF API response: {user_info}")
                return user_info
            elif response.status_code == 401:
                logger.warning("Invalid HuggingFace token")
                return None
            else:
                logger.error(f"HF API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error validating HF token: {e}")
            raise
        except Exception as e:
            logger.error(f"Error validating HF token: {e}")
            return None
    
    def _create_or_update_user(self, user_info: Dict[str, Any], token: str) -> Optional[User]:
        """
        Create or update user from HuggingFace user info.
        
        Args:
            user_info: User info from HuggingFace API
            token: Authentication token
            
        Returns:
            User object
        """
        try:
            # Extract user data from HF response
            user_id = user_info.get('name', user_info.get('id', ''))
            username = user_info.get('name', user_id)
            email = user_info.get('email')
            full_name = user_info.get('fullname', user_info.get('name'))
            avatar_url = user_info.get('avatarUrl')
            
            if not user_id:
                logger.error("No user ID in HuggingFace response")
                return None
            
            # Check if user exists
            existing_user = self.get_user(user_id)
            now = datetime.now()
            
            if existing_user:
                # Update existing user
                self.db.execute_update(
                    '''UPDATE users SET 
                       username = ?, huggingface_token = ?, last_login = ?
                       WHERE id = ?''',
                    (username, token, now.isoformat(), user_id)
                )
                
                # Update user object
                existing_user.username = username
                existing_user.last_login = now
                return existing_user
            else:
                # Create new user
                user = User(
                    id=user_id,
                    username=username,
                    email=email,
                    full_name=full_name,
                    avatar_url=avatar_url,
                    created_at=now,
                    last_login=now
                )
                
                self.db.execute_update(
                    '''INSERT INTO users 
                       (id, username, huggingface_token, created_at, last_login) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (user.id, user.username, token, 
                     user.created_at.isoformat(), user.last_login.isoformat())
                )
                
                logger.info(f"Created new user: {username}")
                return user
                
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            return None
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()