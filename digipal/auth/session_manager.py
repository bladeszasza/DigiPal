"""
Session management for DigiPal authentication system.
"""

import logging
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from .models import User, AuthSession, AuthStatus
from ..storage.database import DatabaseConnection

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions with secure token storage and caching."""
    
    def __init__(self, db_connection: DatabaseConnection, cache_dir: Optional[str] = None):
        """
        Initialize session manager.
        
        Args:
            db_connection: Database connection for persistent storage
            cache_dir: Directory for session cache files (optional)
        """
        self.db = db_connection
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.digipal' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory session cache for performance
        self._session_cache: Dict[str, AuthSession] = {}
        
        # Load existing sessions from database
        self._load_sessions_from_db()
    
    def create_session(self, user: User, token: str, expires_hours: int = 24, is_offline: bool = False) -> AuthSession:
        """
        Create a new authentication session.
        
        Args:
            user: Authenticated user
            token: Authentication token
            expires_hours: Session expiration in hours
            is_offline: Whether this is an offline session
            
        Returns:
            Created authentication session
        """
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        session = AuthSession(
            user_id=user.id,
            token=token,
            expires_at=expires_at,
            is_offline=is_offline
        )
        
        # Ensure user exists in database before saving session
        self._ensure_user_exists(user)
        
        # Store in database
        self._save_session_to_db(session)
        
        # Cache in memory
        self._session_cache[user.id] = session
        
        # Save to file cache for offline access
        if not is_offline:
            self._save_session_to_cache(session)
        
        logger.info(f"Created session for user {user.id} (offline: {is_offline})")
        return session
    
    def get_session(self, user_id: str) -> Optional[AuthSession]:
        """
        Get session for user ID.
        
        Args:
            user_id: User ID to get session for
            
        Returns:
            Authentication session if found and valid, None otherwise
        """
        # Check memory cache first
        if user_id in self._session_cache:
            session = self._session_cache[user_id]
            if session.is_valid:
                session.refresh_access()
                return session
            else:
                # Remove expired session
                del self._session_cache[user_id]
                self._remove_session_from_db(user_id)
        
        # Try to load from database
        session = self._load_session_from_db(user_id)
        if session and session.is_valid:
            self._session_cache[user_id] = session
            session.refresh_access()
            return session
        
        # Try to load from cache for offline mode
        cached_session = self._load_session_from_cache(user_id)
        if cached_session:
            # Mark as offline session
            cached_session.is_offline = True
            cached_session.extend_session(hours=168)  # 1 week for offline
            self._session_cache[user_id] = cached_session
            return cached_session
        
        return None
    
    def validate_session(self, user_id: str, token: str) -> bool:
        """
        Validate session token for user.
        
        Args:
            user_id: User ID
            token: Token to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        session = self.get_session(user_id)
        if not session:
            return False
        
        # For offline sessions, we're more lenient with token validation
        if session.is_offline:
            return self._hash_token(token) == self._hash_token(session.token)
        
        return session.token == token and session.is_valid
    
    def refresh_session(self, user_id: str, extend_hours: int = 24) -> bool:
        """
        Refresh session expiration.
        
        Args:
            user_id: User ID
            extend_hours: Hours to extend session
            
        Returns:
            True if session was refreshed, False otherwise
        """
        session = self.get_session(user_id)
        if not session:
            return False
        
        session.extend_session(extend_hours)
        self._save_session_to_db(session)
        
        if not session.is_offline:
            self._save_session_to_cache(session)
        
        logger.info(f"Refreshed session for user {user_id}")
        return True
    
    def revoke_session(self, user_id: str) -> bool:
        """
        Revoke user session.
        
        Args:
            user_id: User ID
            
        Returns:
            True if session was revoked, False if not found
        """
        # Remove from memory cache
        if user_id in self._session_cache:
            del self._session_cache[user_id]
        
        # Remove from database
        removed_from_db = self._remove_session_from_db(user_id)
        
        # Remove from file cache
        self._remove_session_from_cache(user_id)
        
        if removed_from_db:
            logger.info(f"Revoked session for user {user_id}")
        
        return removed_from_db
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from storage.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        
        # Clean memory cache
        expired_users = [
            user_id for user_id, session in self._session_cache.items()
            if session.is_expired
        ]
        
        for user_id in expired_users:
            del self._session_cache[user_id]
            cleaned_count += 1
        
        # Clean database
        try:
            db_cleaned = self.db.execute_update(
                'DELETE FROM users WHERE session_data IS NOT NULL AND '
                'json_extract(session_data, "$.expires_at") < ?',
                (datetime.now().isoformat(),)
            )
            cleaned_count += db_cleaned
        except Exception as e:
            logger.error(f"Error cleaning expired sessions from database: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
        
        return cleaned_count
    
    def _save_session_to_db(self, session: AuthSession) -> None:
        """Save session to database."""
        try:
            session_json = json.dumps(session.to_dict())
            self.db.execute_update(
                '''UPDATE users SET session_data = ?, last_login = ? 
                   WHERE id = ?''',
                (session_json, session.last_accessed.isoformat(), session.user_id)
            )
        except Exception as e:
            logger.error(f"Error saving session to database: {e}")
    
    def _load_session_from_db(self, user_id: str) -> Optional[AuthSession]:
        """Load session from database."""
        try:
            rows = self.db.execute_query(
                'SELECT session_data FROM users WHERE id = ? AND session_data IS NOT NULL',
                (user_id,)
            )
            
            if rows:
                session_data = json.loads(rows[0]['session_data'])
                return AuthSession.from_dict(session_data)
        except Exception as e:
            logger.error(f"Error loading session from database: {e}")
        
        return None
    
    def _remove_session_from_db(self, user_id: str) -> bool:
        """Remove session from database."""
        try:
            # First check if user exists
            rows = self.db.execute_query('SELECT id FROM users WHERE id = ?', (user_id,))
            if not rows:
                return False
            
            affected = self.db.execute_update(
                'UPDATE users SET session_data = NULL WHERE id = ?',
                (user_id,)
            )
            return affected > 0
        except Exception as e:
            logger.error(f"Error removing session from database: {e}")
            return False
    
    def _save_session_to_cache(self, session: AuthSession) -> None:
        """Save session to file cache for offline access."""
        try:
            cache_file = self.cache_dir / f"session_{self._hash_user_id(session.user_id)}.json"
            
            # Only cache essential session data for offline use
            cache_data = {
                'user_id': session.user_id,
                'token_hash': self._hash_token(session.token),
                'expires_at': session.expires_at.isoformat(),
                'created_at': session.created_at.isoformat(),
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Error saving session to cache: {e}")
    
    def _load_session_from_cache(self, user_id: str) -> Optional[AuthSession]:
        """Load session from file cache."""
        try:
            cache_file = self.cache_dir / f"session_{self._hash_user_id(user_id)}.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is not too old (max 1 week)
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now() - cached_at > timedelta(days=7):
                cache_file.unlink()  # Remove old cache
                return None
            
            # Create session from cache (token will be validated separately)
            return AuthSession(
                user_id=cache_data['user_id'],
                token=cache_data['token_hash'],  # This is hashed, will need special handling
                expires_at=datetime.fromisoformat(cache_data['expires_at']),
                created_at=datetime.fromisoformat(cache_data['created_at']),
                is_offline=True
            )
            
        except Exception as e:
            logger.error(f"Error loading session from cache: {e}")
            return None
    
    def _remove_session_from_cache(self, user_id: str) -> None:
        """Remove session from file cache."""
        try:
            cache_file = self.cache_dir / f"session_{self._hash_user_id(user_id)}.json"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Error removing session from cache: {e}")
    
    def _load_sessions_from_db(self) -> None:
        """Load all valid sessions from database into memory cache."""
        try:
            rows = self.db.execute_query(
                'SELECT id, session_data FROM users WHERE session_data IS NOT NULL'
            )
            
            for row in rows:
                try:
                    session_data = json.loads(row['session_data'])
                    session = AuthSession.from_dict(session_data)
                    
                    if session.is_valid:
                        self._session_cache[row['id']] = session
                except Exception as e:
                    logger.warning(f"Error loading session for user {row['id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading sessions from database: {e}")
    
    def _hash_token(self, token: str) -> str:
        """Hash token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for cache file naming."""
        return hashlib.md5(user_id.encode()).hexdigest()[:16]
    
    def _ensure_user_exists(self, user: User) -> None:
        """Ensure user exists in database before creating session."""
        try:
            # Check if user exists
            rows = self.db.execute_query('SELECT id FROM users WHERE id = ?', (user.id,))
            if not rows:
                # Create user record
                self.db.execute_update(
                    '''INSERT INTO users (id, username, created_at, last_login) 
                       VALUES (?, ?, ?, ?)''',
                    (user.id, user.username, 
                     user.created_at.isoformat() if user.created_at else datetime.now().isoformat(),
                     user.last_login.isoformat() if user.last_login else None)
                )
                logger.info(f"Created user record for session: {user.id}")
        except Exception as e:
            logger.error(f"Error ensuring user exists: {e}")