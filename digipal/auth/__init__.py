"""
Authentication module for DigiPal application.

This module provides HuggingFace authentication integration with session management
and offline development support.
"""

from .auth_manager import AuthManager
from .session_manager import SessionManager
from .models import User, AuthSession, AuthResult

__all__ = [
    'AuthManager',
    'SessionManager', 
    'User',
    'AuthSession',
    'AuthResult'
]