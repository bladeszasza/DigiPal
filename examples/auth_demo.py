#!/usr/bin/env python3
"""
Demo script for DigiPal HuggingFace authentication system.

This script demonstrates how to use the authentication system in both
online and offline modes.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import digipal
sys.path.insert(0, str(Path(__file__).parent.parent))

from digipal.auth import AuthManager, AuthStatus
from digipal.storage.database import DatabaseConnection


def demo_online_authentication():
    """Demonstrate online authentication with HuggingFace."""
    print("=== Online Authentication Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager in online mode
    auth_manager = AuthManager(db_conn, offline_mode=False)
    
    # Get token from environment or prompt user
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("Please set HUGGINGFACE_TOKEN environment variable or enter token:")
        token = input("HuggingFace Token: ").strip()
    
    if not token:
        print("No token provided, skipping online demo")
        return
    
    print(f"Authenticating with token: {token[:10]}...")
    
    # Authenticate
    result = auth_manager.authenticate(token)
    
    if result.is_success:
        print(f"✅ Authentication successful!")
        print(f"   User: {result.user.username}")
        print(f"   Email: {result.user.email}")
        print(f"   Session expires: {result.session.expires_at}")
        
        # Validate session
        validation = auth_manager.validate_session(result.user.id, token)
        if validation.is_success:
            print("✅ Session validation successful")
        
        # Refresh user profile
        refreshed_user = auth_manager.refresh_user_profile(result.user.id)
        if refreshed_user:
            print("✅ Profile refresh successful")
        
        # Logout
        logout_success = auth_manager.logout(result.user.id)
        if logout_success:
            print("✅ Logout successful")
    
    elif result.status == AuthStatus.OFFLINE_MODE:
        print("🔄 Fell back to offline mode due to network issues")
        print(f"   User: {result.user.username}")
        print(f"   Session expires: {result.session.expires_at}")
    
    else:
        print(f"❌ Authentication failed: {result.error_message}")
    
    print()


def demo_offline_authentication():
    """Demonstrate offline authentication for development."""
    print("=== Offline Authentication Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo_offline.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager in offline mode
    auth_manager = AuthManager(db_conn, offline_mode=True)
    
    # Use a development token
    dev_token = "dev_token_for_offline_demo_12345"
    
    print(f"Authenticating in offline mode with token: {dev_token[:20]}...")
    
    # Authenticate
    result = auth_manager.authenticate(dev_token)
    
    if result.status == AuthStatus.OFFLINE_MODE:
        print(f"✅ Offline authentication successful!")
        print(f"   User: {result.user.username}")
        print(f"   Email: {result.user.email}")
        print(f"   Session expires: {result.session.expires_at}")
        print(f"   Is offline: {result.session.is_offline}")
        
        # Validate session
        validation = auth_manager.validate_session(result.user.id, dev_token)
        if validation.status == AuthStatus.OFFLINE_MODE:
            print("✅ Offline session validation successful")
        
        # Test session persistence
        print("\n--- Testing Session Persistence ---")
        
        # Create new auth manager instance (simulates app restart)
        auth_manager2 = AuthManager(db_conn, offline_mode=True)
        
        # Should be able to validate existing session
        validation2 = auth_manager2.validate_session(result.user.id, dev_token)
        if validation2.status == AuthStatus.OFFLINE_MODE:
            print("✅ Session persisted across manager instances")
        
        # Logout
        logout_success = auth_manager.logout(result.user.id)
        if logout_success:
            print("✅ Logout successful")
    
    else:
        print(f"❌ Offline authentication failed: {result.error_message}")
    
    print()


def demo_session_management():
    """Demonstrate session management features."""
    print("=== Session Management Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo_sessions.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager in offline mode for demo
    auth_manager = AuthManager(db_conn, offline_mode=True)
    
    # Create multiple users
    tokens = [
        "demo_token_user1_12345",
        "demo_token_user2_67890",
        "demo_token_user3_abcde"
    ]
    
    users = []
    for i, token in enumerate(tokens):
        result = auth_manager.authenticate(token)
        if result.status == AuthStatus.OFFLINE_MODE:
            users.append(result.user)
            print(f"✅ Created user {i+1}: {result.user.username}")
    
    print(f"\nCreated {len(users)} users")
    
    # Test session refresh
    if users:
        user = users[0]
        token = tokens[0]
        
        print(f"\n--- Testing Session Refresh for {user.username} ---")
        
        # Get current session
        session = auth_manager.session_manager.get_session(user.id)
        original_expires = session.expires_at
        
        # Refresh session
        refresh_success = auth_manager.session_manager.refresh_session(user.id, extend_hours=48)
        if refresh_success:
            updated_session = auth_manager.session_manager.get_session(user.id)
            print(f"✅ Session refreshed")
            print(f"   Original expiry: {original_expires}")
            print(f"   New expiry: {updated_session.expires_at}")
    
    # Test cleanup
    print(f"\n--- Testing Session Cleanup ---")
    cleaned_count = auth_manager.cleanup_expired_sessions()
    print(f"✅ Cleaned up {cleaned_count} expired sessions")
    
    print()


def main():
    """Run all authentication demos."""
    print("DigiPal Authentication System Demo")
    print("=" * 40)
    
    try:
        # Run offline demo first (always works)
        demo_offline_authentication()
        
        # Run session management demo
        demo_session_management()
        
        # Run online demo if token is available
        if os.getenv('HUGGINGFACE_TOKEN'):
            demo_online_authentication()
        else:
            print("💡 Set HUGGINGFACE_TOKEN environment variable to test online authentication")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()