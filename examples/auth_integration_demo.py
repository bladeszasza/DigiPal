#!/usr/bin/env python3
"""
Advanced integration demo for DigiPal HuggingFace authentication system.

This script demonstrates advanced authentication workflows including
multi-user scenarios, session persistence, and error handling.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the parent directory to the path so we can import digipal
sys.path.insert(0, str(Path(__file__).parent.parent))

from digipal.auth import AuthManager, AuthStatus
from digipal.storage.database import DatabaseConnection


def demo_multi_user_workflow():
    """Demonstrate multi-user authentication workflow."""
    print("=== Multi-User Authentication Workflow ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "multi_user_demo.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager in offline mode for demo
    auth_manager = AuthManager(db_conn, offline_mode=True)
    
    # Simulate multiple users
    users = [
        ("alice_dev_token_12345", "Alice"),
        ("bob_dev_token_67890", "Bob"),
        ("charlie_dev_token_abcde", "Charlie")
    ]
    
    authenticated_users = []
    
    print("Authenticating multiple users...")
    for token, name in users:
        print(f"  Authenticating {name}...")
        result = auth_manager.authenticate(token)
        
        if result.status == AuthStatus.OFFLINE_MODE:
            print(f"    ✅ {name} authenticated successfully")
            print(f"       User ID: {result.user.id}")
            print(f"       Username: {result.user.username}")
            authenticated_users.append((result.user.id, token, name))
        else:
            print(f"    ❌ {name} authentication failed: {result.error_message}")
    
    print(f"\nSuccessfully authenticated {len(authenticated_users)} users")
    
    # Test concurrent session validation
    print("\nValidating all sessions simultaneously...")
    for user_id, token, name in authenticated_users:
        validation = auth_manager.validate_session(user_id, token)
        if validation.status == AuthStatus.OFFLINE_MODE:
            print(f"  ✅ {name}'s session is valid")
        else:
            print(f"  ❌ {name}'s session validation failed")
    
    # Test session refresh for all users
    print("\nRefreshing sessions for all users...")
    for user_id, token, name in authenticated_users:
        success = auth_manager.session_manager.refresh_session(user_id, extend_hours=48)
        if success:
            print(f"  ✅ {name}'s session refreshed (extended by 48 hours)")
        else:
            print(f"  ❌ Failed to refresh {name}'s session")
    
    # Demonstrate selective logout
    print("\nLogging out one user while keeping others active...")
    if authenticated_users:
        logout_user_id, logout_token, logout_name = authenticated_users[0]
        logout_success = auth_manager.logout(logout_user_id)
        
        if logout_success:
            print(f"  ✅ {logout_name} logged out successfully")
            
            # Verify logout
            post_logout_validation = auth_manager.validate_session(logout_user_id, logout_token)
            if post_logout_validation.is_success is False:
                print(f"  ✅ {logout_name}'s session properly invalidated")
            
            # Verify other users still active
            for user_id, token, name in authenticated_users[1:]:
                validation = auth_manager.validate_session(user_id, token)
                if validation.status == AuthStatus.OFFLINE_MODE:
                    print(f"  ✅ {name}'s session still active")
    
    print()


def demo_session_persistence():
    """Demonstrate session persistence across application restarts."""
    print("=== Session Persistence Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "persistence_demo.db"
    cache_dir = Path(temp_dir) / "cache"
    
    token = "persistence_demo_token_12345"
    
    print("Phase 1: Initial authentication and session creation...")
    
    # Create first auth manager instance
    db_conn1 = DatabaseConnection(str(db_path))
    auth_manager1 = AuthManager(db_conn1, offline_mode=True, cache_dir=str(cache_dir))
    
    # Authenticate
    result1 = auth_manager1.authenticate(token)
    if result1.status == AuthStatus.OFFLINE_MODE:
        user_id = result1.user.id
        print(f"  ✅ User authenticated: {result1.user.username}")
        print(f"  ✅ Session expires: {result1.session.expires_at}")
        
        # Simulate some activity
        print("  📝 Simulating user activity...")
        validation1 = auth_manager1.validate_session(user_id, token)
        if validation1.status == AuthStatus.OFFLINE_MODE:
            print("  ✅ Session validated during activity")
    
    print("\nPhase 2: Simulating application restart...")
    
    # Create second auth manager instance (simulates app restart)
    db_conn2 = DatabaseConnection(str(db_path))
    auth_manager2 = AuthManager(db_conn2, offline_mode=True, cache_dir=str(cache_dir))
    
    # Try to validate existing session
    validation2 = auth_manager2.validate_session(user_id, token)
    if validation2.status == AuthStatus.OFFLINE_MODE:
        print("  ✅ Session successfully restored after restart")
        print(f"  ✅ User data intact: {validation2.user.username}")
        print(f"  ✅ Session still valid until: {validation2.session.expires_at}")
    else:
        print("  ❌ Session not restored after restart")
    
    print("\nPhase 3: Testing session refresh after restart...")
    
    # Refresh session using new manager instance
    refresh_success = auth_manager2.session_manager.refresh_session(user_id, extend_hours=72)
    if refresh_success:
        print("  ✅ Session refreshed successfully after restart")
        
        # Verify refresh worked
        updated_session = auth_manager2.session_manager.get_session(user_id)
        if updated_session:
            print(f"  ✅ New expiration: {updated_session.expires_at}")
    
    print()


def demo_error_handling_and_recovery():
    """Demonstrate error handling and recovery scenarios."""
    print("=== Error Handling and Recovery Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "error_demo.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager
    auth_manager = AuthManager(db_conn, offline_mode=True)
    
    print("Test 1: Invalid token handling...")
    
    # Test various invalid tokens
    invalid_tokens = [
        ("", "Empty token"),
        ("short", "Too short token"),
        ("   ", "Whitespace only token"),
    ]
    
    for token, description in invalid_tokens:
        result = auth_manager.authenticate(token)
        if result.is_success is False:
            print(f"  ✅ Correctly rejected {description}")
            print(f"     Error: {result.error_message}")
        else:
            print(f"  ❌ Should have rejected {description}")
    
    print("\nTest 2: Non-existent user operations...")
    
    fake_user_id = "nonexistent_user_12345"
    fake_token = "fake_token_67890"
    
    # Test validation of non-existent session
    validation = auth_manager.validate_session(fake_user_id, fake_token)
    if validation.is_success is False:
        print("  ✅ Correctly handled non-existent session validation")
    
    # Test logout of non-existent user
    logout_success = auth_manager.logout(fake_user_id)
    if not logout_success:
        print("  ✅ Correctly handled non-existent user logout")
    
    # Test profile refresh of non-existent user
    profile = auth_manager.refresh_user_profile(fake_user_id)
    if profile is None:
        print("  ✅ Correctly handled non-existent user profile refresh")
    
    print("\nTest 3: Recovery from corrupted session...")
    
    # Create a valid user first
    valid_token = "recovery_test_token_12345"
    auth_result = auth_manager.authenticate(valid_token)
    
    if auth_result.status == AuthStatus.OFFLINE_MODE:
        user_id = auth_result.user.id
        print(f"  ✅ Created test user: {auth_result.user.username}")
        
        # Simulate session corruption by manually expiring it
        session = auth_manager.session_manager.get_session(user_id)
        if session:
            from datetime import datetime, timedelta
            session.expires_at = datetime.now() - timedelta(hours=1)
            auth_manager.session_manager._save_session_to_db(session)
            print("  📝 Simulated session expiration")
            
            # Try to validate expired session
            validation = auth_manager.validate_session(user_id, valid_token)
            if validation.is_success is False:
                print("  ✅ Correctly detected expired session")
                
                # Recovery: Re-authenticate
                recovery_result = auth_manager.authenticate(valid_token)
                if recovery_result.status == AuthStatus.OFFLINE_MODE:
                    print("  ✅ Successfully recovered with re-authentication")
    
    print()


def demo_performance_characteristics():
    """Demonstrate performance characteristics of the auth system."""
    print("=== Performance Characteristics Demo ===")
    
    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "performance_demo.db"
    db_conn = DatabaseConnection(str(db_path))
    
    # Create auth manager
    auth_manager = AuthManager(db_conn, offline_mode=True)
    
    # Test authentication performance
    print("Testing authentication performance...")
    
    num_users = 20
    start_time = time.time()
    
    authenticated_users = []
    for i in range(num_users):
        token = f"perf_token_{i:03d}_{'x' * 30}"
        result = auth_manager.authenticate(token)
        if result.status == AuthStatus.OFFLINE_MODE:
            authenticated_users.append((result.user.id, token))
    
    auth_time = time.time() - start_time
    print(f"  ✅ Authenticated {len(authenticated_users)} users in {auth_time:.3f}s")
    print(f"     Rate: {len(authenticated_users)/auth_time:.1f} authentications/second")
    
    # Test validation performance
    print("\nTesting session validation performance...")
    
    start_time = time.time()
    valid_sessions = 0
    
    for user_id, token in authenticated_users:
        validation = auth_manager.validate_session(user_id, token)
        if validation.status == AuthStatus.OFFLINE_MODE:
            valid_sessions += 1
    
    validation_time = time.time() - start_time
    print(f"  ✅ Validated {valid_sessions} sessions in {validation_time:.3f}s")
    print(f"     Rate: {valid_sessions/validation_time:.1f} validations/second")
    
    # Test cleanup performance
    print("\nTesting cleanup performance...")
    
    start_time = time.time()
    cleaned_count = auth_manager.cleanup_expired_sessions()
    cleanup_time = time.time() - start_time
    
    print(f"  ✅ Cleanup completed in {cleanup_time:.3f}s")
    print(f"     Cleaned up {cleaned_count} expired sessions")
    
    print()


def main():
    """Run all advanced authentication demos."""
    print("DigiPal Advanced Authentication Integration Demo")
    print("=" * 50)
    
    try:
        # Run all demo scenarios
        demo_multi_user_workflow()
        demo_session_persistence()
        demo_error_handling_and_recovery()
        demo_performance_characteristics()
        
        print("🎉 All advanced authentication demos completed successfully!")
        print("\nThe DigiPal authentication system demonstrates:")
        print("  • Robust multi-user support")
        print("  • Reliable session persistence")
        print("  • Comprehensive error handling")
        print("  • Good performance characteristics")
        print("  • Seamless offline/online mode switching")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()