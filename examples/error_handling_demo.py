#!/usr/bin/env python3
"""
Demonstration of DigiPal's comprehensive error handling and recovery system.

This script shows how the error handling system works across different
error scenarios and recovery strategies.
"""

import sys
import tempfile
import sqlite3
from pathlib import Path

# Add the parent directory to the path so we can import digipal modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from digipal.core.exceptions import (
    StorageError, AIModelError, NetworkError, AuthenticationError,
    SpeechProcessingError, ErrorSeverity
)
from digipal.core.error_integration import (
    initialize_error_handling, get_error_handler, handle_error_safely
)
from digipal.core.user_error_messages import MessageTone
from digipal.storage.backup_recovery import BackupRecoveryManager, BackupConfig


def demonstrate_basic_error_handling():
    """Demonstrate basic error handling functionality."""
    print("=== Basic Error Handling Demo ===")
    
    # Test different types of errors
    errors_to_test = [
        StorageError("Database connection failed", error_code="STOR_001"),
        AIModelError("Model loading failed", error_code="AI_001"),
        NetworkError("Connection timeout", error_code="NET_001"),
        AuthenticationError("Invalid credentials", error_code="AUTH_001"),
        SpeechProcessingError("Audio processing failed", error_code="SPEECH_001")
    ]
    
    for error in errors_to_test:
        print(f"\n--- Testing {error.__class__.__name__} ---")
        
        # Handle error safely
        result = handle_error_safely(
            error,
            context={'demo': True, 'error_type': error.__class__.__name__},
            fallback_value="Demo fallback value"
        )
        
        print(f"Success: {result.success}")
        print(f"User Message: {result.user_message}")
        print(f"Recovery Attempted: {result.recovery_attempted}")
        print(f"Fallback Value: {result.fallback_value}")
        
        if result.recovery_guide:
            print("Recovery Steps:")
            for step in result.recovery_guide[:2]:  # Show first 2 steps
                print(f"  {step['step']}. {step['title']}: {step['description']}")


def demonstrate_recovery_strategies():
    """Demonstrate recovery strategies with backup system."""
    print("\n=== Recovery Strategies Demo ===")
    
    # Create temporary database and backup system
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "demo.db"
        backup_dir = Path(temp_dir) / "backups"
        
        # Create test database with DigiPal schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE digipals (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT,
                data TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT,
                data TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE interactions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                digipal_id TEXT,
                data TEXT
            )
        """)
        conn.execute("INSERT INTO digipals VALUES ('demo_pet', 'demo_user', 'DemoPal', '{}')")
        conn.execute("INSERT INTO users VALUES ('demo_user', 'demo', '{}')")
        conn.commit()
        conn.close()
        
        # Initialize backup system
        backup_manager = BackupRecoveryManager(
            str(db_path),
            str(backup_dir),
            BackupConfig(max_backups=3)
        )
        
        # Initialize error handling with backup support
        initialize_error_handling(backup_manager)
        error_handler = get_error_handler()
        
        if error_handler:
            # Create a backup
            try:
                backup_result = backup_manager.create_backup("demo", description="Demo backup")
                if isinstance(backup_result, tuple):
                    success, backup_id = backup_result
                else:
                    success = backup_result
                    backup_id = None
                print(f"Backup created: {success}, ID: {backup_id}")
            except Exception as e:
                print(f"Backup creation failed: {e}")
                success = False
                backup_id = None
            
            # Simulate storage error with recovery
            storage_error = StorageError(
                "Critical database corruption",
                context={'db_path': str(db_path), 'operation': 'save_data'},
                error_code="STOR_CORRUPT_001"
            )
            storage_error.severity = ErrorSeverity.CRITICAL
            
            print(f"\n--- Testing Storage Error Recovery ---")
            result = error_handler.handle_error_comprehensive(
                storage_error,
                attempt_recovery=True
            )
            
            print(f"Recovery Success: {result.success}")
            print(f"Recovery Attempted: {result.recovery_attempted}")
            if result.recovery_result:
                print(f"Recovery Strategy: {result.recovery_result.strategy_used}")
                print(f"Recovery Time: {result.recovery_result.recovery_time_seconds:.3f}s")
            
            # Show system health
            health_status = error_handler.get_system_health_status()
            print(f"\nSystem Health Score: {health_status['overall_health_score']:.2f}")
            print(f"System Status: {health_status['status']}")


def demonstrate_user_friendly_messages():
    """Demonstrate user-friendly error messages with different tones."""
    print("\n=== User-Friendly Messages Demo ===")
    
    # Initialize error handling
    initialize_error_handling()
    error_handler = get_error_handler()
    
    if error_handler:
        # Test different message tones
        test_error = AIModelError(
            "Model inference failed due to memory constraints",
            context={'model_name': 'qwen3-0.6b', 'memory_usage': '8GB'},
            error_code="AI_MEM_001"
        )
        
        tones = [MessageTone.FRIENDLY, MessageTone.PROFESSIONAL, MessageTone.CASUAL, MessageTone.EMPATHETIC]
        
        for tone in tones:
            error_handler.set_message_tone(tone)
            
            result = error_handler.handle_error_comprehensive(
                test_error,
                context={'user_state': 'during_interaction'}
            )
            
            print(f"\n--- {tone.value.title()} Tone ---")
            print(f"Message: {result.user_message}")
            
            if result.recovery_guide:
                print("Quick Recovery Steps:")
                for step in result.recovery_guide[:2]:
                    print(f"  • {step['title']}")


def demonstrate_error_patterns():
    """Demonstrate error pattern analysis."""
    print("\n=== Error Pattern Analysis Demo ===")
    
    initialize_error_handling()
    error_handler = get_error_handler()
    
    if error_handler:
        # Simulate multiple errors to show pattern detection
        print("Simulating error patterns...")
        
        # Generate network errors
        for i in range(8):
            net_error = NetworkError(f"Connection failed {i}", error_code="NET_001")
            error_handler.handle_error_comprehensive(net_error)
        
        # Generate AI model errors
        for i in range(5):
            ai_error = AIModelError(f"Model error {i}", error_code="AI_001")
            error_handler.handle_error_comprehensive(ai_error)
        
        # Generate storage errors
        for i in range(12):
            storage_error = StorageError(f"Storage error {i}", error_code="STOR_001")
            error_handler.handle_error_comprehensive(storage_error)
        
        # Show error statistics
        from digipal.core.error_handler import error_handler as core_handler
        
        print(f"\nError Rate (last 5 min): {core_handler.get_error_rate():.1f} errors/min")
        print(f"Error Storm Detected: {core_handler.is_error_storm_detected()}")
        
        frequent_errors = core_handler.get_most_frequent_errors(3)
        print("\nMost Frequent Errors:")
        for error_key, count in frequent_errors:
            print(f"  {error_key}: {count} occurrences")


def demonstrate_health_monitoring():
    """Demonstrate system health monitoring."""
    print("\n=== System Health Monitoring Demo ===")
    
    initialize_error_handling()
    error_handler = get_error_handler()
    
    if error_handler:
        # Get initial health status
        print("Initial System Health:")
        health_report = error_handler.generate_health_report()
        print(health_report)
        
        # Simulate some errors to affect health
        print("\nSimulating system stress...")
        
        # Create various errors
        errors = [
            StorageError("Disk full", error_code="STOR_DISK_001"),
            AIModelError("GPU memory exhausted", error_code="AI_GPU_001"),
            NetworkError("DNS resolution failed", error_code="NET_DNS_001")
        ]
        
        for error in errors:
            error_handler.handle_error_comprehensive(error)
        
        # Show updated health
        print("\nUpdated System Health:")
        health_status = error_handler.get_system_health_status()
        print(f"Health Score: {health_status['overall_health_score']:.2f}")
        print(f"Status: {health_status['status']}")
        
        # Show recommendations
        health_report = error_handler.generate_health_report()
        if "Recommendations:" in health_report:
            recommendations_section = health_report.split("Recommendations:")[1].split("=== End of Report ===")[0]
            print(f"\nRecommendations:{recommendations_section}")


def demonstrate_safe_function_wrapper():
    """Demonstrate safe function wrapper functionality."""
    print("\n=== Safe Function Wrapper Demo ===")
    
    initialize_error_handling()
    error_handler = get_error_handler()
    
    if error_handler:
        # Create functions that might fail
        def risky_division(a, b):
            """Function that might divide by zero."""
            return a / b
        
        def risky_file_operation(filename):
            """Function that might fail to read file."""
            with open(filename, 'r') as f:
                return f.read()
        
        # Create safe versions
        safe_division = error_handler.create_error_safe_wrapper(
            risky_division,
            fallback_value=0,
            context={'operation': 'division'}
        )
        
        safe_file_read = error_handler.create_error_safe_wrapper(
            risky_file_operation,
            fallback_value="File not found",
            context={'operation': 'file_read'}
        )
        
        # Test safe functions
        print("Testing safe division:")
        print(f"  10 / 2 = {safe_division(10, 2)}")
        print(f"  10 / 0 = {safe_division(10, 0)} (fallback used)")
        
        print("\nTesting safe file read:")
        print(f"  Reading 'nonexistent.txt': {safe_file_read('nonexistent.txt')}")


def main():
    """Run all demonstrations."""
    print("DigiPal Comprehensive Error Handling System Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_error_handling()
        demonstrate_recovery_strategies()
        demonstrate_user_friendly_messages()
        demonstrate_error_patterns()
        demonstrate_health_monitoring()
        demonstrate_safe_function_wrapper()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("The error handling system provides:")
        print("  ✓ Comprehensive error categorization and handling")
        print("  ✓ Automatic recovery strategies")
        print("  ✓ User-friendly error messages with multiple tones")
        print("  ✓ Error pattern analysis and detection")
        print("  ✓ System health monitoring and reporting")
        print("  ✓ Safe function wrappers for error-prone operations")
        print("  ✓ Integration with backup and recovery systems")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()