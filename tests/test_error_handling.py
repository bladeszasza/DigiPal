"""
Comprehensive tests for error handling and recovery systems.

This module tests all error handling scenarios, recovery mechanisms,
and graceful degradation functionality.
"""

import pytest
import tempfile
import shutil
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from digipal.core.exceptions import (
    DigiPalException, AuthenticationError, StorageError, AIModelError,
    SpeechProcessingError, ImageGenerationError, PetLifecycleError,
    MCPProtocolError, NetworkError, ValidationError, SystemError,
    RecoveryError, ErrorSeverity, ErrorCategory
)
from digipal.core.error_handler import (
    ErrorHandler, RetryConfig, CircuitBreakerConfig, CircuitBreaker,
    with_error_handling, with_retry, with_circuit_breaker,
    error_handler, get_error_statistics
)
from digipal.storage.backup_recovery import (
    BackupRecoveryManager, BackupConfig, BackupMetadata
)
from digipal.ai.graceful_degradation import (
    AIServiceManager, FallbackResponseGenerator, DegradationLevel,
    ai_service_manager, with_ai_fallback
)
from digipal.core.models import DigiPal
from digipal.core.enums import EggType, LifeStage


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_digipal_exception_basic(self):
        """Test basic DigiPal exception functionality."""
        error = DigiPalException(
            "Test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            error_code="TEST_001"
        )
        
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "TEST_001"
        assert error.user_message == "A system error occurred. Please try again."
        assert isinstance(error.recovery_suggestions, list)
    
    def test_digipal_exception_with_context(self):
        """Test DigiPal exception with context."""
        context = {"user_id": "test_user", "operation": "test_op"}
        error = DigiPalException(
            "Test error",
            context=context,
            user_message="Custom user message",
            recovery_suggestions=["Try again", "Contact support"]
        )
        
        assert error.context == context
        assert error.user_message == "Custom user message"
        assert "Try again" in error.recovery_suggestions
    
    def test_digipal_exception_to_dict(self):
        """Test exception serialization."""
        error = DigiPalException(
            "Test error",
            category=ErrorCategory.AI_MODEL,
            severity=ErrorSeverity.MEDIUM,
            error_code="AI_001",
            context={"model": "test_model"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['message'] == "Test error"
        assert error_dict['category'] == "ai_model"
        assert error_dict['severity'] == "medium"
        assert error_dict['error_code'] == "AI_001"
        assert error_dict['context']['model'] == "test_model"
    
    def test_authentication_error(self):
        """Test authentication error specifics."""
        error = AuthenticationError("Invalid token")
        
        assert error.category == ErrorCategory.AUTHENTICATION
        assert error.severity == ErrorSeverity.HIGH
        assert "HuggingFace credentials" in error.recovery_suggestions[0]
    
    def test_storage_error(self):
        """Test storage error specifics."""
        error = StorageError("Database connection failed")
        
        assert error.category == ErrorCategory.STORAGE
        assert error.severity == ErrorSeverity.HIGH
        assert "disk space" in error.recovery_suggestions[0]
        error = StorageError("Database connection failed")
        
        assert error.category == ErrorCategory.STORAGE
        assert error.severity == ErrorSeverity.HIGH
        assert "disk space" in error.recovery_suggestions[0]
    
    def test_ai_model_error(self):
        """Test AI model error specifics."""
        error = AIModelError("Model loading failed")
        
        assert error.category == ErrorCategory.AI_MODEL
        assert error.severity == ErrorSeverity.MEDIUM
        assert "Try your request again" in error.recovery_suggestions[0]


class TestErrorHandler:
    """Test error handler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.error_handler = ErrorHandler()
    
    def test_handle_common_exceptions(self):
        """Test handling of common Python exceptions."""
        # Test ConnectionError
        conn_error = ConnectionError("Network unreachable")
        digipal_error = self.error_handler.handle_error(conn_error)
        
        assert isinstance(digipal_error, NetworkError)
        assert "Network error" in str(digipal_error)
        
        # Test FileNotFoundError
        file_error = FileNotFoundError("File not found")
        digipal_error = self.error_handler.handle_error(file_error)
        
        assert isinstance(digipal_error, StorageError)
        assert "File not found" in str(digipal_error)
        
        # Test ValueError
        val_error = ValueError("Invalid value")
        digipal_error = self.error_handler.handle_error(val_error)
        
        assert digipal_error.category == ErrorCategory.VALIDATION
    
    def test_handle_digipal_exception(self):
        """Test handling of existing DigiPal exceptions."""
        original_error = AIModelError("Model failed", context={"model": "test"})
        handled_error = self.error_handler.handle_error(
            original_error, 
            context={"additional": "info"}
        )
        
        assert handled_error is original_error
        assert handled_error.context["model"] == "test"
        assert handled_error.context["additional"] == "info"
    
    def test_log_error(self):
        """Test error logging functionality."""
        error = DigiPalException(
            "Test error",
            severity=ErrorSeverity.HIGH,
            error_code="TEST_001"
        )
        
        with patch('digipal.core.error_handler.logger') as mock_logger:
            self.error_handler.log_error(error)
            mock_logger.error.assert_called_once()
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and management."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = self.error_handler.get_circuit_breaker("test_service", config)
        
        assert isinstance(cb, CircuitBreaker)
        assert cb.config.failure_threshold == 2
        
        # Test reusing existing circuit breaker
        cb2 = self.error_handler.get_circuit_breaker("test_service")
        assert cb is cb2


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(self.config)
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        def success_func():
            return "success"
        
        result = self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling."""
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.failure_count == 1
        assert self.circuit_breaker.state == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.failure_count == 2
        assert self.circuit_breaker.state == "open"
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        # Force circuit to open
        self.circuit_breaker.failure_count = self.config.failure_threshold
        self.circuit_breaker.state = "open"
        self.circuit_breaker.last_failure_time = datetime.now()
        
        def any_func():
            return "should not execute"
        
        with pytest.raises(DigiPalException) as exc_info:
            self.circuit_breaker.call(any_func)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        # Force circuit to open
        self.circuit_breaker.failure_count = self.config.failure_threshold
        self.circuit_breaker.state = "open"
        self.circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        def success_func():
            return "recovered"
        
        # Should attempt recovery
        result = self.circuit_breaker.call(success_func)
        assert result == "recovered"
        assert self.circuit_breaker.state == "closed"
        assert self.circuit_breaker.failure_count == 0


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful function call on first attempt."""
        @with_retry(RetryConfig(max_attempts=3))
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_retry_success_after_failures(self):
        """Test successful function call after initial failures."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        def eventually_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        result = eventually_success_func()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhaustion(self):
        """Test retry exhaustion."""
        @with_retry(RetryConfig(max_attempts=2, base_delay=0.1))
        def always_fail_func():
            raise NetworkError("Always fails")
        
        with pytest.raises(NetworkError):
            always_fail_func()
    
    def test_retry_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, retry_on=[NetworkError]))
        def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            non_retryable_func()
        
        assert call_count == 1  # Should not retry


class TestErrorHandlingDecorator:
    """Test error handling decorator."""
    
    def test_error_handling_with_fallback(self):
        """Test error handling with fallback value."""
        @with_error_handling(fallback_value="fallback", raise_on_critical=False)
        def failing_func():
            raise ValueError("Test error")
        
        result = failing_func()
        assert result == "fallback"
    
    def test_error_handling_critical_error(self):
        """Test error handling with critical error."""
        @with_error_handling(raise_on_critical=True)
        def critical_error_func():
            raise DigiPalException(
                "Critical error",
                severity=ErrorSeverity.CRITICAL
            )
        
        with pytest.raises(DigiPalException):
            critical_error_func()
    
    def test_error_handling_logging(self):
        """Test error logging in decorator."""
        @with_error_handling(log_errors=True, fallback_value=None, raise_on_critical=False)
        def error_func():
            raise ValueError("Test error")
        
        with patch('digipal.core.error_handler.error_handler.log_error') as mock_log:
            with pytest.raises(DigiPalException):
                error_func()
            mock_log.assert_called_once()


class TestBackupRecoveryManager:
    """Test backup and recovery functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        # Create test database
        self._create_test_database()
        
        self.backup_manager = BackupRecoveryManager(
            str(self.db_path),
            str(self.backup_dir),
            BackupConfig(max_backups=3, verify_backups=True)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_database(self):
        """Create a test database with sample data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables (simplified schema)
        cursor.execute('''
            CREATE TABLE digipals (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT,
                data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO users VALUES ('user1', 'testuser')")
        cursor.execute("INSERT INTO digipals VALUES ('pet1', 'user1', 'TestPet', '{}')")
        
        conn.commit()
        conn.close()
    
    def test_create_full_backup(self):
        """Test creating a full database backup."""
        success, backup_id = self.backup_manager.create_backup("manual")
        
        assert success
        assert backup_id is not None
        assert backup_id in self.backup_manager.metadata
        
        # Verify backup file exists
        backup_metadata = self.backup_manager.metadata[backup_id]
        backup_path = Path(backup_metadata.file_path)
        assert backup_path.exists()
    
    def test_backup_verification(self):
        """Test backup verification."""
        success, backup_id = self.backup_manager.create_backup("manual")
        assert success
        
        backup_metadata = self.backup_manager.metadata[backup_id]
        backup_path = Path(backup_metadata.file_path)
        
        # Verify backup is valid
        assert self.backup_manager._verify_backup(backup_path)
        
        # Corrupt backup and verify it fails
        with open(backup_path, 'w') as f:
            f.write("corrupted data")
        
        assert not self.backup_manager._verify_backup(backup_path)
    
    def test_backup_restore(self):
        """Test backup restoration."""
        # Create backup
        success, backup_id = self.backup_manager.create_backup("manual")
        assert success
        
        # Modify original database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM digipals")
        conn.commit()
        conn.close()
        
        # Verify data is gone
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM digipals")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0
        
        # Restore backup
        success = self.backup_manager.restore_backup(backup_id)
        assert success
        
        # Verify data is restored
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM digipals")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1
    
    def test_backup_cleanup(self):
        """Test automatic backup cleanup."""
        # Create more backups than the limit
        backup_ids = []
        for i in range(5):
            success, backup_id = self.backup_manager.create_backup("manual")
            assert success
            backup_ids.append(backup_id)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Should only keep the most recent 3 backups
        assert len(self.backup_manager.metadata) == 3
        
        # Verify the oldest backups were removed
        remaining_ids = set(self.backup_manager.metadata.keys())
        expected_remaining = set(backup_ids[-3:])  # Last 3 backups
        assert remaining_ids == expected_remaining
    
    def test_backup_metadata_persistence(self):
        """Test backup metadata persistence."""
        # Create backup
        success, backup_id = self.backup_manager.create_backup("manual", description="Test backup")
        assert success
        
        # Create new manager instance (simulates restart)
        new_manager = BackupRecoveryManager(
            str(self.db_path),
            str(self.backup_dir),
            BackupConfig()
        )
        
        # Verify metadata was loaded
        assert backup_id in new_manager.metadata
        assert new_manager.metadata[backup_id].description == "Test backup"
    
    def test_backup_statistics(self):
        """Test backup statistics."""
        # Create some backups
        self.backup_manager.create_backup("manual")
        self.backup_manager.create_backup("automatic")
        
        stats = self.backup_manager.get_backup_statistics()
        
        assert stats['total_backups'] == 2
        assert stats['total_size_bytes'] > 0
        assert 'manual' in stats['backup_types']
        assert 'automatic' in stats['backup_types']
        assert stats['oldest_backup'] is not None
        assert stats['newest_backup'] is not None


class TestFallbackResponseGenerator:
    """Test fallback response generation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator = FallbackResponseGenerator()
        self.test_pet = DigiPal(
            user_id="test_user",
            name="TestPet",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD
        )
    
    def test_generate_basic_response(self):
        """Test basic fallback response generation."""
        response = self.generator.generate_fallback_response(
            "Hello", self.test_pet
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_command_specific_responses(self):
        """Test command-specific fallback responses."""
        response = self.generator.generate_fallback_response(
            "eat", self.test_pet, command="eat"
        )
        
        # Should contain eating-related response
        assert any(word in response.lower() for word in ['eat', 'food', 'hungry', 'good', 'thank'])
    
    def test_life_stage_appropriate_responses(self):
        """Test life stage appropriate responses."""
        baby_pet = DigiPal(
            user_id="test_user",
            name="BabyPet",
            egg_type=EggType.BLUE,
            life_stage=LifeStage.BABY
        )
        
        response = self.generator.generate_fallback_response(
            "Hello", baby_pet
        )
        
        # Baby responses should be simple
        assert any(word in response.lower() for word in ['goo', 'baby', 'mama', 'baba', '*'])
    
    def test_personality_modifiers(self):
        """Test personality-based response modifiers."""
        friendly_pet = DigiPal(
            user_id="test_user",
            name="FriendlyPet",
            egg_type=EggType.GREEN,
            life_stage=LifeStage.YOUNG_ADULT
        )
        friendly_pet.personality_traits = {'friendliness': 0.8}
        
        response = self.generator.generate_fallback_response(
            "Hello", friendly_pet
        )
        
        # Should have friendly modifier
        assert '*' in response or 'smile' in response.lower() or 'friendly' in response.lower()
    
    def test_emergency_mode_responses(self):
        """Test emergency mode responses."""
        response = self.generator.generate_fallback_response(
            "Hello", self.test_pet, degradation_level=DegradationLevel.EMERGENCY_MODE
        )
        
        # Emergency responses should be minimal
        assert len(response) < 20
        assert response in ["Hi!", "*baby*", "*egg*", "Hey.", "Hello!", "Greetings.", "Hello, friend."]


class TestAIServiceManager:
    """Test AI service management and degradation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service_manager = AIServiceManager()
        self.test_pet = DigiPal(
            user_id="test_user",
            name="TestPet",
            egg_type=EggType.RED,
            life_stage=LifeStage.YOUNG_ADULT
        )
    
    def test_service_status_tracking(self):
        """Test service status tracking."""
        status = self.service_manager.get_service_status()
        
        assert 'services' in status
        assert 'degradation_level' in status
        assert 'circuit_breakers' in status
        
        # All services should start as available
        assert all(status['services'].values())
        assert status['degradation_level'] == DegradationLevel.FULL_SERVICE.value
    
    def test_service_failure_handling(self):
        """Test service failure handling."""
        def failing_service():
            raise AIModelError("Service failed")
        
        def fallback_service():
            return "fallback result"
        
        result = self.service_manager.call_ai_service(
            "test_service", failing_service, fallback_service
        )
        
        assert result == "fallback result"
        assert not self.service_manager.service_status.get("test_service", True)
    
    def test_degradation_level_updates(self):
        """Test degradation level updates based on service availability."""
        # Simulate service failures
        self.service_manager.service_status['language_model'] = False
        self.service_manager.service_status['speech_processing'] = False
        self.service_manager._update_degradation_level()
        
        # Should be in reduced functionality mode
        assert self.service_manager.current_degradation_level in [
            DegradationLevel.BASIC_RESPONSES,
            DegradationLevel.MINIMAL_FUNCTION
        ]
    
    def test_degraded_response_generation(self):
        """Test degraded response generation."""
        # Force degradation
        self.service_manager.current_degradation_level = DegradationLevel.BASIC_RESPONSES
        
        interaction = self.service_manager.generate_degraded_response(
            "Hello", self.test_pet
        )
        
        assert interaction.success
        assert len(interaction.pet_response) > 0
        assert "AI services are currently limited" in interaction.pet_response
    
    def test_force_service_recovery(self):
        """Test forced service recovery."""
        # Simulate service failure
        service_name = "test_service"
        circuit_breaker = self.service_manager.circuit_breakers.get(service_name)
        if not circuit_breaker:
            from digipal.core.error_handler import CircuitBreakerConfig, CircuitBreaker
            circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
            self.service_manager.circuit_breakers[service_name] = circuit_breaker
        
        circuit_breaker.state = "open"
        circuit_breaker.failure_count = 5
        
        # Force recovery
        self.service_manager.force_service_recovery(service_name)
        
        assert circuit_breaker.state == "half-open"
        assert circuit_breaker.failure_count == 0


class TestIntegrationScenarios:
    """Test integrated error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        # Create test database
        self._create_test_database()
        
        self.backup_manager = BackupRecoveryManager(
            str(self.db_path),
            str(self.backup_dir)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_database(self):
        """Create a test database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE digipals (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT
            )
        ''')
        
        cursor.execute("INSERT INTO digipals VALUES ('pet1', 'user1', 'TestPet')")
        conn.commit()
        conn.close()
    
    def test_database_corruption_recovery(self):
        """Test recovery from database corruption."""
        # Create backup
        success, backup_id = self.backup_manager.create_backup("pre_operation")
        assert success
        
        # Simulate database corruption
        with open(self.db_path, 'w') as f:
            f.write("corrupted data")
        
        # Attempt to use corrupted database
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("SELECT * FROM digipals")
            conn.close()
            assert False, "Should have failed with corrupted database"
        except sqlite3.DatabaseError:
            pass  # Expected
        
        # Restore from backup
        success = self.backup_manager.restore_backup(backup_id)
        assert success
        
        # Verify database is working again
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM digipals")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1
    
    def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""
        service_manager = AIServiceManager()
        
        # Simulate multiple service failures
        def failing_language_model():
            raise AIModelError("Language model failed")
        
        def failing_speech_processor():
            raise AIModelError("Speech processor failed")
        
        def failing_image_generator():
            raise AIModelError("Image generator failed")
        
        # All services fail
        with pytest.raises(AIModelError):
            service_manager.call_ai_service("language_model", failing_language_model)
        
        with pytest.raises(AIModelError):
            service_manager.call_ai_service("speech_processing", failing_speech_processor)
        
        with pytest.raises(AIModelError):
            service_manager.call_ai_service("image_generation", failing_image_generator)
        
        # Should be in emergency mode
        assert service_manager.current_degradation_level == DegradationLevel.EMERGENCY_MODE
        
        # Should still be able to generate basic responses
        test_pet = DigiPal(
            user_id="test_user",
            name="TestPet",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD
        )
        
        interaction = service_manager.generate_degraded_response("Hello", test_pet)
        assert interaction.success
        assert len(interaction.pet_response) > 0
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        # Generate various errors
        errors = [
            AIModelError("Model error 1"),
            StorageError("Storage error 1"),
            AIModelError("Model error 2"),
            NetworkError("Network error 1")
        ]
        
        for error in errors:
            error_handler.log_error(error)
        
        stats = get_error_statistics()
        
        assert 'error_counts' in stats
        assert 'last_errors' in stats
        
        # Should track error frequencies
        ai_errors = sum(1 for key in stats['error_counts'].keys() if 'ai_model' in key)
        assert ai_errors >= 2  # At least 2 AI model errors


if __name__ == "__main__":
    pytest.main([__file__])


class TestRecoveryStrategies:
    """Test recovery strategies for different error types."""
    
    def setup_method(self):
        """Set up test environment."""
        from digipal.core.recovery_strategies import (
            StorageRecoveryStrategy, AIModelRecoveryStrategy, 
            NetworkRecoveryStrategy, AuthenticationRecoveryStrategy,
            PetLifecycleRecoveryStrategy, SystemRecoveryOrchestrator
        )
        
        self.storage_recovery = StorageRecoveryStrategy()
        self.ai_recovery = AIModelRecoveryStrategy()
        self.network_recovery = NetworkRecoveryStrategy()
        self.auth_recovery = AuthenticationRecoveryStrategy()
        self.pet_recovery = PetLifecycleRecoveryStrategy()
        self.orchestrator = SystemRecoveryOrchestrator()
    
    def test_storage_corruption_recovery(self):
        """Test storage corruption recovery strategy."""
        error = StorageError(
            "Database corruption detected",
            context={'db_path': '/test/path.db'},
            error_code='STOR_CORRUPT_001'
        )
        
        # Test without backup manager (should fail gracefully)
        result = self.storage_recovery.recover_corrupted_database(error)
        assert result is False
    
    def test_disk_space_recovery(self):
        """Test disk space recovery strategy."""
        error = StorageError(
            "No space left on device",
            context={'available_space': 0},
            error_code='STOR_SPACE_001'
        )
        
        result = self.storage_recovery.recover_disk_space_issue(error)
        assert result is True  # Should succeed in cleaning up
    
    def test_permission_error_recovery(self):
        """Test permission error recovery strategy."""
        error = StorageError(
            "Permission denied",
            context={'file_path': '/restricted/path'},
            error_code='STOR_PERM_001'
        )
        
        result = self.storage_recovery.recover_permission_error(error)
        assert result is True
        assert 'alternative_storage_path' in error.context
    
    def test_ai_model_loading_recovery(self):
        """Test AI model loading recovery strategy."""
        error = AIModelError(
            "Model loading failed",
            context={'model_name': 'qwen3-0.6b'},
            error_code='AI_LOAD_001'
        )
        
        result = self.ai_recovery.recover_model_loading_failure(error)
        assert result is True
        assert error.context['use_reduced_precision'] is True
        assert error.context['use_cpu_only'] is True
    
    def test_ai_inference_recovery(self):
        """Test AI model inference recovery strategy."""
        error = AIModelError(
            "Inference failed",
            context={'input_length': 1000},
            error_code='AI_INFER_001'
        )
        
        result = self.ai_recovery.recover_model_inference_failure(error)
        assert result is True
        assert error.context['use_fallback_responses'] is True
        assert error.context['degradation_level'] == 'basic_responses'
    
    def test_memory_error_recovery(self):
        """Test memory error recovery strategy."""
        error = AIModelError(
            "Out of memory",
            context={'memory_usage': '8GB'},
            error_code='AI_MEM_001'
        )
        
        result = self.ai_recovery.recover_memory_error(error)
        assert result is True
        assert error.context['reduce_batch_size'] is True
        assert error.context['use_fp16'] is True
        assert error.context['offload_to_cpu'] is True
    
    def test_network_timeout_recovery(self):
        """Test network timeout recovery strategy."""
        error = NetworkError(
            "Connection timeout",
            context={'endpoint': 'https://api.example.com'},
            error_code='NET_TIMEOUT_001'
        )
        
        result = self.network_recovery.recover_connection_timeout(error)
        assert result is True
        assert error.context['enable_offline_mode'] is True
        assert error.context['use_cached_data'] is True
    
    def test_dns_resolution_recovery(self):
        """Test DNS resolution recovery strategy."""
        error = NetworkError(
            "DNS resolution failed",
            context={'hostname': 'api.example.com'},
            error_code='NET_DNS_001'
        )
        
        result = self.network_recovery.recover_dns_resolution_failure(error)
        assert result is True
        assert 'alternative_dns_servers' in error.context
        assert error.context['use_ip_addresses'] is True
    
    def test_rate_limit_recovery(self):
        """Test rate limit recovery strategy."""
        error = NetworkError(
            "Rate limit exceeded",
            context={'requests_per_minute': 100},
            error_code='NET_RATE_001'
        )
        
        result = self.network_recovery.recover_rate_limit_error(error)
        assert result is True
        assert error.context['implement_backoff'] is True
        assert error.context['use_cached_responses'] is True
    
    def test_token_expiry_recovery(self):
        """Test token expiry recovery strategy."""
        error = AuthenticationError(
            "Token expired",
            context={'token_age': 3600},
            error_code='AUTH_TOKEN_001'
        )
        
        result = self.auth_recovery.recover_token_expiry(error)
        assert result is True
        assert error.context['use_offline_auth'] is True
        assert error.context['extend_session_timeout'] is True
    
    def test_invalid_credentials_recovery(self):
        """Test invalid credentials recovery strategy."""
        error = AuthenticationError(
            "Invalid credentials",
            context={'username': 'test_user'},
            error_code='AUTH_CRED_001'
        )
        
        result = self.auth_recovery.recover_invalid_credentials(error)
        assert result is True
        assert error.context['enable_guest_mode'] is True
        assert error.context['limited_functionality'] is True
    
    def test_pet_evolution_recovery(self):
        """Test pet evolution recovery strategy."""
        error = PetLifecycleError(
            "Evolution failed",
            context={'pet_id': 'test_pet_123', 'evolution_stage': 'child_to_teen'},
            error_code='PET_EVOL_001'
        )
        
        result = self.pet_recovery.recover_evolution_failure(error)
        assert result is True
        assert error.context['reset_evolution_state'] is True
        assert error.context['use_safe_evolution'] is True
    
    def test_comprehensive_recovery_orchestration(self):
        """Test comprehensive recovery orchestration."""
        error = StorageError(
            "Critical storage failure",
            context={'operation': 'save_pet'},
            error_code='STOR_CRIT_001'
        )
        
        recovery_result = self.orchestrator.execute_comprehensive_recovery(error)
        
        assert isinstance(recovery_result.success, bool)
        assert recovery_result.strategy_used == "comprehensive_recovery"
        assert isinstance(recovery_result.recovery_time_seconds, float)
        assert recovery_result.recovery_time_seconds >= 0
    
    def test_recovery_recommendations(self):
        """Test recovery recommendations generation."""
        error = AIModelError(
            "Model loading failed",
            context={'model_name': 'qwen3-0.6b'},
            error_code='AI_LOAD_001'
        )
        
        recommendations = self.orchestrator.get_recovery_recommendations(error)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("memory" in rec.lower() for rec in recommendations)
        assert any("model" in rec.lower() for rec in recommendations)


class TestUserErrorMessages:
    """Test user-friendly error messages and recovery guides."""
    
    def setup_method(self):
        """Set up test environment."""
        from digipal.core.user_error_messages import (
            UserErrorMessageGenerator, MessageTone, get_user_friendly_error_message,
            get_recovery_guide
        )
        
        self.message_generator = UserErrorMessageGenerator()
        self.MessageTone = MessageTone
    
    def test_friendly_error_message_generation(self):
        """Test friendly error message generation."""
        error = AuthenticationError(
            "Invalid token",
            error_code='AUTH_001'
        )
        
        message = self.message_generator.generate_user_message(
            error, 
            tone=self.MessageTone.FRIENDLY
        )
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "🔐" in message or "login" in message.lower()
    
    def test_professional_error_message_generation(self):
        """Test professional error message generation."""
        error = StorageError(
            "Database connection failed",
            error_code='STOR_001'
        )
        
        message = self.message_generator.generate_user_message(
            error,
            tone=self.MessageTone.PROFESSIONAL
        )
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "storage" in message.lower() or "data" in message.lower()
    
    def test_contextual_error_messages(self):
        """Test contextual error messages."""
        error = AIModelError(
            "Model loading failed",
            error_code='AI_001'
        )
        
        # Test first-time user context
        message = self.message_generator.generate_user_message(
            error,
            user_context={'user_state': 'first_time_user'}
        )
        
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_recovery_guide_generation(self):
        """Test recovery guide generation."""
        error = NetworkError(
            "Connection failed",
            error_code='NET_001'
        )
        
        guide = self.message_generator.get_recovery_guide(error, max_steps=3)
        
        assert isinstance(guide, list)
        assert len(guide) <= 3
        
        for step in guide:
            assert 'step' in step
            assert 'title' in step
            assert 'description' in step
            assert 'action' in step
            assert 'difficulty' in step
    
    def test_difficulty_filtered_recovery_guide(self):
        """Test difficulty-filtered recovery guide."""
        error = StorageError(
            "Permission denied",
            error_code='STOR_002'
        )
        
        easy_guide = self.message_generator.get_recovery_guide(
            error, 
            difficulty_filter='easy'
        )
        
        assert isinstance(easy_guide, list)
        for step in easy_guide:
            assert step['difficulty'] == 'easy'
    
    def test_progress_message_generation(self):
        """Test progress message generation."""
        recovery_step = {
            'step': 2,
            'title': 'Check Internet Connection',
            'description': 'Verify network connectivity',
            'action': 'check_connection',
            'difficulty': 'easy'
        }
        
        progress_msg = self.message_generator.generate_progress_message(recovery_step)
        
        assert isinstance(progress_msg, str)
        assert "Step 2" in progress_msg or "Check Internet Connection" in progress_msg
    
    def test_success_message_generation(self):
        """Test success message generation."""
        success_msg = self.message_generator.generate_success_message(
            'authentication', 
            'credential_refresh'
        )
        
        assert isinstance(success_msg, str)
        assert len(success_msg) > 0
        assert "🎉" in success_msg or "logged in" in success_msg.lower()
    
    def test_global_message_functions(self):
        """Test global message functions."""
        error = SpeechProcessingError(
            "Audio processing failed",
            error_code='SPEECH_001'
        )
        
        # Test global function
        message = get_user_friendly_error_message(error)
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Test global recovery guide function
        guide = get_recovery_guide(error)
        assert isinstance(guide, list)


class TestErrorPatternAnalysis:
    """Test error pattern analysis and detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.error_handler = ErrorHandler()
    
    def test_error_pattern_tracking(self):
        """Test error pattern tracking."""
        # Generate multiple errors of the same type
        for i in range(5):
            error = NetworkError(f"Connection failed {i}", error_code='NET_001')
            self.error_handler.handle_error(error)
        
        # Check error patterns
        assert 'network:NET_001' in self.error_handler.error_patterns
        assert len(self.error_handler.error_patterns['network:NET_001']) == 5
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        # Generate errors
        for i in range(10):
            error = StorageError(f"Storage error {i}", error_code='STOR_001')
            self.error_handler.handle_error(error)
        
        error_rate = self.error_handler.get_error_rate('storage', window_minutes=5)
        assert error_rate == 2.0  # 10 errors / 5 minutes
    
    def test_error_storm_detection(self):
        """Test error storm detection."""
        # Generate many errors quickly
        for i in range(60):  # More than 10 per minute
            error = SystemError(f"System error {i}", error_code='SYS_001')
            self.error_handler.handle_error(error)
        
        is_storm = self.error_handler.is_error_storm_detected()
        assert is_storm is True
    
    def test_most_frequent_errors(self):
        """Test most frequent errors tracking."""
        # Generate different types of errors
        for i in range(10):
            self.error_handler.handle_error(NetworkError(f"Net {i}", error_code='NET_001'))
        
        for i in range(5):
            self.error_handler.handle_error(StorageError(f"Storage {i}", error_code='STOR_001'))
        
        for i in range(15):
            self.error_handler.handle_error(AIModelError(f"AI {i}", error_code='AI_001'))
        
        frequent_errors = self.error_handler.get_most_frequent_errors(limit=3)
        
        assert len(frequent_errors) == 3
        assert frequent_errors[0][0] == 'ai_model:AI_001'  # Most frequent
        assert frequent_errors[0][1] == 15
        assert frequent_errors[1][0] == 'network:NET_001'  # Second most frequent
        assert frequent_errors[1][1] == 10


class TestBackupRecoveryIntegration:
    """Test backup and recovery integration with error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        # Create test database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE digipals (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT,
                data TEXT
            )
        """)
        conn.execute("INSERT INTO digipals VALUES ('pet1', 'user1', 'TestPal', '{}')")
        conn.commit()
        conn.close()
        
        from digipal.storage.backup_recovery import BackupRecoveryManager, BackupConfig
        self.backup_manager = BackupRecoveryManager(
            str(self.db_path),
            str(self.backup_dir),
            BackupConfig(max_backups=5)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_pre_operation_backup_creation(self):
        """Test pre-operation backup creation."""
        backup_id = self.backup_manager.create_pre_operation_backup(
            "test_operation",
            {"test": "context"}
        )
        
        assert backup_id is not None
        assert backup_id in self.backup_manager.metadata
        
        metadata = self.backup_manager.metadata[backup_id]
        assert metadata.backup_type == "pre_operation"
        assert "test_operation" in metadata.description
    
    def test_automatic_backup_on_critical_error(self):
        """Test automatic backup creation on critical errors."""
        from digipal.core.recovery_strategies import SystemRecoveryOrchestrator
        
        orchestrator = SystemRecoveryOrchestrator(self.backup_manager)
        
        critical_error = StorageError(
            "Critical database corruption",
            context={'operation': 'save_pet'},
            error_code='STOR_CRIT_001'
        )
        critical_error.severity = ErrorSeverity.CRITICAL
        
        # Execute recovery (should create backup)
        result = orchestrator.execute_comprehensive_recovery(critical_error)
        
        # Check if backup was created
        backups = self.backup_manager.list_backups(backup_type="pre_operation")
        assert len(backups) > 0
        
        # Find the error recovery backup
        recovery_backup = None
        for backup in backups:
            if "error_recovery" in backup.description:
                recovery_backup = backup
                break
        
        assert recovery_backup is not None
    
    def test_backup_restoration_on_storage_error(self):
        """Test backup restoration during storage error recovery."""
        # Create a backup first
        success, backup_id = self.backup_manager.create_backup("manual", description="Test backup")
        assert success
        
        # Simulate database corruption
        with open(self.db_path, 'w') as f:
            f.write("corrupted data")
        
        # Test restoration
        restore_success = self.backup_manager.restore_backup(backup_id)
        assert restore_success
        
        # Verify database is restored
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM digipals")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1  # Original test data should be restored


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        from digipal.core.error_handler import CircuitBreakerConfig
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            expected_exception=AIModelError
        )
        
        self.error_handler = ErrorHandler()
    
    def test_circuit_breaker_opening(self):
        """Test circuit breaker opening after failures."""
        circuit_breaker = self.error_handler.get_circuit_breaker("test_service", self.config)
        
        def failing_function():
            raise AIModelError("Service unavailable")
        
        # Cause failures to open circuit
        for i in range(3):
            with pytest.raises(AIModelError):
                circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == "open"
        assert circuit_breaker.failure_count == 3
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        circuit_breaker = self.error_handler.get_circuit_breaker("test_service", self.config)
        
        def failing_function():
            raise AIModelError("Service unavailable")
        
        def working_function():
            return "success"
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(AIModelError):
                circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Next call should attempt recovery
        result = circuit_breaker.call(working_function)
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0


class TestGracefulDegradationIntegration:
    """Test graceful degradation integration with error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        from digipal.ai.graceful_degradation import AIServiceManager
        from digipal.core.models import DigiPal
        from digipal.core.enums import EggType, LifeStage
        
        self.ai_service_manager = AIServiceManager()
        
        # Create test DigiPal
        self.test_pet = DigiPal(
            id="test_pet",
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            generation=1,
            hp=100, mp=50, offense=30, defense=25, speed=20, brains=15,
            discipline=50, happiness=75, weight=10, care_mistakes=0, energy=80,
            birth_time=datetime.now(),
            last_interaction=datetime.now(),
            evolution_timer=0.0,
            conversation_history=[],
            learned_commands=set(),
            personality_traits={'friendly': 0.8, 'curious': 0.6},
            current_image_path="",
            image_generation_prompt=""
        )
    
    def test_service_degradation_on_failures(self):
        """Test service degradation when AI services fail."""
        def failing_language_model(*args, **kwargs):
            raise AIModelError("Language model unavailable")
        
        def failing_speech_processor(*args, **kwargs):
            raise AIModelError("Speech processor unavailable")
        
        def failing_image_generator(*args, **kwargs):
            raise AIModelError("Image generator unavailable")
        
        # Simulate service failures
        try:
            self.ai_service_manager.call_ai_service("language_model", failing_language_model)
        except AIModelError:
            pass
        
        try:
            self.ai_service_manager.call_ai_service("speech_processing", failing_speech_processor)
        except AIModelError:
            pass
        
        try:
            self.ai_service_manager.call_ai_service("image_generation", failing_image_generator)
        except AIModelError:
            pass
        
        # Check degradation level
        status = self.ai_service_manager.get_service_status()
        assert status['degradation_level'] == 'emergency_mode'
        assert not any(status['services'].values())  # All services should be marked as down
    
    def test_fallback_response_generation(self):
        """Test fallback response generation during degradation."""
        # Force degradation
        self.ai_service_manager.service_status = {
            'language_model': False,
            'speech_processing': False,
            'image_generation': False
        }
        self.ai_service_manager._update_degradation_level()
        
        # Generate degraded response
        interaction = self.ai_service_manager.generate_degraded_response(
            "Hello DigiPal!",
            self.test_pet,
            "greeting"
        )
        
        assert interaction.success is True
        assert len(interaction.pet_response) > 0
        assert interaction.pet_response != ""
    
    def test_service_recovery_forcing(self):
        """Test forcing service recovery."""
        # Mark service as failed
        self.ai_service_manager.service_status['language_model'] = False
        
        # Force recovery
        self.ai_service_manager.force_service_recovery('language_model')
        
        # Check circuit breaker state
        cb = self.ai_service_manager.circuit_breakers['language_model']
        assert cb.state == "half-open"
        assert cb.failure_count == 0


class TestEndToEndErrorScenarios:
    """Test complete end-to-end error scenarios."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        import tempfile
        
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        # Initialize all components
        from digipal.storage.backup_recovery import BackupRecoveryManager
        from digipal.core.recovery_strategies import initialize_system_recovery
        
        self.backup_manager = BackupRecoveryManager(
            str(self.db_path),
            str(self.backup_dir)
        )
        
        initialize_system_recovery(self.backup_manager)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_storage_failure_recovery(self):
        """Test complete storage failure and recovery scenario."""
        from digipal.core.recovery_strategies import get_system_recovery_orchestrator
        
        # Create initial database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE digipals (id TEXT, data TEXT)")
        conn.execute("INSERT INTO digipals VALUES ('pet1', 'test_data')")
        conn.commit()
        conn.close()
        
        # Create backup
        success, backup_id = self.backup_manager.create_backup("manual")
        assert success
        
        # Simulate complete storage failure
        self.db_path.unlink()  # Delete database
        
        storage_error = StorageError(
            "Database file not found",
            context={'db_path': str(self.db_path)},
            error_code='STOR_MISSING_001'
        )
        
        # Execute recovery
        orchestrator = get_system_recovery_orchestrator()
        if orchestrator:
            recovery_result = orchestrator.execute_comprehensive_recovery(storage_error)
            
            # Recovery might not fully succeed without proper integration,
            # but should provide recovery context
            assert isinstance(recovery_result.success, bool)
            assert recovery_result.strategy_used == "comprehensive_recovery"
    
    def test_cascading_failure_scenario(self):
        """Test cascading failure scenario across multiple systems."""
        from digipal.core.recovery_strategies import get_system_recovery_orchestrator
        
        # Simulate network failure leading to auth failure leading to storage issues
        network_error = NetworkError(
            "Connection timeout",
            context={'service': 'huggingface_api'},
            error_code='NET_TIMEOUT_001'
        )
        
        auth_error = AuthenticationError(
            "Token validation failed due to network",
            context={'caused_by': 'network_error'},
            error_code='AUTH_NET_001'
        )
        
        storage_error = StorageError(
            "Cannot save without authentication",
            context={'caused_by': 'auth_error'},
            error_code='STOR_AUTH_001'
        )
        
        orchestrator = get_system_recovery_orchestrator()
        if orchestrator:
            # Test recovery for each error in the cascade
            for error in [network_error, auth_error, storage_error]:
                recovery_result = orchestrator.execute_comprehensive_recovery(error)
                assert isinstance(recovery_result.success, bool)
                assert recovery_result.recovery_time_seconds >= 0
    
    def test_error_message_user_experience(self):
        """Test complete user experience for error messages."""
        from digipal.core.user_error_messages import get_user_friendly_error_message, get_recovery_guide
        
        # Test various error scenarios
        errors = [
            AuthenticationError("Invalid token", error_code='AUTH_001'),
            StorageError("Database locked", error_code='STOR_001'),
            AIModelError("Model loading failed", error_code='AI_001'),
            NetworkError("DNS resolution failed", error_code='NET_001'),
            SpeechProcessingError("Audio quality poor", error_code='SPEECH_001')
        ]
        
        for error in errors:
            # Test message generation
            message = get_user_friendly_error_message(error)
            assert isinstance(message, str)
            assert len(message) > 0
            
            # Test recovery guide
            guide = get_recovery_guide(error)
            assert isinstance(guide, list)
            assert len(guide) > 0
            
            # Verify guide structure
            for step in guide:
                assert 'step' in step
                assert 'title' in step
                assert 'description' in step
                assert 'action' in step
                assert 'difficulty' in step
    
    def test_performance_under_error_load(self):
        """Test system performance under high error load."""
        import time
        
        start_time = time.time()
        
        # Generate many errors quickly
        for i in range(100):
            error = SystemError(f"Load test error {i}", error_code=f'LOAD_{i:03d}')
            error_handler.handle_error(error)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle 100 errors in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        # Check error statistics
        stats = get_error_statistics()
        assert stats['error_counts']
        assert len(stats['error_counts']) > 0