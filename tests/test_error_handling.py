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