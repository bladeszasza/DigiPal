"""
Recovery strategies for DigiPal error handling system.

This module implements specific recovery strategies for different types of errors,
providing automated recovery mechanisms and fallback procedures.
"""

import logging
import time
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass

from .exceptions import (
    DigiPalException, StorageError, AIModelError, NetworkError,
    AuthenticationError, ImageGenerationError, SpeechProcessingError,
    PetLifecycleError, MCPProtocolError, RecoveryError, ErrorSeverity
)
from .error_handler import recovery_manager
from ..storage.backup_recovery import BackupRecoveryManager

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: str
    message: str
    context: Dict[str, Any]
    recovery_time_seconds: float


class StorageRecoveryStrategy:
    """Recovery strategies for storage-related errors."""
    
    def __init__(self, backup_manager: Optional[BackupRecoveryManager] = None):
        """Initialize storage recovery strategy."""
        self.backup_manager = backup_manager
    
    def recover_corrupted_database(self, error: StorageError) -> bool:
        """
        Recover from database corruption.
        
        Args:
            error: Storage error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting database corruption recovery")
            
            if not self.backup_manager:
                logger.error("No backup manager available for recovery")
                return False
            
            # Get the most recent backup
            backups = self.backup_manager.list_backups()
            if not backups:
                logger.error("No backups available for recovery")
                return False
            
            latest_backup = backups[0]  # Already sorted by timestamp
            
            # Attempt to restore from backup
            success = self.backup_manager.restore_backup(latest_backup.backup_id)
            
            if success:
                logger.info(f"Database recovered from backup: {latest_backup.backup_id}")
                return True
            else:
                logger.error("Failed to restore from backup")
                return False
                
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
    
    def recover_disk_space_issue(self, error: StorageError) -> bool:
        """
        Recover from disk space issues.
        
        Args:
            error: Storage error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting disk space recovery")
            
            # Clean up old backups
            if self.backup_manager:
                old_backups = self.backup_manager.list_backups()
                # Keep only the 3 most recent backups
                backups_to_delete = old_backups[3:]
                
                for backup in backups_to_delete:
                    self.backup_manager.delete_backup(backup.backup_id)
                    logger.info(f"Deleted old backup: {backup.backup_id}")
            
            # Clean up temporary files
            temp_dirs = [Path("/tmp"), Path.cwd() / "temp", Path.cwd() / "cache"]
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for temp_file in temp_dir.glob("digipal_*"):
                        try:
                            if temp_file.is_file():
                                temp_file.unlink()
                            elif temp_file.is_dir():
                                shutil.rmtree(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            logger.info("Disk space cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Disk space recovery failed: {e}")
            return False
    
    def recover_permission_error(self, error: StorageError) -> bool:
        """
        Recover from permission errors.
        
        Args:
            error: Storage error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting permission error recovery")
            
            # Try to create alternative storage location
            alternative_paths = [
                Path.home() / ".digipal" / "data",
                Path("/tmp") / "digipal_data",
                Path.cwd() / "digipal_data_alt"
            ]
            
            for alt_path in alternative_paths:
                try:
                    alt_path.mkdir(parents=True, exist_ok=True)
                    
                    # Test write access
                    test_file = alt_path / "test_write.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                    
                    logger.info(f"Alternative storage location available: {alt_path}")
                    # Store the alternative path in error context for later use
                    error.context['alternative_storage_path'] = str(alt_path)
                    return True
                    
                except Exception as e:
                    logger.debug(f"Alternative path {alt_path} not accessible: {e}")
                    continue
            
            logger.error("No alternative storage locations available")
            return False
            
        except Exception as e:
            logger.error(f"Permission error recovery failed: {e}")
            return False


class AIModelRecoveryStrategy:
    """Recovery strategies for AI model-related errors."""
    
    def __init__(self):
        """Initialize AI model recovery strategy."""
        self.model_fallback_hierarchy = {
            'language_model': ['qwen3-0.6b', 'simple_responses', 'static_responses'],
            'speech_processing': ['kyutai', 'basic_speech', 'text_only'],
            'image_generation': ['flux', 'stable_diffusion', 'default_images']
        }
    
    def recover_model_loading_failure(self, error: AIModelError) -> bool:
        """
        Recover from model loading failures.
        
        Args:
            error: AI model error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting AI model loading recovery")
            
            # Clear model cache
            import gc
            import torch
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            # Try loading with reduced precision
            error.context['use_reduced_precision'] = True
            error.context['use_cpu_only'] = True
            
            logger.info("Set fallback options for model loading")
            return True
            
        except Exception as e:
            logger.error(f"Model loading recovery failed: {e}")
            return False
    
    def recover_model_inference_failure(self, error: AIModelError) -> bool:
        """
        Recover from model inference failures.
        
        Args:
            error: AI model error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting AI model inference recovery")
            
            # Switch to fallback response mode
            error.context['use_fallback_responses'] = True
            error.context['degradation_level'] = 'basic_responses'
            
            logger.info("Switched to fallback response mode")
            return True
            
        except Exception as e:
            logger.error(f"Model inference recovery failed: {e}")
            return False
    
    def recover_memory_error(self, error: AIModelError) -> bool:
        """
        Recover from memory-related errors.
        
        Args:
            error: AI model error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting memory error recovery")
            
            import gc
            import torch
            
            # Aggressive memory cleanup
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reduce batch sizes and model precision
            error.context['reduce_batch_size'] = True
            error.context['use_fp16'] = True
            error.context['offload_to_cpu'] = True
            
            logger.info("Applied memory optimization settings")
            return True
            
        except Exception as e:
            logger.error(f"Memory error recovery failed: {e}")
            return False


class NetworkRecoveryStrategy:
    """Recovery strategies for network-related errors."""
    
    def __init__(self):
        """Initialize network recovery strategy."""
        self.retry_delays = [1, 2, 5, 10, 30]  # Progressive delays in seconds
    
    def recover_connection_timeout(self, error: NetworkError) -> bool:
        """
        Recover from connection timeout errors.
        
        Args:
            error: Network error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting connection timeout recovery")
            
            # Switch to offline mode
            error.context['enable_offline_mode'] = True
            error.context['use_cached_data'] = True
            
            logger.info("Switched to offline mode")
            return True
            
        except Exception as e:
            logger.error(f"Connection timeout recovery failed: {e}")
            return False
    
    def recover_dns_resolution_failure(self, error: NetworkError) -> bool:
        """
        Recover from DNS resolution failures.
        
        Args:
            error: Network error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting DNS resolution recovery")
            
            # Try alternative DNS servers or cached endpoints
            alternative_endpoints = [
                "8.8.8.8",  # Google DNS
                "1.1.1.1",  # Cloudflare DNS
                "208.67.222.222"  # OpenDNS
            ]
            
            error.context['alternative_dns_servers'] = alternative_endpoints
            error.context['use_ip_addresses'] = True
            
            logger.info("Set alternative DNS configuration")
            return True
            
        except Exception as e:
            logger.error(f"DNS resolution recovery failed: {e}")
            return False
    
    def recover_rate_limit_error(self, error: NetworkError) -> bool:
        """
        Recover from rate limiting errors.
        
        Args:
            error: Network error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting rate limit recovery")
            
            # Implement exponential backoff
            error.context['implement_backoff'] = True
            error.context['backoff_multiplier'] = 2.0
            error.context['max_backoff_seconds'] = 300
            
            # Switch to cached responses temporarily
            error.context['use_cached_responses'] = True
            error.context['cache_duration_hours'] = 1
            
            logger.info("Applied rate limiting recovery measures")
            return True
            
        except Exception as e:
            logger.error(f"Rate limit recovery failed: {e}")
            return False


class AuthenticationRecoveryStrategy:
    """Recovery strategies for authentication-related errors."""
    
    def recover_token_expiry(self, error: AuthenticationError) -> bool:
        """
        Recover from token expiry errors.
        
        Args:
            error: Authentication error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting token expiry recovery")
            
            # Switch to offline mode with cached authentication
            error.context['use_offline_auth'] = True
            error.context['extend_session_timeout'] = True
            
            logger.info("Switched to offline authentication mode")
            return True
            
        except Exception as e:
            logger.error(f"Token expiry recovery failed: {e}")
            return False
    
    def recover_invalid_credentials(self, error: AuthenticationError) -> bool:
        """
        Recover from invalid credentials errors.
        
        Args:
            error: Authentication error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting invalid credentials recovery")
            
            # Enable guest mode
            error.context['enable_guest_mode'] = True
            error.context['limited_functionality'] = True
            
            logger.info("Enabled guest mode for limited functionality")
            return True
            
        except Exception as e:
            logger.error(f"Invalid credentials recovery failed: {e}")
            return False


class PetLifecycleRecoveryStrategy:
    """Recovery strategies for pet lifecycle errors."""
    
    def __init__(self, backup_manager: Optional[BackupRecoveryManager] = None):
        """Initialize pet lifecycle recovery strategy."""
        self.backup_manager = backup_manager
    
    def recover_corrupted_pet_data(self, error: PetLifecycleError) -> bool:
        """
        Recover from corrupted pet data.
        
        Args:
            error: Pet lifecycle error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting corrupted pet data recovery")
            
            if not self.backup_manager:
                logger.error("No backup manager available for pet recovery")
                return False
            
            # Try to restore pet from backup
            pet_id = error.context.get('pet_id')
            user_id = error.context.get('user_id')
            
            if pet_id:
                # Look for pet-specific backups
                backups = self.backup_manager.list_backups()
                pet_backups = [b for b in backups if b.pet_id == pet_id]
                
                if pet_backups:
                    latest_backup = pet_backups[0]
                    success = self.backup_manager.restore_backup(latest_backup.backup_id)
                    if success:
                        logger.info(f"Pet data recovered from backup: {latest_backup.backup_id}")
                        return True
            
            # Fallback to user-level backup
            if user_id:
                user_backups = [b for b in self.backup_manager.list_backups() if b.user_id == user_id]
                if user_backups:
                    latest_backup = user_backups[0]
                    success = self.backup_manager.restore_backup(latest_backup.backup_id)
                    if success:
                        logger.info(f"User data recovered from backup: {latest_backup.backup_id}")
                        return True
            
            logger.error("No suitable backups found for pet recovery")
            return False
            
        except Exception as e:
            logger.error(f"Pet data recovery failed: {e}")
            return False
    
    def recover_evolution_failure(self, error: PetLifecycleError) -> bool:
        """
        Recover from evolution failures.
        
        Args:
            error: Pet lifecycle error that occurred
            
        Returns:
            True if recovery successful
        """
        try:
            logger.info("Attempting evolution failure recovery")
            
            # Reset evolution state to previous stable state
            error.context['reset_evolution_state'] = True
            error.context['use_safe_evolution'] = True
            error.context['skip_complex_calculations'] = True
            
            logger.info("Set evolution recovery parameters")
            return True
            
        except Exception as e:
            logger.error(f"Evolution failure recovery failed: {e}")
            return False


class SystemRecoveryOrchestrator:
    """Orchestrates recovery strategies across all system components."""
    
    def __init__(self, backup_manager: Optional[BackupRecoveryManager] = None):
        """Initialize system recovery orchestrator."""
        self.backup_manager = backup_manager
        
        # Initialize recovery strategies
        self.storage_recovery = StorageRecoveryStrategy(backup_manager)
        self.ai_model_recovery = AIModelRecoveryStrategy()
        self.network_recovery = NetworkRecoveryStrategy()
        self.auth_recovery = AuthenticationRecoveryStrategy()
        self.pet_lifecycle_recovery = PetLifecycleRecoveryStrategy(backup_manager)
        
        # Register recovery strategies
        self._register_recovery_strategies()
    
    def _register_recovery_strategies(self):
        """Register all recovery strategies with the recovery manager."""
        
        # Storage recovery strategies
        recovery_manager.register_recovery_strategy(
            'storage', self.storage_recovery.recover_corrupted_database
        )
        recovery_manager.register_recovery_strategy(
            'storage', self.storage_recovery.recover_disk_space_issue
        )
        recovery_manager.register_recovery_strategy(
            'storage', self.storage_recovery.recover_permission_error
        )
        
        # AI model recovery strategies
        recovery_manager.register_recovery_strategy(
            'ai_model', self.ai_model_recovery.recover_model_loading_failure
        )
        recovery_manager.register_recovery_strategy(
            'ai_model', self.ai_model_recovery.recover_model_inference_failure
        )
        recovery_manager.register_recovery_strategy(
            'ai_model', self.ai_model_recovery.recover_memory_error
        )
        
        # Network recovery strategies
        recovery_manager.register_recovery_strategy(
            'network', self.network_recovery.recover_connection_timeout
        )
        recovery_manager.register_recovery_strategy(
            'network', self.network_recovery.recover_dns_resolution_failure
        )
        recovery_manager.register_recovery_strategy(
            'network', self.network_recovery.recover_rate_limit_error
        )
        
        # Authentication recovery strategies
        recovery_manager.register_recovery_strategy(
            'authentication', self.auth_recovery.recover_token_expiry
        )
        recovery_manager.register_recovery_strategy(
            'authentication', self.auth_recovery.recover_invalid_credentials
        )
        
        # Pet lifecycle recovery strategies
        recovery_manager.register_recovery_strategy(
            'pet_lifecycle', self.pet_lifecycle_recovery.recover_corrupted_pet_data
        )
        recovery_manager.register_recovery_strategy(
            'pet_lifecycle', self.pet_lifecycle_recovery.recover_evolution_failure
        )
        
        logger.info("All recovery strategies registered")
    
    def execute_comprehensive_recovery(self, error: DigiPalException) -> RecoveryResult:
        """
        Execute comprehensive recovery for any DigiPal error.
        
        Args:
            error: The error to recover from
            
        Returns:
            RecoveryResult with recovery outcome
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive recovery for error: {error.category.value}")
            
            # Create pre-recovery backup if critical error
            if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                if self.backup_manager:
                    backup_id = self.backup_manager.create_pre_operation_backup(
                        "error_recovery",
                        {"error_type": error.category.value, "error_code": error.error_code}
                    )
                    if backup_id:
                        logger.info(f"Pre-recovery backup created: {backup_id}")
            
            # Attempt recovery using registered strategies
            recovery_success = recovery_manager.attempt_recovery(error)
            
            recovery_time = time.time() - start_time
            
            if recovery_success:
                return RecoveryResult(
                    success=True,
                    strategy_used="comprehensive_recovery",
                    message="Recovery completed successfully",
                    context=error.context,
                    recovery_time_seconds=recovery_time
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used="comprehensive_recovery",
                    message="Recovery failed - manual intervention required",
                    context=error.context,
                    recovery_time_seconds=recovery_time
                )
                
        except Exception as recovery_error:
            recovery_time = time.time() - start_time
            logger.error(f"Comprehensive recovery failed: {recovery_error}")
            
            return RecoveryResult(
                success=False,
                strategy_used="comprehensive_recovery",
                message=f"Recovery failed with error: {str(recovery_error)}",
                context={"recovery_error": str(recovery_error)},
                recovery_time_seconds=recovery_time
            )
    
    def get_recovery_recommendations(self, error: DigiPalException) -> List[str]:
        """
        Get recovery recommendations for a specific error.
        
        Args:
            error: The error to get recommendations for
            
        Returns:
            List of recovery recommendations
        """
        recommendations = []
        
        # Add error-specific recommendations
        recommendations.extend(error.recovery_suggestions)
        
        # Add category-specific recommendations
        category_recommendations = {
            'storage': [
                "Check available disk space",
                "Verify file permissions",
                "Run database integrity check",
                "Consider restoring from backup"
            ],
            'ai_model': [
                "Check available memory",
                "Verify model files are not corrupted",
                "Try restarting the application",
                "Consider using reduced model precision"
            ],
            'network': [
                "Check internet connection",
                "Verify firewall settings",
                "Try using offline mode",
                "Check for service outages"
            ],
            'authentication': [
                "Verify credentials are correct",
                "Check token expiration",
                "Try logging out and back in",
                "Consider using offline mode"
            ],
            'pet_lifecycle': [
                "Check pet data integrity",
                "Verify backup availability",
                "Try reloading pet data",
                "Consider restoring from backup"
            ]
        }
        
        category_recs = category_recommendations.get(error.category.value, [])
        recommendations.extend(category_recs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations


# Global system recovery orchestrator
system_recovery_orchestrator = None


def initialize_system_recovery(backup_manager: Optional[BackupRecoveryManager] = None):
    """
    Initialize the global system recovery orchestrator.
    
    Args:
        backup_manager: Backup manager instance
    """
    global system_recovery_orchestrator
    system_recovery_orchestrator = SystemRecoveryOrchestrator(backup_manager)
    logger.info("System recovery orchestrator initialized")


def get_system_recovery_orchestrator() -> Optional[SystemRecoveryOrchestrator]:
    """Get the global system recovery orchestrator."""
    return system_recovery_orchestrator