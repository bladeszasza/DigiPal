"""
Error handling integration module for DigiPal.

This module provides a unified interface for all error handling,
recovery, and user messaging functionality across the DigiPal system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass

from .exceptions import DigiPalException, ErrorSeverity, ErrorCategory
from .error_handler import error_handler, with_error_handling, with_retry, RetryConfig
from .recovery_strategies import get_system_recovery_orchestrator, RecoveryResult
from .user_error_messages import (
    get_user_friendly_error_message, get_recovery_guide, 
    MessageTone, user_message_generator
)
from ..storage.backup_recovery import BackupRecoveryManager
from ..ai.graceful_degradation import ai_service_manager

logger = logging.getLogger(__name__)


@dataclass
class ErrorHandlingResult:
    """Result of comprehensive error handling."""
    success: bool
    original_error: DigiPalException
    user_message: str
    recovery_attempted: bool
    recovery_result: Optional[RecoveryResult]
    recovery_guide: List[Dict[str, Any]]
    fallback_value: Any
    context: Dict[str, Any]


class DigiPalErrorHandler:
    """Unified error handler for the entire DigiPal system."""
    
    def __init__(self, backup_manager: Optional[BackupRecoveryManager] = None):
        """
        Initialize the unified error handler.
        
        Args:
            backup_manager: Backup manager for recovery operations
        """
        self.backup_manager = backup_manager
        self.error_callbacks: Dict[str, List[Callable]] = {}
        self.user_context: Dict[str, Any] = {}
        self.message_tone = MessageTone.FRIENDLY
        
        # Initialize recovery system if backup manager is provided
        if backup_manager:
            from .recovery_strategies import initialize_system_recovery
            initialize_system_recovery(backup_manager)
        
        logger.info("DigiPal unified error handler initialized")
    
    def set_user_context(self, context: Dict[str, Any]):
        """
        Set user context for personalized error messages.
        
        Args:
            context: User context information
        """
        self.user_context.update(context)
    
    def set_message_tone(self, tone: MessageTone):
        """
        Set the tone for error messages.
        
        Args:
            tone: Message tone to use
        """
        self.message_tone = tone
    
    def register_error_callback(self, error_category: str, callback: Callable):
        """
        Register a callback for specific error categories.
        
        Args:
            error_category: Error category to listen for
            callback: Callback function to execute
        """
        if error_category not in self.error_callbacks:
            self.error_callbacks[error_category] = []
        
        self.error_callbacks[error_category].append(callback)
        logger.info(f"Registered error callback for category: {error_category}")
    
    def handle_error_comprehensive(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
        provide_fallback: bool = True,
        fallback_value: Any = None
    ) -> ErrorHandlingResult:
        """
        Comprehensive error handling with recovery and user messaging.
        
        Args:
            error: The error that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            provide_fallback: Whether to provide fallback functionality
            fallback_value: Fallback value to return
            
        Returns:
            ErrorHandlingResult with complete handling information
        """
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Convert to DigiPal exception if needed
            if not isinstance(error, DigiPalException):
                digipal_error = error_handler.handle_error(error, context)
            else:
                digipal_error = error
                digipal_error.context.update(context)
            
            # Log the error
            error_handler.log_error(digipal_error, {'handling_start': start_time.isoformat()})
            
            # Generate user-friendly message
            user_message = get_user_friendly_error_message(
                digipal_error,
                tone=self.message_tone,
                user_context=self.user_context
            )
            
            # Get recovery guide
            recovery_guide = get_recovery_guide(digipal_error, max_steps=4)
            
            # Attempt recovery if requested and appropriate
            recovery_result = None
            recovery_attempted = False
            
            if attempt_recovery and digipal_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                recovery_attempted = True
                orchestrator = get_system_recovery_orchestrator()
                
                if orchestrator:
                    recovery_result = orchestrator.execute_comprehensive_recovery(digipal_error)
                    
                    if recovery_result.success:
                        logger.info(f"Recovery successful for error: {digipal_error.error_code}")
                        user_message = user_message_generator.generate_success_message(
                            digipal_error.category.value,
                            recovery_result.strategy_used
                        )
                    else:
                        logger.warning(f"Recovery failed for error: {digipal_error.error_code}")
            
            # Handle graceful degradation for AI services
            if digipal_error.category == ErrorCategory.AI_MODEL:
                self._handle_ai_service_degradation(digipal_error)
            
            # Execute registered callbacks
            self._execute_error_callbacks(digipal_error)
            
            # Determine success based on recovery and severity
            overall_success = (
                recovery_result.success if recovery_result 
                else digipal_error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
            )
            
            # Prepare result
            result = ErrorHandlingResult(
                success=overall_success,
                original_error=digipal_error,
                user_message=user_message,
                recovery_attempted=recovery_attempted,
                recovery_result=recovery_result,
                recovery_guide=recovery_guide,
                fallback_value=fallback_value if provide_fallback else None,
                context={
                    **digipal_error.context,
                    'handling_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'user_context': self.user_context,
                    'message_tone': self.message_tone.value
                }
            )
            
            return result
            
        except Exception as handling_error:
            logger.error(f"Error handling failed: {handling_error}")
            
            # Return minimal error result
            return ErrorHandlingResult(
                success=False,
                original_error=error_handler.handle_error(handling_error),
                user_message="An unexpected error occurred during error handling. Please restart the application.",
                recovery_attempted=False,
                recovery_result=None,
                recovery_guide=[],
                fallback_value=fallback_value if provide_fallback else None,
                context={'handling_error': str(handling_error)}
            )
    
    def _handle_ai_service_degradation(self, error: DigiPalException):
        """Handle AI service degradation."""
        try:
            # Update AI service status
            service_name = error.context.get('service_name', 'unknown')
            if service_name in ai_service_manager.service_status:
                ai_service_manager.service_status[service_name] = False
                ai_service_manager._update_degradation_level()
                
                logger.info(f"AI service {service_name} marked as degraded")
        except Exception as e:
            logger.error(f"Failed to handle AI service degradation: {e}")
    
    def _execute_error_callbacks(self, error: DigiPalException):
        """Execute registered error callbacks."""
        try:
            category_callbacks = self.error_callbacks.get(error.category.value, [])
            general_callbacks = self.error_callbacks.get('all', [])
            
            all_callbacks = category_callbacks + general_callbacks
            
            for callback in all_callbacks:
                try:
                    callback(error)
                except Exception as callback_error:
                    logger.error(f"Error callback failed: {callback_error}")
        except Exception as e:
            logger.error(f"Failed to execute error callbacks: {e}")
    
    async def handle_error_async(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
        provide_fallback: bool = True,
        fallback_value: Any = None
    ) -> ErrorHandlingResult:
        """
        Asynchronous comprehensive error handling.
        
        Args:
            error: The error that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            provide_fallback: Whether to provide fallback functionality
            fallback_value: Fallback value to return
            
        Returns:
            ErrorHandlingResult with complete handling information
        """
        # Run synchronous error handling in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.handle_error_comprehensive,
            error,
            context,
            attempt_recovery,
            provide_fallback,
            fallback_value
        )
    
    def create_error_safe_wrapper(
        self,
        func: Callable,
        fallback_value: Any = None,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Create an error-safe wrapper for any function.
        
        Args:
            func: Function to wrap
            fallback_value: Value to return on error
            retry_config: Retry configuration
            context: Additional context
            
        Returns:
            Error-safe wrapped function
        """
        def wrapper(*args, **kwargs):
            try:
                # Apply retry if configured
                if retry_config:
                    @with_retry(retry_config)
                    def retryable_func():
                        return func(*args, **kwargs)
                    
                    return retryable_func()
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                result = self.handle_error_comprehensive(
                    e,
                    context=context,
                    fallback_value=fallback_value
                )
                
                if result.success or result.fallback_value is not None:
                    return result.fallback_value
                else:
                    raise result.original_error
        
        return wrapper
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            System health information
        """
        try:
            from .error_handler import get_error_statistics, health_checker
            
            # Get error statistics
            error_stats = get_error_statistics()
            
            # Get AI service status
            ai_status = ai_service_manager.get_service_status()
            
            # Get backup status
            backup_stats = {}
            if self.backup_manager:
                backup_stats = self.backup_manager.get_backup_statistics()
            
            # Get health check status
            health_status = health_checker.get_system_health()
            
            # Calculate overall health score
            health_score = self._calculate_health_score(error_stats, ai_status, health_status)
            
            return {
                'overall_health_score': health_score,
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'critical',
                'error_statistics': error_stats,
                'ai_service_status': ai_status,
                'backup_statistics': backup_stats,
                'component_health': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health status: {e}")
            return {
                'overall_health_score': 0.0,
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_health_score(
        self,
        error_stats: Dict[str, Any],
        ai_status: Dict[str, Any],
        health_status: Dict[str, Any]
    ) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        try:
            score = 1.0
            
            # Reduce score based on error rate
            error_rate = error_handler.get_error_rate(window_minutes=5)
            if error_rate > 0:
                score -= min(error_rate * 0.1, 0.5)  # Max 50% reduction for errors
            
            # Reduce score based on AI service degradation
            degradation_level = ai_status.get('degradation_level', 'full_service')
            degradation_penalties = {
                'full_service': 0.0,
                'reduced_features': 0.1,
                'basic_responses': 0.3,
                'minimal_function': 0.5,
                'emergency_mode': 0.7
            }
            score -= degradation_penalties.get(degradation_level, 0.5)
            
            # Reduce score based on component health
            if health_status.get('overall_health') == 'critical':
                score -= 0.4
            elif health_status.get('overall_health') == 'unhealthy':
                score -= 0.2
            elif health_status.get('overall_health') == 'degraded':
                score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.5  # Default to moderate health
    
    def generate_health_report(self) -> str:
        """
        Generate a human-readable health report.
        
        Returns:
            Health report string
        """
        try:
            health_status = self.get_system_health_status()
            
            report_lines = [
                "=== DigiPal System Health Report ===",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Overall Status: {health_status['status'].upper()}",
                f"Health Score: {health_status['overall_health_score']:.2f}/1.00",
                ""
            ]
            
            # Error statistics
            error_stats = health_status.get('error_statistics', {})
            if error_stats.get('error_counts'):
                report_lines.extend([
                    "Recent Errors:",
                    f"  - Total error types: {len(error_stats['error_counts'])}",
                    f"  - Most frequent: {list(error_stats['error_counts'].keys())[:3]}"
                ])
            else:
                report_lines.append("Recent Errors: None")
            
            report_lines.append("")
            
            # AI service status
            ai_status = health_status.get('ai_service_status', {})
            degradation_level = ai_status.get('degradation_level', 'unknown')
            report_lines.extend([
                f"AI Services: {degradation_level.replace('_', ' ').title()}",
                f"  - Language Model: {'✓' if ai_status.get('services', {}).get('language_model') else '✗'}",
                f"  - Speech Processing: {'✓' if ai_status.get('services', {}).get('speech_processing') else '✗'}",
                f"  - Image Generation: {'✓' if ai_status.get('services', {}).get('image_generation') else '✗'}",
                ""
            ])
            
            # Backup status
            backup_stats = health_status.get('backup_statistics', {})
            if backup_stats:
                report_lines.extend([
                    f"Backup System:",
                    f"  - Total backups: {backup_stats.get('total_backups', 0)}",
                    f"  - Storage used: {backup_stats.get('total_size_bytes', 0) / 1024 / 1024:.1f} MB",
                    ""
                ])
            
            # Recommendations
            recommendations = self._generate_health_recommendations(health_status)
            if recommendations:
                report_lines.extend([
                    "Recommendations:",
                    *[f"  - {rec}" for rec in recommendations],
                    ""
                ])
            
            report_lines.append("=== End of Report ===")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            return f"Health report generation failed: {str(e)}"
    
    def _generate_health_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system status."""
        recommendations = []
        
        try:
            health_score = health_status.get('overall_health_score', 1.0)
            
            if health_score < 0.5:
                recommendations.append("System health is critical - consider restarting the application")
            
            # AI service recommendations
            ai_status = health_status.get('ai_service_status', {})
            degradation_level = ai_status.get('degradation_level', 'full_service')
            
            if degradation_level != 'full_service':
                recommendations.append("AI services are degraded - check internet connection and model availability")
            
            # Error rate recommendations
            error_stats = health_status.get('error_statistics', {})
            if error_stats.get('error_counts'):
                most_frequent = list(error_stats['error_counts'].keys())[:1]
                if most_frequent:
                    error_type = most_frequent[0].split(':')[0]
                    recommendations.append(f"High {error_type} error rate - check system logs for details")
            
            # Backup recommendations
            backup_stats = health_status.get('backup_statistics', {})
            if backup_stats.get('total_backups', 0) == 0:
                recommendations.append("No backups found - enable automatic backups for data safety")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations - check system logs")
        
        return recommendations


# Global unified error handler instance
unified_error_handler: Optional[DigiPalErrorHandler] = None


def initialize_error_handling(backup_manager: Optional[BackupRecoveryManager] = None):
    """
    Initialize the global unified error handler.
    
    Args:
        backup_manager: Backup manager for recovery operations
    """
    global unified_error_handler
    unified_error_handler = DigiPalErrorHandler(backup_manager)
    logger.info("Global unified error handler initialized")


def get_error_handler() -> Optional[DigiPalErrorHandler]:
    """Get the global unified error handler."""
    return unified_error_handler


def handle_error_safely(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    fallback_value: Any = None
) -> ErrorHandlingResult:
    """
    Safely handle an error using the global error handler.
    
    Args:
        error: The error that occurred
        context: Additional context
        fallback_value: Fallback value
        
    Returns:
        ErrorHandlingResult
    """
    if unified_error_handler:
        return unified_error_handler.handle_error_comprehensive(
            error, context, fallback_value=fallback_value
        )
    else:
        # Fallback to basic error handling
        digipal_error = error_handler.handle_error(error, context)
        return ErrorHandlingResult(
            success=False,
            original_error=digipal_error,
            user_message="An error occurred. Please try again.",
            recovery_attempted=False,
            recovery_result=None,
            recovery_guide=[],
            fallback_value=fallback_value,
            context=context or {}
        )


def create_safe_function(
    func: Callable,
    fallback_value: Any = None,
    context: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a safe version of any function.
    
    Args:
        func: Function to make safe
        fallback_value: Value to return on error
        context: Additional context
        
    Returns:
        Safe function wrapper
    """
    if unified_error_handler:
        return unified_error_handler.create_error_safe_wrapper(
            func, fallback_value, context=context
        )
    else:
        # Basic wrapper
        def safe_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                return fallback_value
        
        return safe_wrapper