"""
Error handling utilities and decorators for DigiPal application.

This module provides comprehensive error handling, retry mechanisms,
and graceful degradation functionality.
"""

import logging
import functools
import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from datetime import datetime, timedelta

from .exceptions import (
    DigiPalException, ErrorSeverity, ErrorCategory,
    AIModelError, StorageError, NetworkError, RecoveryError
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retry_on: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff
            jitter: Whether to add random jitter to delays
            retry_on: List of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retry_on = retry_on or [NetworkError, AIModelError]


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker configuration.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise DigiPalException(
                    "Circuit breaker is open - service unavailable",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    user_message="Service is temporarily unavailable. Please try again later."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful function call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "open"


class ErrorHandler:
    """Central error handler for DigiPal application."""
    
    def __init__(self):
        """Initialize error handler."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create circuit breaker for a service.
        
        Args:
            name: Service name
            config: Circuit breaker configuration
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config)
        
        return self.circuit_breakers[name]
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> DigiPalException:
        """
        Handle and convert errors to DigiPal exceptions.
        
        Args:
            error: Original exception
            context: Additional context information
            
        Returns:
            DigiPalException with appropriate categorization
        """
        context = context or {}
        
        # If already a DigiPal exception, just add context and return
        if isinstance(error, DigiPalException):
            error.context.update(context)
            return error
        
        # Convert common exceptions to DigiPal exceptions
        if isinstance(error, (ConnectionError, TimeoutError)):
            return NetworkError(
                f"Network error: {str(error)}",
                context=context,
                error_code="NET_001"
            )
        
        if isinstance(error, FileNotFoundError):
            return StorageError(
                f"File not found: {str(error)}",
                context=context,
                error_code="STOR_001"
            )
        
        if isinstance(error, PermissionError):
            return StorageError(
                f"Permission denied: {str(error)}",
                context=context,
                error_code="STOR_002"
            )
        
        if isinstance(error, ValueError):
            return DigiPalException(
                f"Invalid value: {str(error)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                context=context,
                error_code="VAL_001"
            )
        
        # Default to system error
        return DigiPalException(
            f"Unexpected error: {str(error)}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=context,
            error_code="SYS_001"
        )
    
    def log_error(self, error: DigiPalException, extra_context: Optional[Dict[str, Any]] = None):
        """
        Log error with appropriate level and context.
        
        Args:
            error: DigiPal exception to log
            extra_context: Additional context for logging
        """
        context = {**error.context, **(extra_context or {})}
        
        log_data = {
            'error_code': error.error_code,
            'category': error.category.value,
            'severity': error.severity.value,
            'user_message': error.user_message,
            'context': context
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {str(error)}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {str(error)}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {str(error)}", extra=log_data)
        else:
            logger.info(f"LOW SEVERITY: {str(error)}", extra=log_data)
        
        # Track error frequency
        error_key = f"{error.category.value}:{error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.now()


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    fallback_value: Any = None,
    log_errors: bool = True,
    raise_on_critical: bool = True,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
        raise_on_critical: Whether to raise critical errors
        context: Additional context for error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                digipal_error = error_handler.handle_error(e, context)
                
                if log_errors:
                    error_handler.log_error(digipal_error)
                
                if raise_on_critical and digipal_error.severity == ErrorSeverity.CRITICAL:
                    raise digipal_error
                
                if digipal_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and fallback_value is None:
                    raise digipal_error
                
                return fallback_value
        
        return wrapper
    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retry functionality.
    
    Args:
        config: Retry configuration
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry on this exception
                    should_retry = any(isinstance(e, exc_type) for exc_type in config.retry_on)
                    
                    if not should_retry or attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = config.base_delay
                    if config.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    delay = min(delay, config.max_delay)
                    
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})")
                    time.sleep(delay)
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def with_circuit_breaker(service_name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for circuit breaker functionality.
    
    Args:
        service_name: Name of the service for circuit breaker
        config: Circuit breaker configuration
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            circuit_breaker = error_handler.get_circuit_breaker(service_name, config)
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


async def with_async_error_handling(
    fallback_value: Any = None,
    log_errors: bool = True,
    raise_on_critical: bool = True,
    context: Optional[Dict[str, Any]] = None
):
    """
    Async decorator for comprehensive error handling.
    
    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
        raise_on_critical: Whether to raise critical errors
        context: Additional context for error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                digipal_error = error_handler.handle_error(e, context)
                
                if log_errors:
                    error_handler.log_error(digipal_error)
                
                if raise_on_critical and digipal_error.severity == ErrorSeverity.CRITICAL:
                    raise digipal_error
                
                if digipal_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and fallback_value is None:
                    raise digipal_error
                
                return fallback_value
        
        return wrapper
    return decorator


async def with_async_retry(config: Optional[RetryConfig] = None):
    """
    Async decorator for retry functionality.
    
    Args:
        config: Retry configuration
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry on this exception
                    should_retry = any(isinstance(e, exc_type) for exc_type in config.retry_on)
                    
                    if not should_retry or attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = config.base_delay
                    if config.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    delay = min(delay, config.max_delay)
                    
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})")
                    await asyncio.sleep(delay)
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def create_fallback_response(error: DigiPalException, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized fallback response for errors.
    
    Args:
        error: DigiPal exception
        context: Additional context
        
    Returns:
        Fallback response dictionary
    """
    return {
        'success': False,
        'error': {
            'message': error.user_message,
            'category': error.category.value,
            'severity': error.severity.value,
            'recovery_suggestions': error.recovery_suggestions,
            'error_code': error.error_code
        },
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }


def get_error_statistics() -> Dict[str, Any]:
    """
    Get error statistics for monitoring and debugging.
    
    Returns:
        Dictionary with error statistics
    """
    return {
        'error_counts': dict(error_handler.error_counts),
        'last_errors': {k: v.isoformat() for k, v in error_handler.last_errors.items()},
        'circuit_breaker_states': {
            name: {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for name, cb in error_handler.circuit_breakers.items()
        }
    }


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self):
        """Initialize health checker."""
        self.component_health: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_interval = 300  # 5 minutes
    
    def register_component(self, component_name: str, health_check_func: Callable[[], bool]):
        """
        Register a component for health checking.
        
        Args:
            component_name: Name of the component
            health_check_func: Function that returns True if component is healthy
        """
        self.component_health[component_name] = True
        self._health_check_functions[component_name] = health_check_func
    
    def check_component_health(self, component_name: str) -> bool:
        """
        Check health of a specific component.
        
        Args:
            component_name: Name of component to check
            
        Returns:
            True if component is healthy
        """
        if component_name not in self._health_check_functions:
            return False
        
        try:
            health_func = self._health_check_functions[component_name]
            is_healthy = health_func()
            self.component_health[component_name] = is_healthy
            self.last_health_check[component_name] = datetime.now()
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            self.component_health[component_name] = False
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with health information
        """
        # Check all components
        for component in self._health_check_functions.keys():
            self.check_component_health(component)
        
        healthy_components = sum(1 for health in self.component_health.values() if health)
        total_components = len(self.component_health)
        
        overall_health = "healthy" if healthy_components == total_components else "degraded"
        if healthy_components == 0:
            overall_health = "critical"
        elif healthy_components < total_components * 0.5:
            overall_health = "unhealthy"
        
        return {
            'overall_health': overall_health,
            'healthy_components': healthy_components,
            'total_components': total_components,
            'component_status': dict(self.component_health),
            'last_checks': {k: v.isoformat() for k, v in self.last_health_check.items()}
        }
    
    def __init__(self):
        """Initialize health checker."""
        self.component_health: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_interval = 300  # 5 minutes
        self._health_check_functions: Dict[str, Callable[[], bool]] = {}


# Global health checker instance
health_checker = HealthChecker()


class RecoveryManager:
    """Manages recovery operations for various system failures."""
    
    def __init__(self):
        """Initialize recovery manager."""
        self.recovery_strategies: Dict[str, List[Callable]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
    
    def register_recovery_strategy(self, error_type: str, recovery_func: Callable):
        """
        Register a recovery strategy for an error type.
        
        Args:
            error_type: Type of error (e.g., 'storage_error', 'ai_model_error')
            recovery_func: Function to attempt recovery
        """
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        self.recovery_strategies[error_type].append(recovery_func)
    
    def attempt_recovery(self, error: DigiPalException) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error: The error to recover from
            
        Returns:
            True if recovery was successful
        """
        error_type = error.category.value
        
        if error_type not in self.recovery_strategies:
            logger.warning(f"No recovery strategies for error type: {error_type}")
            return False
        
        recovery_attempt = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'error_code': error.error_code,
            'strategies_attempted': [],
            'success': False
        }
        
        for strategy in self.recovery_strategies[error_type]:
            try:
                strategy_name = strategy.__name__
                logger.info(f"Attempting recovery strategy: {strategy_name}")
                
                success = strategy(error)
                recovery_attempt['strategies_attempted'].append({
                    'strategy': strategy_name,
                    'success': success
                })
                
                if success:
                    recovery_attempt['success'] = True
                    logger.info(f"Recovery successful using strategy: {strategy_name}")
                    break
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                recovery_attempt['strategies_attempted'].append({
                    'strategy': strategy.__name__,
                    'success': False,
                    'error': str(recovery_error)
                })
        
        self.recovery_history.append(recovery_attempt)
        return recovery_attempt['success']
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {
                'total_attempts': 0,
                'successful_recoveries': 0,
                'success_rate': 0.0,
                'error_types': {}
            }
        
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for attempt in self.recovery_history if attempt['success'])
        success_rate = successful_recoveries / total_attempts if total_attempts > 0 else 0.0
        
        error_types = {}
        for attempt in self.recovery_history:
            error_type = attempt['error_type']
            if error_type not in error_types:
                error_types[error_type] = {'attempts': 0, 'successes': 0}
            error_types[error_type]['attempts'] += 1
            if attempt['success']:
                error_types[error_type]['successes'] += 1
        
        return {
            'total_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'success_rate': success_rate,
            'error_types': error_types,
            'recent_attempts': self.recovery_history[-10:]  # Last 10 attempts
        }


# Global recovery manager instance
recovery_manager = RecoveryManager()