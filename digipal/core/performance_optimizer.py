"""
Performance optimization module for DigiPal application.

This module provides:
- Lazy model loading with optimization
- Background task system for attribute decay and evolution checks
- Resource cleanup and garbage collection
- Database query optimization
- Memory usage monitoring
"""

import logging
import time
import threading
import gc
import psutil
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import sqlite3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import DigiPal
from .enums import LifeStage
from ..storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadingConfig:
    """Configuration for model loading optimization."""
    lazy_loading: bool = True
    quantization: bool = True
    model_cache_size: int = 2  # Maximum models to keep in memory
    unload_after_idle_minutes: int = 30
    preload_on_startup: bool = False
    use_cpu_offload: bool = True


@dataclass
class BackgroundTaskConfig:
    """Configuration for background tasks."""
    attribute_decay_interval: int = 300  # 5 minutes
    evolution_check_interval: int = 600   # 10 minutes
    memory_cleanup_interval: int = 1800   # 30 minutes
    database_optimization_interval: int = 3600  # 1 hour
    performance_monitoring_interval: int = 60   # 1 minute


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    active_pets: int
    cached_models: int
    database_connections: int
    response_time_avg: float


class LazyModelLoader:
    """Lazy loading system for AI models with optimization."""
    
    def __init__(self, config: ModelLoadingConfig):
        """Initialize lazy model loader."""
        self.config = config
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, datetime] = {}
        self.model_loading_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._cleanup_thread = None
        self._stop_cleanup = False
        
        logger.info("Lazy model loader initialized")
    
    def get_language_model(self, model_name: str = "Qwen/Qwen3-0.6B") -> Tuple[Any, Any]:
        """
        Get language model with lazy loading.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_key = f"language_{model_name}"
        
        with self.model_loading_locks[model_key]:
            # Check if model is already loaded
            if model_key in self.loaded_models:
                self.model_last_used[model_key] = datetime.now()
                model_data = self.loaded_models[model_key]
                return model_data['model'], model_data['tokenizer']
            
            # Load model if not in cache
            if self.config.lazy_loading:
                logger.info(f"Lazy loading language model: {model_name}")
                model, tokenizer = self._load_language_model(model_name)
                
                # Cache the model
                self.loaded_models[model_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'language',
                    'size_mb': self._estimate_model_size(model)
                }
                self.model_last_used[model_key] = datetime.now()
                
                # Manage cache size
                self._manage_model_cache()
                
                return model, tokenizer
            else:
                # Direct loading without caching
                return self._load_language_model(model_name)
    
    def get_speech_model(self, model_name: str = "kyutai/stt-2.6b-en_fr-trfs") -> Tuple[Any, Any]:
        """
        Get speech model with lazy loading.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Tuple of (model, processor)
        """
        model_key = f"speech_{model_name}"
        
        with self.model_loading_locks[model_key]:
            # Check if model is already loaded
            if model_key in self.loaded_models:
                self.model_last_used[model_key] = datetime.now()
                model_data = self.loaded_models[model_key]
                return model_data['model'], model_data['processor']
            
            # Load model if not in cache
            if self.config.lazy_loading:
                logger.info(f"Lazy loading speech model: {model_name}")
                model, processor = self._load_speech_model(model_name)
                
                # Cache the model
                self.loaded_models[model_key] = {
                    'model': model,
                    'processor': processor,
                    'type': 'speech',
                    'size_mb': self._estimate_model_size(model)
                }
                self.model_last_used[model_key] = datetime.now()
                
                # Manage cache size
                self._manage_model_cache()
                
                return model, processor
            else:
                # Direct loading without caching
                return self._load_speech_model(model_name)
    
    def _load_language_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load language model with optimization."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure model loading
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto" if torch.cuda.is_available() else None
            }
            
            # Add quantization if enabled
            if self.config.quantization and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Enable CPU offload if configured
            if self.config.use_cpu_offload and hasattr(model, 'enable_model_cpu_offload'):
                model.enable_model_cpu_offload()
            
            logger.info(f"Successfully loaded language model: {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load language model {model_name}: {e}")
            raise
    
    def _load_speech_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load speech model with optimization."""
        try:
            from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            processor = KyutaiSpeechToTextProcessor.from_pretrained(model_name)
            model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                model_name, 
                device_map=device, 
                torch_dtype="auto"
            )
            
            logger.info(f"Successfully loaded speech model: {model_name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load speech model {model_name}: {e}")
            raise
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB."""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() / (1024 * 1024)
            else:
                # Rough estimation based on parameters
                total_params = sum(p.numel() for p in model.parameters())
                # Assume 4 bytes per parameter (float32) or 2 bytes (float16)
                bytes_per_param = 2 if self.config.quantization else 4
                return (total_params * bytes_per_param) / (1024 * 1024)
        except:
            return 1000.0  # Default estimate
    
    def _manage_model_cache(self) -> None:
        """Manage model cache size by unloading least recently used models."""
        if len(self.loaded_models) <= self.config.model_cache_size:
            return
        
        # Sort models by last used time
        models_by_usage = sorted(
            self.loaded_models.items(),
            key=lambda x: self.model_last_used.get(x[0], datetime.min)
        )
        
        # Unload oldest models
        models_to_unload = models_by_usage[:-self.config.model_cache_size]
        
        for model_key, model_data in models_to_unload:
            self._unload_model(model_key)
            logger.info(f"Unloaded model {model_key} due to cache size limit")
    
    def _unload_model(self, model_key: str) -> None:
        """Unload a specific model from memory."""
        if model_key in self.loaded_models:
            model_data = self.loaded_models[model_key]
            
            # Clear model references
            if 'model' in model_data:
                del model_data['model']
            if 'tokenizer' in model_data:
                del model_data['tokenizer']
            if 'processor' in model_data:
                del model_data['processor']
            
            # Remove from cache
            del self.loaded_models[model_key]
            if model_key in self.model_last_used:
                del self.model_last_used[model_key]
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread for idle models."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup = False
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started model cleanup thread")
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop for idle models."""
        while not self._stop_cleanup:
            try:
                current_time = datetime.now()
                idle_threshold = timedelta(minutes=self.config.unload_after_idle_minutes)
                
                models_to_unload = []
                for model_key, last_used in self.model_last_used.items():
                    if current_time - last_used > idle_threshold:
                        models_to_unload.append(model_key)
                
                for model_key in models_to_unload:
                    self._unload_model(model_key)
                    logger.info(f"Unloaded idle model: {model_key}")
                
                # Sleep for cleanup interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in model cleanup loop: {e}")
                time.sleep(60)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about model cache."""
        total_size_mb = sum(
            model_data.get('size_mb', 0) 
            for model_data in self.loaded_models.values()
        )
        
        return {
            'loaded_models': len(self.loaded_models),
            'cache_limit': self.config.model_cache_size,
            'total_size_mb': total_size_mb,
            'models': {
                key: {
                    'type': data.get('type', 'unknown'),
                    'size_mb': data.get('size_mb', 0),
                    'last_used': self.model_last_used.get(key, datetime.min).isoformat()
                }
                for key, data in self.loaded_models.items()
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown the model loader and cleanup resources."""
        logger.info("Shutting down lazy model loader")
        
        # Stop cleanup thread
        self.stop_cleanup_thread()
        
        # Unload all models
        for model_key in list(self.loaded_models.keys()):
            self._unload_model(model_key)
        
        logger.info("Lazy model loader shutdown complete")


class BackgroundTaskManager:
    """Manages background tasks for attribute decay, evolution checks, etc."""
    
    def __init__(self, config: BackgroundTaskConfig, storage_manager: StorageManager):
        """Initialize background task manager."""
        self.config = config
        self.storage_manager = storage_manager
        self.active_tasks: Dict[str, threading.Thread] = {}
        self.task_stop_events: Dict[str, threading.Event] = {}
        self.task_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.task_performance: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("Background task manager initialized")
    
    def register_task(self, task_name: str, callback: Callable, interval_seconds: int) -> None:
        """
        Register a background task.
        
        Args:
            task_name: Unique task name
            callback: Function to call periodically
            interval_seconds: Interval between calls in seconds
        """
        self.task_callbacks[task_name] = callback
        self.task_stop_events[task_name] = threading.Event()
        
        # Create and start task thread
        task_thread = threading.Thread(
            target=self._task_loop,
            args=(task_name, callback, interval_seconds),
            daemon=True,
            name=f"BackgroundTask-{task_name}"
        )
        
        self.active_tasks[task_name] = task_thread
        task_thread.start()
        
        logger.info(f"Registered background task: {task_name} (interval: {interval_seconds}s)")
    
    def _task_loop(self, task_name: str, callback: Callable, interval_seconds: int) -> None:
        """Background task execution loop."""
        stop_event = self.task_stop_events[task_name]
        
        while not stop_event.is_set():
            try:
                start_time = time.time()
                
                # Execute task callback
                callback()
                
                # Track performance
                execution_time = time.time() - start_time
                self.task_performance[task_name].append(execution_time)
                
                # Keep only recent performance data
                if len(self.task_performance[task_name]) > 100:
                    self.task_performance[task_name] = self.task_performance[task_name][-50:]
                
                # Wait for next execution
                stop_event.wait(timeout=interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in background task {task_name}: {e}")
                # Wait before retrying
                stop_event.wait(timeout=min(60, interval_seconds))
    
    def stop_task(self, task_name: str) -> None:
        """Stop a specific background task."""
        if task_name in self.task_stop_events:
            self.task_stop_events[task_name].set()
        
        if task_name in self.active_tasks:
            thread = self.active_tasks[task_name]
            thread.join(timeout=5)
            del self.active_tasks[task_name]
        
        logger.info(f"Stopped background task: {task_name}")
    
    def stop_all_tasks(self) -> None:
        """Stop all background tasks."""
        logger.info("Stopping all background tasks")
        
        # Signal all tasks to stop
        for stop_event in self.task_stop_events.values():
            stop_event.set()
        
        # Wait for all threads to finish
        for task_name, thread in self.active_tasks.items():
            thread.join(timeout=5)
            logger.debug(f"Stopped task: {task_name}")
        
        # Clear task data
        self.active_tasks.clear()
        self.task_stop_events.clear()
        
        logger.info("All background tasks stopped")
    
    def get_task_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all tasks."""
        performance_stats = {}
        
        for task_name, execution_times in self.task_performance.items():
            if execution_times:
                performance_stats[task_name] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'max_execution_time': max(execution_times),
                    'min_execution_time': min(execution_times),
                    'total_executions': len(execution_times)
                }
            else:
                performance_stats[task_name] = {
                    'avg_execution_time': 0.0,
                    'max_execution_time': 0.0,
                    'min_execution_time': 0.0,
                    'total_executions': 0
                }
        
        return performance_stats


class DatabaseOptimizer:
    """Database query optimization and maintenance."""
    
    def __init__(self, storage_manager: StorageManager):
        """Initialize database optimizer."""
        self.storage_manager = storage_manager
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        logger.info("Database optimizer initialized")
    
    def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization tasks."""
        results = {}
        
        try:
            # Analyze database
            results['analyze'] = self._analyze_database()
            
            # Vacuum database
            results['vacuum'] = self._vacuum_database()
            
            # Update statistics
            results['statistics'] = self._update_statistics()
            
            # Check indexes
            results['indexes'] = self._check_indexes()
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_database(self) -> Dict[str, Any]:
        """Analyze database for optimization opportunities."""
        try:
            db = self.storage_manager.db
            
            # Get table sizes
            table_info = db.execute_query("""
                SELECT name, 
                       (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as table_count
                FROM sqlite_master m WHERE type='table'
            """)
            
            # Get index information
            index_info = db.execute_query("""
                SELECT name, tbl_name, sql 
                FROM sqlite_master 
                WHERE type='index' AND sql IS NOT NULL
            """)
            
            return {
                'tables': table_info,
                'indexes': index_info,
                'database_size': self._get_database_size()
            }
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return {'error': str(e)}
    
    def _vacuum_database(self) -> Dict[str, Any]:
        """Vacuum database to reclaim space."""
        try:
            db = self.storage_manager.db
            
            # Get size before vacuum
            size_before = self._get_database_size()
            
            # Perform vacuum
            db.execute_update("VACUUM")
            
            # Get size after vacuum
            size_after = self._get_database_size()
            
            space_reclaimed = size_before - size_after
            
            return {
                'size_before_mb': size_before / (1024 * 1024),
                'size_after_mb': size_after / (1024 * 1024),
                'space_reclaimed_mb': space_reclaimed / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return {'error': str(e)}
    
    def _update_statistics(self) -> Dict[str, Any]:
        """Update database statistics."""
        try:
            db = self.storage_manager.db
            
            # Analyze all tables
            db.execute_update("ANALYZE")
            
            return {'status': 'completed'}
            
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
            return {'error': str(e)}
    
    def _check_indexes(self) -> Dict[str, Any]:
        """Check and suggest index optimizations."""
        try:
            db = self.storage_manager.db
            
            # Check for missing indexes on frequently queried columns
            suggestions = []
            
            # Check digipals table
            digipals_count = db.execute_query("SELECT COUNT(*) as count FROM digipals")[0]['count']
            if digipals_count > 1000:
                # Suggest index on user_id if not exists
                existing_indexes = db.execute_query("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND tbl_name='digipals' AND sql LIKE '%user_id%'
                """)
                
                if not existing_indexes:
                    suggestions.append("CREATE INDEX idx_digipals_user_id ON digipals(user_id)")
            
            # Check interactions table
            interactions_count = db.execute_query("SELECT COUNT(*) as count FROM interactions")[0]['count']
            if interactions_count > 5000:
                # Suggest index on digipal_id and timestamp
                existing_indexes = db.execute_query("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND tbl_name='interactions' 
                    AND sql LIKE '%digipal_id%' AND sql LIKE '%timestamp%'
                """)
                
                if not existing_indexes:
                    suggestions.append("CREATE INDEX idx_interactions_digipal_timestamp ON interactions(digipal_id, timestamp)")
            
            return {
                'suggestions': suggestions,
                'digipals_count': digipals_count,
                'interactions_count': interactions_count
            }
            
        except Exception as e:
            logger.error(f"Index check failed: {e}")
            return {'error': str(e)}
    
    def _get_database_size(self) -> int:
        """Get database file size in bytes."""
        try:
            import os
            return os.path.getsize(self.storage_manager.db_path)
        except:
            return 0
    
    def create_suggested_indexes(self) -> Dict[str, Any]:
        """Create suggested indexes for better performance."""
        try:
            db = self.storage_manager.db
            results = []
            
            # Essential indexes for DigiPal application
            indexes_to_create = [
                "CREATE INDEX IF NOT EXISTS idx_digipals_user_id ON digipals(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_digipals_active ON digipals(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_interactions_digipal ON interactions(digipal_id)",
                "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_care_actions_digipal ON care_actions(digipal_id)",
                "CREATE INDEX IF NOT EXISTS idx_users_id ON users(id)"
            ]
            
            for index_sql in indexes_to_create:
                try:
                    db.execute_update(index_sql)
                    results.append(f"Created: {index_sql}")
                except Exception as e:
                    results.append(f"Failed: {index_sql} - {e}")
            
            return {'results': results}
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return {'error': str(e)}


class PerformanceMonitor:
    """System performance monitoring and optimization."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1440  # 24 hours of minute-by-minute data
        
        logger.info("Performance monitor initialized")
    
    def collect_metrics(self, active_pets: int = 0, cached_models: int = 0, 
                       response_time_avg: float = 0.0) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics
            gpu_memory_usage = 0.0
            if torch.cuda.is_available():
                gpu_memory_usage = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            
            # Database connections (approximate)
            database_connections = 1  # Simplified for SQLite
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory_usage,
                active_pets=active_pets,
                cached_models=cached_models,
                database_connections=database_connections,
                response_time_avg=response_time_avg
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Manage history size
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                active_pets=active_pets,
                cached_models=cached_models,
                database_connections=0,
                response_time_avg=response_time_avg
            )
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            recent_metrics = self.metrics_history[-60:]  # Last 60 data points
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
        
        # Calculate peaks
        max_cpu = max(m.cpu_usage for m in recent_metrics)
        max_memory = max(m.memory_usage for m in recent_metrics)
        max_gpu_memory = max(m.gpu_memory_usage for m in recent_metrics)
        
        return {
            'time_period_hours': hours,
            'data_points': len(recent_metrics),
            'averages': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'gpu_memory_usage': avg_gpu_memory,
                'response_time': avg_response_time
            },
            'peaks': {
                'cpu_usage': max_cpu,
                'memory_usage': max_memory,
                'gpu_memory_usage': max_gpu_memory
            },
            'current': {
                'active_pets': recent_metrics[-1].active_pets if recent_metrics else 0,
                'cached_models': recent_metrics[-1].cached_models if recent_metrics else 0
            }
        }
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance issues and return alerts."""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        latest = self.metrics_history[-1]
        
        # CPU usage alert
        if latest.cpu_usage > 80:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning' if latest.cpu_usage < 90 else 'critical',
                'message': f"High CPU usage: {latest.cpu_usage:.1f}%",
                'value': latest.cpu_usage
            })
        
        # Memory usage alert
        if latest.memory_usage > 85:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning' if latest.memory_usage < 95 else 'critical',
                'message': f"High memory usage: {latest.memory_usage:.1f}%",
                'value': latest.memory_usage
            })
        
        # GPU memory alert
        if latest.gpu_memory_usage > 90:
            alerts.append({
                'type': 'high_gpu_memory',
                'severity': 'warning',
                'message': f"High GPU memory usage: {latest.gpu_memory_usage:.1f}%",
                'value': latest.gpu_memory_usage
            })
        
        # Response time alert
        if latest.response_time_avg > 5.0:
            alerts.append({
                'type': 'slow_response',
                'severity': 'warning',
                'message': f"Slow response time: {latest.response_time_avg:.2f}s",
                'value': latest.response_time_avg
            })
        
        return alerts
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations based on metrics."""
        suggestions = []
        
        if not self.metrics_history:
            return suggestions
        
        # Analyze recent performance
        recent_metrics = self.metrics_history[-60:]  # Last hour of data
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # CPU optimization suggestions
        if avg_cpu > 70:
            suggestions.append("Consider reducing background task frequency")
            suggestions.append("Enable model CPU offloading to reduce CPU load")
        
        # Memory optimization suggestions
        if avg_memory > 80:
            suggestions.append("Reduce model cache size to free memory")
            suggestions.append("Enable more aggressive memory cleanup")
            suggestions.append("Consider using smaller quantized models")
        
        # GPU memory optimization suggestions
        if avg_gpu_memory > 80:
            suggestions.append("Enable 4-bit quantization for models")
            suggestions.append("Reduce maximum concurrent model loading")
            suggestions.append("Use CPU offloading for less frequently used models")
        
        # General suggestions
        if len(recent_metrics) > 0:
            max_active_pets = max(m.active_pets for m in recent_metrics)
            if max_active_pets > 100:
                suggestions.append("Consider implementing pet data pagination")
                suggestions.append("Optimize database queries with better indexing")
        
        return suggestions


class ResourceCleanupManager:
    """Manages resource cleanup and garbage collection."""
    
    def __init__(self):
        """Initialize resource cleanup manager."""
        self.cleanup_callbacks: List[Callable] = []
        self.last_cleanup = datetime.now()
        
        logger.info("Resource cleanup manager initialized")
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)
    
    def perform_cleanup(self, force_gc: bool = True) -> Dict[str, Any]:
        """Perform comprehensive resource cleanup."""
        cleanup_results = {}
        
        try:
            # Execute registered cleanup callbacks
            for i, callback in enumerate(self.cleanup_callbacks):
                try:
                    callback()
                    cleanup_results[f'callback_{i}'] = 'success'
                except Exception as e:
                    cleanup_results[f'callback_{i}'] = f'error: {e}'
            
            # Python garbage collection
            if force_gc:
                collected = gc.collect()
                cleanup_results['gc_collected'] = collected
            
            # PyTorch cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_results['cuda_cache_cleared'] = True
            
            # Update last cleanup time
            self.last_cleanup = datetime.now()
            cleanup_results['cleanup_time'] = self.last_cleanup.isoformat()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            cleanup_results['error'] = str(e)
        
        return cleanup_results
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Python memory (approximate)
            import sys
            python_objects = len(gc.get_objects())
            
            # PyTorch memory
            torch_memory = {}
            if torch.cuda.is_available():
                torch_memory = {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
                }
            
            return {
                'system_memory': {
                    'total_mb': memory.total / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'used_mb': memory.used / (1024 * 1024),
                    'percent': memory.percent
                },
                'python_objects': python_objects,
                'torch_memory': torch_memory,
                'last_cleanup': self.last_cleanup.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {'error': str(e)}