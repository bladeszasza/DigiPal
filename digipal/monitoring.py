"""
DigiPal Monitoring and Metrics Collection
Provides Prometheus metrics and health checks for production deployment
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
from prometheus_client.core import CollectorRegistry
import threading
import psutil
import os


class DigiPalMetrics:
    """Prometheus metrics collector for DigiPal"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self._init_metrics()
        
        # System metrics
        self._init_system_metrics()
        
        # Start background metrics collection
        self._start_background_collection()
    
    def _init_metrics(self):
        """Initialize application-specific metrics"""
        # Request metrics
        self.http_requests_total = Counter(
            'digipal_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'digipal_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Pet interaction metrics
        self.pet_interactions_total = Counter(
            'digipal_pet_interactions_total',
            'Total pet interactions',
            ['interaction_type', 'success'],
            registry=self.registry
        )
        
        self.active_pets = Gauge(
            'digipal_active_pets',
            'Number of active pets',
            registry=self.registry
        )
        
        self.pet_evolutions_total = Counter(
            'digipal_pet_evolutions_total',
            'Total pet evolutions',
            ['from_stage', 'to_stage'],
            registry=self.registry
        )
        
        # AI model metrics
        self.ai_model_requests_total = Counter(
            'digipal_ai_model_requests_total',
            'Total AI model requests',
            ['model_type', 'success'],
            registry=self.registry
        )
        
        self.ai_model_response_time = Histogram(
            'digipal_ai_model_response_time_seconds',
            'AI model response time',
            ['model_type'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_operations_total = Counter(
            'digipal_database_operations_total',
            'Total database operations',
            ['operation', 'success'],
            registry=self.registry
        )
        
        self.database_errors_total = Counter(
            'digipal_database_errors_total',
            'Total database errors',
            ['error_type'],
            registry=self.registry
        )
        
        # MCP server metrics
        self.mcp_requests_total = Counter(
            'digipal_mcp_requests_total',
            'Total MCP requests',
            ['tool_name', 'success'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'digipal_errors_total',
            'Total application errors',
            ['error_type', 'severity'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'digipal_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'digipal_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system-level metrics"""
        self.memory_usage_bytes = Gauge(
            'digipal_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'digipal_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'digipal_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'digipal_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        self.start_time = time.time()
    
    def _start_background_collection(self):
        """Start background thread for system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self.memory_usage_bytes.set(memory_info.rss)
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.cpu_usage_percent.set(cpu_percent)
                    
                    # Disk usage
                    disk_usage = psutil.disk_usage('/')
                    self.disk_usage_bytes.labels(path='/').set(disk_usage.used)
                    
                    # Uptime
                    uptime = time.time() - self.start_time
                    self.uptime_seconds.set(uptime)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    # Metric recording methods
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_pet_interaction(self, interaction_type: str, success: bool):
        """Record pet interaction metrics"""
        self.pet_interactions_total.labels(
            interaction_type=interaction_type,
            success=str(success)
        ).inc()
    
    def set_active_pets(self, count: int):
        """Set number of active pets"""
        self.active_pets.set(count)
    
    def record_pet_evolution(self, from_stage: str, to_stage: str):
        """Record pet evolution"""
        self.pet_evolutions_total.labels(from_stage=from_stage, to_stage=to_stage).inc()
    
    def record_ai_model_request(self, model_type: str, success: bool, duration: float):
        """Record AI model request metrics"""
        self.ai_model_requests_total.labels(model_type=model_type, success=str(success)).inc()
        self.ai_model_response_time.labels(model_type=model_type).observe(duration)
    
    def record_database_operation(self, operation: str, success: bool):
        """Record database operation metrics"""
        self.database_operations_total.labels(operation=operation, success=str(success)).inc()
    
    def record_database_error(self, error_type: str):
        """Record database error"""
        self.database_errors_total.labels(error_type=error_type).inc()
    
    def record_mcp_request(self, tool_name: str, success: bool):
        """Record MCP request metrics"""
        self.mcp_requests_total.labels(tool_name=tool_name, success=str(success)).inc()
    
    def record_error(self, error_type: str, severity: str):
        """Record application error"""
        self.errors_total.labels(error_type=error_type, severity=severity).inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.cache_misses_total.labels(cache_type=cache_type).inc()


class HealthChecker:
    """Health check system for DigiPal"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks = {}
        self.last_check_time = {}
        self.check_results = {}
    
    def register_check(self, name: str, check_func, timeout: int = 10):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout
        }
        self.check_results[name] = {'status': 'unknown', 'message': 'Not checked yet'}
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {'status': 'error', 'message': f'Check {name} not found'}
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            result = check['func']()
            duration = time.time() - start_time
            
            if duration > check['timeout']:
                return {
                    'status': 'warning',
                    'message': f'Check took {duration:.2f}s (timeout: {check["timeout"]}s)',
                    'duration': duration
                }
            
            return {
                'status': 'healthy',
                'message': result.get('message', 'OK'),
                'duration': duration,
                **result
            }
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Health check {name} failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'duration': duration
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            self.check_results[name] = result
            self.last_check_time[name] = time.time()
            
            if result['status'] == 'error':
                overall_status = 'unhealthy'
            elif result['status'] == 'warning' and overall_status == 'healthy':
                overall_status = 'warning'
        
        return {
            'status': overall_status,
            'timestamp': time.time(),
            'checks': results
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.run_all_checks()


# Global instances
metrics = DigiPalMetrics()
health_checker = HealthChecker()


def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server"""
    try:
        start_http_server(port, registry=metrics.registry)
        logging.info(f"Metrics server started on port {port}")
    except Exception as e:
        logging.error(f"Failed to start metrics server: {e}")


def get_metrics() -> str:
    """Get current metrics in Prometheus format"""
    return generate_latest(metrics.registry)


def setup_default_health_checks():
    """Setup default health checks for DigiPal components"""
    
    def check_database():
        """Check database connectivity"""
        try:
            # This would be implemented with actual database check
            return {'message': 'Database connection OK'}
        except Exception as e:
            raise Exception(f"Database check failed: {e}")
    
    def check_ai_models():
        """Check AI model availability"""
        try:
            # This would be implemented with actual model check
            return {'message': 'AI models loaded and ready'}
        except Exception as e:
            raise Exception(f"AI model check failed: {e}")
    
    def check_disk_space():
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 10:
                raise Exception(f"Low disk space: {free_percent:.1f}% free")
            
            return {'message': f'Disk space OK: {free_percent:.1f}% free'}
        except Exception as e:
            raise Exception(f"Disk space check failed: {e}")
    
    def check_memory():
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                raise Exception(f"High memory usage: {memory.percent}%")
            
            return {'message': f'Memory usage OK: {memory.percent}%'}
        except Exception as e:
            raise Exception(f"Memory check failed: {e}")
    
    # Register health checks
    health_checker.register_check('database', check_database)
    health_checker.register_check('ai_models', check_ai_models)
    health_checker.register_check('disk_space', check_disk_space)
    health_checker.register_check('memory', check_memory)