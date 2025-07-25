"""
DigiPal Configuration Management
Handles environment-specific configuration for deployment
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str = "digipal.db"
    backup_dir: str = "assets/backups"
    backup_interval_hours: int = 24
    max_backups: int = 10


@dataclass
class AIModelConfig:
    """AI model configuration settings"""
    qwen_model: str = "Qwen/Qwen3-0.6B"
    kyutai_model: str = "kyutai/stt-2.6b-en_fr-trfs"
    flux_model: str = "black-forest-labs/FLUX.1-dev"
    device: str = "auto"
    torch_dtype: str = "auto"
    enable_quantization: bool = True
    max_memory_gb: Optional[int] = None


@dataclass
class GradioConfig:
    """Gradio interface configuration"""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    debug: bool = False
    auth: Optional[tuple] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None


@dataclass
class MCPConfig:
    """MCP server configuration"""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8080
    max_connections: int = 100
    timeout_seconds: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/digipal.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_structured_logging: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: Optional[str] = None
    session_timeout_hours: int = 24
    max_login_attempts: int = 5
    rate_limit_per_minute: int = 60
    enable_cors: bool = True
    allowed_origins: list = None


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    cache_size_mb: int = 512
    background_update_interval: int = 60
    max_concurrent_users: int = 100
    enable_model_caching: bool = True
    image_cache_max_age_days: int = 30


class DigiPalConfig:
    """Main configuration class for DigiPal application"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv("DIGIPAL_ENV", "development")
        self.load_config()
    
    def load_config(self):
        """Load configuration based on environment"""
        # Base configuration
        self.database = DatabaseConfig()
        self.ai_models = AIModelConfig()
        self.gradio = GradioConfig()
        self.mcp = MCPConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Environment-specific overrides
        if self.env == "production":
            self._load_production_config()
        elif self.env == "testing":
            self._load_testing_config()
        elif self.env == "development":
            self._load_development_config()
        
        # Load from environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _load_production_config(self):
        """Production environment configuration"""
        self.database.path = "/app/data/digipal.db"
        self.database.backup_dir = "/app/data/backups"
        
        self.gradio.debug = False
        self.gradio.share = False
        
        self.logging.level = "INFO"
        self.logging.file_path = "/app/logs/digipal.log"
        
        self.security.session_timeout_hours = 12
        self.security.rate_limit_per_minute = 30
        
        self.performance.cache_size_mb = 1024
        self.performance.max_concurrent_users = 500
    
    def _load_testing_config(self):
        """Testing environment configuration"""
        self.database.path = "test_digipal.db"
        self.database.backup_dir = "test_assets/backups"
        
        self.gradio.debug = True
        self.gradio.server_port = 7861
        
        self.logging.level = "DEBUG"
        self.logging.file_path = None  # Console only
        
        self.ai_models.enable_quantization = False
        self.performance.cache_size_mb = 128
    
    def _load_development_config(self):
        """Development environment configuration"""
        self.gradio.debug = True
        self.gradio.share = False
        
        self.logging.level = "DEBUG"
        self.logging.file_path = "logs/digipal_dev.log"
        
        self.security.rate_limit_per_minute = 120
        self.performance.cache_size_mb = 256
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Database
        if os.getenv("DIGIPAL_DB_PATH"):
            self.database.path = os.getenv("DIGIPAL_DB_PATH")
        
        # Gradio
        if os.getenv("GRADIO_SERVER_NAME"):
            self.gradio.server_name = os.getenv("GRADIO_SERVER_NAME")
        if os.getenv("GRADIO_SERVER_PORT"):
            self.gradio.server_port = int(os.getenv("GRADIO_SERVER_PORT"))
        if os.getenv("GRADIO_SHARE"):
            self.gradio.share = os.getenv("GRADIO_SHARE").lower() == "true"
        
        # Logging
        if os.getenv("DIGIPAL_LOG_LEVEL"):
            self.logging.level = os.getenv("DIGIPAL_LOG_LEVEL")
        if os.getenv("DIGIPAL_LOG_FILE"):
            self.logging.file_path = os.getenv("DIGIPAL_LOG_FILE")
        
        # Security
        if os.getenv("DIGIPAL_SECRET_KEY"):
            self.security.secret_key = os.getenv("DIGIPAL_SECRET_KEY")
        
        # AI Models
        if os.getenv("QWEN_MODEL"):
            self.ai_models.qwen_model = os.getenv("QWEN_MODEL")
        if os.getenv("KYUTAI_MODEL"):
            self.ai_models.kyutai_model = os.getenv("KYUTAI_MODEL")
        if os.getenv("FLUX_MODEL"):
            self.ai_models.flux_model = os.getenv("FLUX_MODEL")
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Ensure required directories exist
        Path(self.database.backup_dir).mkdir(parents=True, exist_ok=True)
        
        if self.logging.file_path:
            Path(self.logging.file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Validate port ranges
        if not (1024 <= self.gradio.server_port <= 65535):
            raise ValueError(f"Invalid Gradio port: {self.gradio.server_port}")
        
        if not (1024 <= self.mcp.port <= 65535):
            raise ValueError(f"Invalid MCP port: {self.mcp.port}")
        
        # Validate memory settings
        if self.performance.cache_size_mb < 64:
            logging.warning("Cache size is very low, performance may be affected")
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"sqlite:///{self.database.path}"
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary"""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.logging.format
                },
                "structured": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": "structlog.dev.ConsoleRenderer",
                } if self.logging.enable_structured_logging else {
                    "format": self.logging.format
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.logging.level,
                    "formatter": "structured" if self.logging.enable_structured_logging else "standard",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "digipal": {
                    "level": self.logging.level,
                    "handlers": ["console"],
                    "propagate": False
                }
            },
            "root": {
                "level": self.logging.level,
                "handlers": ["console"]
            }
        }
        
        # Add file handler if specified
        if self.logging.file_path:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self.logging.level,
                "formatter": "standard",
                "filename": self.logging.file_path,
                "maxBytes": self.logging.max_file_size_mb * 1024 * 1024,
                "backupCount": self.logging.backup_count
            }
            config["loggers"]["digipal"]["handlers"].append("file")
            config["root"]["handlers"].append("file")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "env": self.env,
            "database": self.database.__dict__,
            "ai_models": self.ai_models.__dict__,
            "gradio": self.gradio.__dict__,
            "mcp": self.mcp.__dict__,
            "logging": self.logging.__dict__,
            "security": self.security.__dict__,
            "performance": self.performance.__dict__
        }


# Global configuration instance
config = DigiPalConfig()


def get_config() -> DigiPalConfig:
    """Get the global configuration instance"""
    return config


def reload_config(env: str = None):
    """Reload configuration with optional environment override"""
    global config
    config = DigiPalConfig(env)
    return config