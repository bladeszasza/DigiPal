#!/usr/bin/env python3
"""
Launch DigiPal with proper setup for production deployment.
Includes health checks, monitoring, and configuration management.
"""

import sys
import os
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.ui.gradio_interface import GradioInterface
from digipal.monitoring import (
    metrics, health_checker, start_metrics_server, 
    setup_default_health_checks, get_metrics
)
from config import get_config

# Global references for cleanup
digipal_core: Optional[DigiPalCore] = None
gradio_interface: Optional[GradioInterface] = None

# Configure logging based on config
config = get_config()
logging.config.dictConfig(config.get_log_config())
logger = logging.getLogger(__name__)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_application()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def shutdown_application():
    """Gracefully shutdown all application components."""
    global digipal_core, gradio_interface
    
    logger.info("Shutting down DigiPal components...")
    
    if digipal_core:
        try:
            digipal_core.shutdown()
            logger.info("DigiPal core shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down DigiPal core: {e}")
    
    if gradio_interface:
        try:
            # Gradio doesn't have explicit shutdown, but we can log
            logger.info("Gradio interface shutdown initiated")
        except Exception as e:
            logger.error(f"Error shutting down Gradio interface: {e}")


def setup_health_checks(storage_manager, ai_communication, auth_manager):
    """Setup application-specific health checks."""
    
    def check_database():
        """Check database connectivity and basic operations."""
        try:
            # Test database connection
            storage_manager.get_connection()
            return {'message': 'Database connection OK'}
        except Exception as e:
            raise Exception(f"Database check failed: {e}")
    
    def check_ai_models():
        """Check AI model availability and basic functionality."""
        try:
            # Test AI communication system
            if hasattr(ai_communication, 'language_model') and ai_communication.language_model:
                return {'message': 'AI models loaded and ready'}
            else:
                return {'message': 'AI models not loaded (offline mode)'}
        except Exception as e:
            raise Exception(f"AI model check failed: {e}")
    
    def check_authentication():
        """Check authentication system."""
        try:
            # Test auth manager
            if auth_manager:
                return {'message': 'Authentication system ready'}
            else:
                raise Exception("Auth manager not initialized")
        except Exception as e:
            raise Exception(f"Authentication check failed: {e}")
    
    # Register application-specific health checks
    health_checker.register_check('database', check_database)
    health_checker.register_check('ai_models', check_ai_models)
    health_checker.register_check('authentication', check_authentication)


def create_health_endpoint():
    """Create health check endpoint for the application."""
    import gradio as gr
    
    def health_check():
        """Health check endpoint handler."""
        try:
            health_status = health_checker.get_health_status()
            return {
                "status": health_status["status"],
                "timestamp": health_status["timestamp"],
                "checks": health_status["checks"]
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }
    
    def metrics_endpoint():
        """Metrics endpoint handler."""
        try:
            return get_metrics()
        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return f"Error: {e}"
    
    return health_check, metrics_endpoint


def main():
    """Main function to launch DigiPal with full production setup."""
    global digipal_core, gradio_interface
    
    logger.info("🥚 Launching DigiPal...")
    logger.info(f"Environment: {config.env}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    try:
        # Start metrics server if enabled
        if config.env == "production":
            try:
                start_metrics_server(port=8000)
                logger.info("📊 Metrics server started on port 8000")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")
        
        # Setup default health checks
        setup_default_health_checks()
        
        # Initialize storage manager
        db_path = config.database.path
        storage_manager = StorageManager(db_path)
        logger.info(f"💾 Storage manager initialized with database: {db_path}")
        
        # Initialize AI communication
        ai_communication = AICommunication()
        logger.info("🤖 AI communication system initialized")
        
        # Initialize DigiPal core
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        logger.info("🎮 DigiPal core engine initialized")
        
        # Initialize auth manager
        db_connection = DatabaseConnection(db_path)
        auth_manager = AuthManager(db_connection)
        logger.info("🔐 Authentication manager initialized")
        
        # Setup application-specific health checks
        setup_health_checks(storage_manager, ai_communication, auth_manager)
        
        # Initialize Gradio interface
        gradio_interface = GradioInterface(digipal_core, auth_manager)
        logger.info("🌐 Gradio interface initialized")
        
        # Create health and metrics endpoints
        health_check, metrics_endpoint = create_health_endpoint()
        
        logger.info("✅ All components initialized successfully!")
        
        # Display startup information
        print(f"\n🎮 DigiPal is starting up...")
        print(f"📊 Environment: {config.env}")
        print(f"🌐 Web interface: http://{config.gradio.server_name}:{config.gradio.server_port}")
        print(f"🏥 Health check: http://{config.gradio.server_name}:{config.gradio.server_port}/health")
        if config.env == "production":
            print(f"📈 Metrics: http://localhost:8000/metrics")
        print()
        print("📝 Instructions:")
        print("1. Open the web interface URL in your browser")
        print("2. For offline mode: check 'Enable Offline Mode' and enter any token")
        print("3. For online mode: enter your HuggingFace token")
        print("4. Start interacting with your DigiPal!")
        print()
        print("Press Ctrl+C to stop the server")
        
        # Record startup metrics
        metrics.record_http_request("GET", "/startup", 200, 0.0)
        
        # Launch the interface with configuration
        gradio_interface.launch_interface(
            share=config.gradio.share,
            server_name=config.gradio.server_name,
            server_port=config.gradio.server_port,
            debug=config.gradio.debug,
            ssl_keyfile=config.gradio.ssl_keyfile,
            ssl_certfile=config.gradio.ssl_certfile
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown_application()
        print("\n👋 DigiPal shutdown complete. Goodbye!")
        
    except Exception as e:
        logger.error(f"Error launching DigiPal: {e}", exc_info=True)
        metrics.record_error("startup_error", "critical")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())