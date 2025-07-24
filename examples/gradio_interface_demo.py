"""
Demo script for DigiPal Gradio interface.

This script demonstrates the basic functionality of the Gradio web interface
with mock components for development and testing purposes.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from digipal.ui.gradio_interface import GradioInterface
from digipal.core.digipal_core import DigiPalCore
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_mock_components():
    """Create mock components for demo purposes."""
    
    # Create temporary database for demo
    demo_db_path = project_root / "demo_digipal.db"
    
    try:
        # Initialize database connection
        db_connection = DatabaseConnection(str(demo_db_path))
        
        # Initialize storage manager
        storage_manager = StorageManager(db_connection, str(project_root / "demo_assets"))
        
        # Initialize AI communication (with mock models for demo)
        ai_communication = AICommunication(
            language_model_path="mock",  # Will use mock in demo
            speech_model_config={"mock": True},
            offline_mode=True
        )
        
        # Initialize DigiPal core
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize authentication manager (offline mode for demo)
        auth_manager = AuthManager(db_connection, offline_mode=True)
        
        logger.info("Mock components created successfully")
        return digipal_core, auth_manager
        
    except Exception as e:
        logger.error(f"Error creating mock components: {e}")
        # Create minimal mock objects if initialization fails
        from unittest.mock import Mock
        
        mock_digipal_core = Mock(spec=DigiPalCore)
        mock_auth_manager = Mock(spec=AuthManager)
        
        # Setup basic mock behaviors
        mock_auth_manager.offline_mode = True
        mock_digipal_core.load_existing_pet.return_value = None
        
        logger.info("Using minimal mock components due to initialization error")
        return mock_digipal_core, mock_auth_manager


def main():
    """Main demo function."""
    logger.info("Starting DigiPal Gradio Interface Demo")
    
    try:
        # Create components
        digipal_core, auth_manager = create_mock_components()
        
        # Create Gradio interface
        interface = GradioInterface(digipal_core, auth_manager)
        
        # Create and launch the interface
        logger.info("Creating Gradio interface...")
        gradio_app = interface.create_interface()
        
        logger.info("Launching interface on http://localhost:7860")
        print("\n" + "="*60)
        print("🥚 DigiPal Gradio Interface Demo")
        print("="*60)
        print("Interface will open in your browser at: http://localhost:7860")
        print("\nDemo Features:")
        print("• Authentication tab with offline mode")
        print("• Egg selection interface with three egg types")
        print("• Main DigiPal interface with care actions")
        print("• Game-style UI with custom CSS")
        print("\nTo test:")
        print("1. Enable 'Offline Mode' in the login tab")
        print("2. Enter any token (e.g., 'demo_token_123')")
        print("3. Click Login to proceed to egg selection")
        print("4. Choose an egg type to create your DigiPal")
        print("5. Interact with your DigiPal in the main interface")
        print("\nPress Ctrl+C to stop the demo")
        print("="*60)
        
        # Launch interface
        interface.launch_interface(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            debug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        print(f"\nError: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install gradio")
    finally:
        # Cleanup
        demo_db_path = project_root / "demo_digipal.db"
        if demo_db_path.exists():
            try:
                demo_db_path.unlink()
                logger.info("Cleaned up demo database")
            except Exception as e:
                logger.warning(f"Could not clean up demo database: {e}")


if __name__ == "__main__":
    main()