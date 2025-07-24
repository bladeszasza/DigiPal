#!/usr/bin/env python3
"""
Launch DigiPal with proper setup for testing.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.ui.gradio_interface import GradioInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main function to launch DigiPal."""
    print("🥚 Launching DigiPal...")
    
    try:
        # Create database path
        db_path = project_root / "test.db"
        
        # Initialize storage manager
        storage_manager = StorageManager(str(db_path))
        
        # Initialize AI communication
        ai_communication = AICommunication()
        
        # Initialize DigiPal core
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize auth manager
        db_connection = DatabaseConnection(str(db_path))
        auth_manager = AuthManager(db_connection)
        
        # Initialize Gradio interface
        gradio_interface = GradioInterface(digipal_core, auth_manager)
        
        print("✅ All components initialized successfully!")
        print("🌐 Starting web interface...")
        print("📝 Instructions:")
        print("1. Open http://127.0.0.1:7860 in your browser")
        print("2. For offline mode: check 'Enable Offline Mode' and enter any token")
        print("3. For online mode: enter your HuggingFace token")
        print("4. Test the tab switching after login!")
        print()
        print("Press Ctrl+C to stop the server")
        
        # Launch the interface
        gradio_interface.launch_interface(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down DigiPal...")
        digipal_core.shutdown()
        print("Goodbye!")
        
    except Exception as e:
        logger.error(f"Error launching DigiPal: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())