#!/usr/bin/env python3
"""
Simple launcher script for DigiPal Gradio interface.
"""

import sys
import os
import logging
from pathlib import Path
from unittest.mock import Mock

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from digipal.ui.gradio_interface import GradioInterface
from digipal.core.digipal_core import DigiPalCore
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_components():
    """Create components for DigiPal interface."""
    
    try:
        # Create temporary database for demo
        demo_db_path = project_root / "demo_digipal.db"
        
        # Initialize database connection
        db_connection = DatabaseConnection(str(demo_db_path))
        
        # Create mock DigiPal core (since full implementation may not be ready)
        mock_digipal_core = Mock(spec=DigiPalCore)
        mock_digipal_core.load_existing_pet.return_value = None
        mock_digipal_core.create_new_pet.return_value = Mock(id="test_pet")
        mock_digipal_core.process_interaction.return_value = (True, Mock(pet_response="Hello! I'm your DigiPal!"))
        mock_digipal_core.apply_care_action.return_value = (True, Mock(pet_response="Thanks for taking care of me!"))
        mock_digipal_core.get_pet_state.return_value = Mock(
            name="TestPal",
            life_stage=Mock(value="baby"),
            age_hours=2.5,
            status_summary="Happy",
            hp=80,
            energy=75,
            happiness=90,
            weight=25
        )
        mock_digipal_core.get_pet_statistics.return_value = {
            'interaction_summary': {'recent_interactions': []}
        }
        
        # Initialize authentication manager (with real HuggingFace support)
        auth_manager = AuthManager(db_connection, offline_mode=False)
        
        logger.info("Components created successfully")
        return mock_digipal_core, auth_manager
        
    except Exception as e:
        logger.error(f"Error creating components: {e}")
        raise


def main():
    """Main launcher function."""
    print("\n" + "="*60)
    print("🥚 DigiPal - Your Digital Companion")
    print("="*60)
    
    try:
        # Create components
        digipal_core, auth_manager = create_components()
        
        # Create Gradio interface
        interface = GradioInterface(digipal_core, auth_manager)
        
        # Find available port
        import socket
        def find_free_port():
            for port in range(7860, 7870):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    continue
            return 7860
        
        port = find_free_port()
        
        print(f"Starting DigiPal interface on http://localhost:{port}")
        print("\nHow to use:")
        print("1. Open the URL in your browser")
        print("2. In the Login tab:")
        print("   - Enter your HuggingFace token: hf_EXAMPLE_TOKEN_REMOVED")
        print("   - Click 'Login' (offline mode not needed with valid token)")
        print("3. Choose an egg type in the 'Choose Your Egg' tab")
        print("4. Interact with your DigiPal in the 'Your DigiPal' tab")
        print("\nPress Ctrl+C to stop the server")
        print("="*60)
        
        # Launch interface
        interface.launch_interface(
            share=False,
            server_name="127.0.0.1",
            server_port=port,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n\nDigiPal interface stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have gradio installed:")
        print("pip install gradio")
    finally:
        # Cleanup
        demo_db_path = project_root / "demo_digipal.db"
        if demo_db_path.exists():
            try:
                demo_db_path.unlink()
                print("Cleaned up demo database")
            except Exception as e:
                print(f"Could not clean up demo database: {e}")


if __name__ == "__main__":
    main()