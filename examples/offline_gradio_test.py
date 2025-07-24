#!/usr/bin/env python3
"""
Offline test for DigiPal Gradio interface without requiring HuggingFace token.

This script demonstrates the complete DigiPal interface functionality
in offline mode for local development and testing.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.ui.gradio_interface import GradioInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_offline_components():
    """Set up all DigiPal components for offline testing."""
    logger.info("Setting up DigiPal components for offline testing...")
    
    # Create test database path
    test_db_path = project_root / "test_offline.db"
    
    # Initialize storage manager
    storage_manager = StorageManager(str(test_db_path))
    
    # Initialize AI communication (will use offline mode)
    ai_communication = AICommunication()
    
    # Initialize DigiPal core
    digipal_core = DigiPalCore(storage_manager, ai_communication)
    
    # Initialize auth manager with offline mode enabled
    from digipal.storage.database import DatabaseConnection
    db_connection = DatabaseConnection(str(test_db_path))
    auth_manager = AuthManager(db_connection, offline_mode=True)
    
    # Initialize Gradio interface
    gradio_interface = GradioInterface(digipal_core, auth_manager)
    
    logger.info("All components initialized successfully")
    return gradio_interface, digipal_core


def demonstrate_offline_features(digipal_core: DigiPalCore):
    """Demonstrate DigiPal features programmatically."""
    logger.info("Demonstrating DigiPal features...")
    
    # Create a test user first
    test_user_id = "offline_test_user"
    from digipal.auth.models import User
    from datetime import datetime
    
    test_user = User(
        id=test_user_id,
        username="OfflineTestUser",
        created_at=datetime.now()
    )
    
    # Save the user to database
    try:
        success = digipal_core.storage_manager.create_user(test_user_id, test_user.username, "")
        if success:
            logger.info(f"Created test user: {test_user.username}")
        else:
            logger.info(f"User {test_user.username} might already exist")
    except Exception as e:
        logger.info(f"Error creating user: {e}")
    
    # Create a new DigiPal
    from digipal.core.enums import EggType
    pet = digipal_core.create_new_pet(EggType.RED, test_user_id, "TestPal")
    logger.info(f"Created new DigiPal: {pet.name} (ID: {pet.id})")
    
    # Test text interaction
    success, interaction = digipal_core.process_interaction(test_user_id, "Hello DigiPal!")
    logger.info(f"Text interaction - Success: {success}, Response: {interaction.pet_response}")
    
    # Test care actions
    success, care_interaction = digipal_core.apply_care_action(test_user_id, "feed")
    logger.info(f"Care action (feed) - Success: {success}, Response: {care_interaction.pet_response}")
    
    # Get pet state
    pet_state = digipal_core.get_pet_state(test_user_id)
    if pet_state:
        logger.info(f"Pet state - Name: {pet_state.name}, Stage: {pet_state.life_stage.value}, "
                   f"HP: {pet_state.hp}, Energy: {pet_state.energy}, Happiness: {pet_state.happiness}")
    
    # Get statistics
    stats = digipal_core.get_pet_statistics(test_user_id)
    logger.info(f"Pet statistics loaded: {len(stats)} categories")
    
    return test_user_id


def main():
    """Main function to run offline DigiPal test."""
    print("🥚 DigiPal Offline Test")
    print("=" * 50)
    print("This test runs DigiPal in offline mode without requiring a HuggingFace token.")
    print("You can test all interface features locally.")
    print()
    
    try:
        # Set up components
        gradio_interface, digipal_core = setup_offline_components()
        
        # Demonstrate features programmatically
        test_user_id = demonstrate_offline_features(digipal_core)
        
        print("\n✅ Offline setup complete!")
        print("\nStarting Gradio interface...")
        print("📝 Instructions:")
        print("1. Open the web interface in your browser")
        print("2. Check 'Enable Offline Mode (Development)'")
        print("3. Enter any token (it will be ignored in offline mode)")
        print("4. Click 'Login' to proceed")
        print("5. Test all interface features!")
        print()
        print("🔗 Interface will be available at: http://127.0.0.1:7860")
        print("Press Ctrl+C to stop the server")
        print()
        
        # Launch the interface
        gradio_interface.launch_interface(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down DigiPal offline test...")
        digipal_core.shutdown()
        print("Goodbye!")
        
    except Exception as e:
        logger.error(f"Error running offline test: {e}")
        print(f"\n❌ Error: {e}")
        print("Check the logs for more details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())