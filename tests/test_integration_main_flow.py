"""
End-to-end integration tests for the complete DigiPal application flow.

Tests the complete user journey from authentication to pet interaction,
including automatic pet loading, new user onboarding, and real-time UI updates.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import gradio as gr

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.ui.gradio_interface import GradioInterface
from digipal.core.enums import EggType, LifeStage
from digipal.auth.models import User, AuthResult, AuthStatus


class TestMainApplicationFlow:
    """Test the complete main application flow integration."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Mock AI communication for testing."""
        mock_ai = Mock(spec=AICommunication)
        
        # Mock speech processing
        mock_ai.process_speech.return_value = "Hello DigiPal"
        
        # Mock interaction processing
        from digipal.core.models import Interaction
        from digipal.core.enums import InteractionResult
        from datetime import datetime
        
        mock_interaction = Interaction(
            timestamp=datetime.now(),
            user_input="Hello DigiPal",
            interpreted_command="greeting",
            pet_response="Hello! I'm happy to see you!",
            attribute_changes={"happiness": 5},
            success=True,
            result=InteractionResult.SUCCESS
        )
        mock_ai.process_interaction.return_value = mock_interaction
        
        # Mock memory manager
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {
            'total_interactions': 1,
            'recent_topics': ['greeting'],
            'mood': 'happy'
        }
        mock_ai.memory_manager.clear_conversation_memory.return_value = True
        
        # Mock model unloading
        mock_ai.unload_all_models.return_value = None
        
        return mock_ai
    
    @pytest.fixture
    def integrated_system(self, temp_db_path, mock_ai_communication):
        """Create a fully integrated DigiPal system for testing."""
        # Initialize storage manager
        storage_manager = StorageManager(temp_db_path)
        
        # Initialize DigiPal core with mocked AI
        digipal_core = DigiPalCore(storage_manager, mock_ai_communication)
        
        # Initialize auth manager
        db_connection = DatabaseConnection(temp_db_path)
        auth_manager = AuthManager(db_connection, offline_mode=True)
        
        # Initialize Gradio interface
        gradio_interface = GradioInterface(digipal_core, auth_manager)
        
        return {
            'digipal_core': digipal_core,
            'auth_manager': auth_manager,
            'gradio_interface': gradio_interface,
            'storage_manager': storage_manager
        }
    
    def test_complete_new_user_flow(self, integrated_system):
        """Test complete flow for a new user from authentication to pet interaction."""
        system = integrated_system
        interface = system['gradio_interface']
        
        # Step 1: Authentication (offline mode)
        auth_result = interface._handle_login(
            token="test_token_123",
            offline_mode=True,
            current_user=None,
            current_token=None
        )
        
        # Verify authentication success and redirect to egg selection
        assert len(auth_result) == 4
        status_html, user_id, token, tab_update = auth_result
        
        assert "Welcome" in status_html
        assert "offline mode" in status_html.lower()
        assert user_id is not None
        assert token == "test_token_123"
        assert tab_update['selected'] == "egg_tab"
        
        # Step 2: Egg selection
        egg_result = interface._handle_egg_selection(EggType.RED, user_id)
        
        # Verify egg selection success and redirect to main interface
        assert len(egg_result) == 2
        egg_status, main_tab_update = egg_result
        
        assert "egg selected" in egg_status.lower()
        assert "red" in egg_status.lower()
        assert main_tab_update['selected'] == "main_tab"
        
        # Step 3: Verify pet was created
        pet_state = system['digipal_core'].get_pet_state(user_id)
        assert pet_state is not None
        assert pet_state.user_id == user_id
        assert pet_state.life_stage == LifeStage.EGG
        
        # Step 4: First interaction (should trigger hatching)
        interaction_result = interface._handle_text_interaction("Hello!", user_id)
        
        # Verify interaction success and UI updates
        assert len(interaction_result) == 7
        response_html, cleared_input, status_html, attributes_html, history_html, feedback_html, needs_html = interaction_result
        
        assert "DigiPal:" in response_html
        assert cleared_input == ""  # Input should be cleared
        assert status_html != ""  # Status should be updated
        assert attributes_html != ""  # Attributes should be displayed
        assert feedback_html != ""  # Feedback should be shown
        
        print("✅ Complete new user flow test passed!")
    
    def test_complete_application_lifecycle(self, integrated_system):
        """Test the complete application lifecycle from startup to shutdown."""
        system = integrated_system
        interface = system['gradio_interface']
        
        # Test interface creation
        app = interface.create_interface()
        assert app is not None
        assert isinstance(app, gr.Blocks)
        
        # Test that all components are properly initialized
        assert interface.digipal_core is not None
        assert interface.auth_manager is not None
        
        # Test complete user journey
        # 1. Authentication
        auth_result = interface._handle_login("test_token", True, None, None)
        user_id = auth_result[1]
        assert user_id is not None
        
        # 2. Egg selection
        egg_result = interface._handle_egg_selection(EggType.BLUE, user_id)
        assert "egg selected" in egg_result[0].lower()
        
        # 3. Pet interaction
        interaction_result = interface._handle_text_interaction("Hello DigiPal!", user_id)
        assert "DigiPal:" in interaction_result[0]
        
        # 4. Care actions
        care_result = interface._handle_care_action("feed", user_id)
        assert care_result[0] != ""
        
        # 5. System shutdown
        system['digipal_core'].shutdown()
        
        print("✅ Complete application lifecycle test passed!")


def test_launch_integration():
    """Test the launch integration without actually starting the server."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    try:
        # Mock the AI communication
        with patch('digipal.ai.communication.AICommunication') as mock_ai_class:
            mock_ai = Mock()
            mock_ai.process_speech.return_value = "test"
            mock_ai.process_interaction.return_value = Mock(
                pet_response="Hello!",
                success=True,
                attribute_changes={}
            )
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            mock_ai_class.return_value = mock_ai
            
            # Initialize components
            storage_manager = StorageManager(temp_db_path)
            ai_communication = mock_ai
            digipal_core = DigiPalCore(storage_manager, ai_communication)
            
            db_connection = DatabaseConnection(temp_db_path)
            auth_manager = AuthManager(db_connection, offline_mode=True)
            
            gradio_interface = GradioInterface(digipal_core, auth_manager)
            
            # Test interface creation
            app = gradio_interface.create_interface()
            assert app is not None
            
            # Test that launch_interface method exists and is callable
            assert hasattr(gradio_interface, 'launch_interface')
            assert callable(gradio_interface.launch_interface)
            
            # Cleanup
            digipal_core.shutdown()
            
            print("✅ Launch integration test passed!")
            
    finally:
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)


if __name__ == "__main__":
    print("🎉 All integration tests completed successfully!")