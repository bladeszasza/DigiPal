"""
Tests for Gradio interface components.
"""

import pytest
import gradio as gr
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from digipal.ui.gradio_interface import GradioInterface
from digipal.core.digipal_core import DigiPalCore, PetState
from digipal.core.enums import EggType, LifeStage
from digipal.auth.auth_manager import AuthManager
from digipal.auth.models import AuthResult, AuthStatus, User


class TestGradioInterface:
    """Test cases for GradioInterface class."""
    
    @pytest.fixture
    def mock_digipal_core(self):
        """Create mock DigiPal core."""
        mock_core = Mock(spec=DigiPalCore)
        return mock_core
    
    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock authentication manager."""
        mock_auth = Mock(spec=AuthManager)
        return mock_auth
    
    @pytest.fixture
    def gradio_interface(self, mock_digipal_core, mock_auth_manager):
        """Create GradioInterface instance."""
        return GradioInterface(mock_digipal_core, mock_auth_manager)
    
    def test_initialization(self, gradio_interface, mock_digipal_core, mock_auth_manager):
        """Test GradioInterface initialization."""
        assert gradio_interface.digipal_core == mock_digipal_core
        assert gradio_interface.auth_manager == mock_auth_manager
        assert gradio_interface.current_user_id is None
        assert gradio_interface.current_token is None
        assert gradio_interface.app is None
    
    def test_create_interface(self, gradio_interface):
        """Test interface creation."""
        interface = gradio_interface.create_interface()
        
        assert isinstance(interface, gr.Blocks)
        assert gradio_interface.app == interface
    
    def test_custom_css_generation(self, gradio_interface):
        """Test custom CSS generation."""
        css = gradio_interface._get_custom_css()
        
        assert isinstance(css, str)
        assert len(css) > 0
        assert ".gradio-container" in css
        assert ".auth-container" in css
        assert ".egg-selection-container" in css
        assert ".pet-display-column" in css
    
    def test_authentication_tab_creation(self, gradio_interface):
        """Test authentication tab component creation."""
        # Create components within proper Gradio context
        with gr.Blocks() as test_interface:
            auth_components = gradio_interface._create_authentication_tab()
        
        # Check required components exist
        required_components = [
            'token_input', 'login_btn', 'auth_status', 'offline_toggle'
        ]
        
        for component in required_components:
            assert component in auth_components
            assert auth_components[component] is not None
    
    def test_egg_selection_interface_creation(self, gradio_interface):
        """Test egg selection interface creation."""
        # Create components within proper Gradio context
        with gr.Blocks() as test_interface:
            egg_components = gradio_interface._create_egg_selection_interface()
        
        # Check required components exist
        required_components = [
            'red_egg_btn', 'blue_egg_btn', 'green_egg_btn', 'egg_status'
        ]
        
        for component in required_components:
            assert component in egg_components
            assert egg_components[component] is not None
    
    def test_main_interface_creation(self, gradio_interface):
        """Test main DigiPal interface creation."""
        # Create components within proper Gradio context
        with gr.Blocks() as test_interface:
            main_components = gradio_interface._create_digipal_main_interface()
        
        # Check required components exist
        required_components = [
            'pet_image', 'pet_name_display', 'status_info', 'attributes_display',
            'feed_btn', 'train_btn', 'praise_btn', 'scold_btn', 'rest_btn', 'play_btn',
            'audio_input', 'text_input', 'send_btn', 'response_display', 'conversation_history'
        ]
        
        for component in required_components:
            assert component in main_components
            assert main_components[component] is not None
    
    def test_handle_login_success(self, gradio_interface, mock_auth_manager, mock_digipal_core):
        """Test successful login handling."""
        # Setup mocks
        mock_user = User(
            id="test_user",
            username="testuser",
            created_at=datetime.now()
        )
        
        mock_auth_result = AuthResult(
            status=AuthStatus.SUCCESS,
            user=mock_user
        )
        
        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_digipal_core.load_existing_pet.return_value = None  # New user
        
        # Test login
        result = gradio_interface._handle_login("test_token", False, None, None)
        
        # Verify results
        auth_status, user_id, token, tabs = result
        
        assert "Welcome, testuser" in auth_status
        assert user_id == "test_user"
        assert token == "test_token"
        assert gradio_interface.current_user_id == "test_user"
        assert gradio_interface.current_token == "test_token"
    
    def test_handle_login_existing_user(self, gradio_interface, mock_auth_manager, mock_digipal_core):
        """Test login with existing DigiPal."""
        # Setup mocks
        mock_user = User(
            id="existing_user",
            username="existinguser",
            created_at=datetime.now()
        )
        
        mock_auth_result = AuthResult(
            status=AuthStatus.SUCCESS,
            user=mock_user
        )
        
        mock_pet = Mock()
        mock_pet.id = "pet_123"
        
        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_digipal_core.load_existing_pet.return_value = mock_pet
        
        # Test login
        result = gradio_interface._handle_login("test_token", False, None, None)
        
        # Verify results
        auth_status, user_id, token, tabs = result
        
        assert "Welcome back, existinguser" in auth_status
        assert user_id == "existing_user"
    
    def test_handle_login_failure(self, gradio_interface, mock_auth_manager):
        """Test failed login handling."""
        # Setup mock for failed authentication
        mock_auth_result = AuthResult(
            status=AuthStatus.INVALID_TOKEN,
            error_message="Invalid token"
        )
        
        mock_auth_manager.authenticate.return_value = mock_auth_result
        
        # Test login
        result = gradio_interface._handle_login("bad_token", False, None, None)
        
        # Verify results
        auth_status, user_id, token, tabs = result
        
        assert "Login failed" in auth_status
        assert "Invalid token" in auth_status
        assert user_id is None
        assert token is None
    
    def test_handle_login_offline_mode(self, gradio_interface, mock_auth_manager, mock_digipal_core):
        """Test offline mode login."""
        # Setup mocks
        mock_user = User(
            id="offline_user",
            username="offlineuser",
            created_at=datetime.now()
        )
        
        mock_auth_result = AuthResult(
            status=AuthStatus.OFFLINE_MODE,
            user=mock_user
        )
        
        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_digipal_core.load_existing_pet.return_value = None
        
        # Test offline login
        result = gradio_interface._handle_login("test_token", True, None, None)
        
        # Verify results
        auth_status, user_id, token, tabs = result
        
        assert "Welcome, offlineuser" in auth_status
        assert "Running in offline mode" in auth_status
        assert mock_auth_manager.offline_mode is True
    
    def test_handle_egg_selection(self, gradio_interface, mock_digipal_core):
        """Test egg selection handling."""
        # Setup mock
        mock_pet = Mock()
        mock_pet.id = "new_pet_123"
        mock_digipal_core.create_new_pet.return_value = mock_pet
        
        # Test egg selection
        result = gradio_interface._handle_egg_selection(EggType.RED, "test_user")
        
        # Verify results
        egg_status, tabs = result
        
        assert "Egg Selected" in egg_status
        assert "red egg" in egg_status
        mock_digipal_core.create_new_pet.assert_called_once_with(EggType.RED, "test_user")
    
    def test_handle_egg_selection_no_user(self, gradio_interface):
        """Test egg selection without logged in user."""
        result = gradio_interface._handle_egg_selection(EggType.RED, None)
        
        egg_status, tabs = result
        
        assert "Please login first" in egg_status
    
    def test_handle_text_interaction(self, gradio_interface, mock_digipal_core):
        """Test text interaction handling."""
        # Setup mocks
        mock_interaction = Mock()
        mock_interaction.pet_response = "Hello there!"
        mock_interaction.success = True
        
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.BABY
        mock_pet_state.age_hours = 2.5
        mock_pet_state.status_summary = "Happy"
        mock_pet_state.hp = 80
        mock_pet_state.energy = 75
        mock_pet_state.happiness = 90
        mock_pet_state.weight = 25
        mock_pet_state.needs_attention = False
        
        mock_digipal_core.process_interaction.return_value = (True, mock_interaction)
        mock_digipal_core.get_pet_state.return_value = mock_pet_state
        mock_digipal_core.get_pet_statistics.return_value = {
            'interaction_summary': {'recent_interactions': []}
        }
        
        # Test interaction
        result = gradio_interface._handle_text_interaction("Hello DigiPal", "test_user")
        
        # Verify results (now returns 7 values)
        response, cleared_input, status, attributes, history, feedback, needs = result
        
        assert "Hello there!" in response
        assert cleared_input == ""  # Input should be cleared
        assert "TestPal" in status
        assert "Baby" in status
        assert "Message sent!" in feedback
        mock_digipal_core.process_interaction.assert_called_once_with("test_user", "Hello DigiPal")
    
    def test_handle_care_action(self, gradio_interface, mock_digipal_core):
        """Test care action handling."""
        # Setup mocks
        mock_interaction = Mock()
        mock_interaction.pet_response = "Thanks for feeding me!"
        mock_interaction.success = True
        
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.CHILD
        mock_pet_state.age_hours = 10.0
        mock_pet_state.status_summary = "Well fed"
        mock_pet_state.hp = 85
        mock_pet_state.energy = 80
        mock_pet_state.happiness = 95
        mock_pet_state.weight = 30
        mock_pet_state.needs_attention = False
        
        mock_digipal_core.apply_care_action.return_value = (True, mock_interaction)
        mock_digipal_core.get_pet_state.return_value = mock_pet_state
        
        # Test care action
        result = gradio_interface._handle_care_action("feed", "test_user")
        
        # Verify results (now returns 5 values)
        response, status, attributes, feedback, needs = result
        
        assert "Care Action:</strong> 🍖 Feed" in response
        assert "Thanks for feeding me!" in response
        assert "TestPal" in status
        assert "Feed completed!" in feedback
        mock_digipal_core.apply_care_action.assert_called_once_with("test_user", "feed")
    
    def test_format_pet_status(self, gradio_interface):
        """Test pet status formatting."""
        # Create mock pet state
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.TEEN
        mock_pet_state.age_hours = 48.5
        mock_pet_state.status_summary = "Energetic"
        mock_pet_state.hp = 90
        mock_pet_state.energy = 85
        mock_pet_state.happiness = 75
        mock_pet_state.weight = 35
        
        # Test formatting
        status_html, attributes_html = gradio_interface._format_pet_status(mock_pet_state)
        
        # Verify status HTML
        assert "TestPal" in status_html
        assert "Teen" in status_html
        assert "48.5 hours" in status_html
        assert "Energetic" in status_html
        
        # Verify attributes HTML
        assert "HP" in attributes_html
        assert "Energy" in attributes_html
        assert "Happiness" in attributes_html
        assert "Weight" in attributes_html
        assert "90" in attributes_html  # HP value
        assert "85" in attributes_html  # Energy value
    
    def test_format_pet_status_none(self, gradio_interface):
        """Test pet status formatting with None input."""
        status_html, attributes_html = gradio_interface._format_pet_status(None)
        
        assert status_html == ""
        assert attributes_html == ""
    
    def test_format_conversation_history(self, gradio_interface, mock_digipal_core):
        """Test conversation history formatting."""
        # Setup mock
        mock_interactions = [
            {'user_input': 'Hello', 'pet_response': 'Hi there!'},
            {'user_input': 'How are you?', 'pet_response': 'I am doing great!'}
        ]
        
        mock_stats = {
            'interaction_summary': {
                'recent_interactions': mock_interactions
            }
        }
        
        mock_digipal_core.get_pet_statistics.return_value = mock_stats
        
        # Test formatting
        history_html = gradio_interface._format_conversation_history("test_user")
        
        # Verify content
        assert "Hello" in history_html
        assert "Hi there!" in history_html
        assert "How are you?" in history_html
        assert "I am doing great!" in history_html
        assert "history-item" in history_html
    
    def test_format_conversation_history_empty(self, gradio_interface, mock_digipal_core):
        """Test conversation history formatting with no interactions."""
        mock_stats = {
            'interaction_summary': {
                'recent_interactions': []
            }
        }
        
        mock_digipal_core.get_pet_statistics.return_value = mock_stats
        
        # Test formatting
        history_html = gradio_interface._format_conversation_history("test_user")
        
        assert "No conversation history yet" in history_html
    
    @patch('digipal.ui.gradio_interface.logger')
    def test_format_conversation_history_error(self, mock_logger, gradio_interface, mock_digipal_core):
        """Test conversation history formatting with error."""
        # Setup mock to raise exception
        mock_digipal_core.get_pet_statistics.side_effect = Exception("Database error")
        
        # Test formatting
        history_html = gradio_interface._format_conversation_history("test_user")
        
        assert "Error loading conversation history" in history_html
        mock_logger.error.assert_called_once()
    
    def test_handle_text_interaction_no_user(self, gradio_interface):
        """Test text interaction without user."""
        result = gradio_interface._handle_text_interaction("Hello", None)
        
        response, text, status, attributes, history, feedback, needs = result
        
        assert "Please enter a message" in response
    
    def test_handle_text_interaction_empty_text(self, gradio_interface):
        """Test text interaction with empty text."""
        result = gradio_interface._handle_text_interaction("", "test_user")
        
        response, text, status, attributes, history, feedback, needs = result
        
        assert "Please enter a message" in response
    
    def test_handle_care_action_no_user(self, gradio_interface):
        """Test care action without user."""
        result = gradio_interface._handle_care_action("feed", None)
        
        response, status, attributes, feedback, needs = result
        
        assert "Please login first" in response


class TestGradioInterfaceIntegration:
    """Integration tests for GradioInterface."""
    
    @pytest.fixture
    def mock_components(self):
        """Create all necessary mock components."""
        with patch('digipal.ui.gradio_interface.DatabaseConnection') as mock_db:
            mock_digipal_core = Mock(spec=DigiPalCore)
            mock_auth_manager = Mock(spec=AuthManager)
            
            return {
                'digipal_core': mock_digipal_core,
                'auth_manager': mock_auth_manager,
                'db': mock_db
            }
    
    def test_full_interface_creation(self, mock_components):
        """Test complete interface creation without errors."""
        interface = GradioInterface(
            mock_components['digipal_core'],
            mock_components['auth_manager']
        )
        
        # This should not raise any exceptions
        gradio_app = interface.create_interface()
        
        assert isinstance(gradio_app, gr.Blocks)
        assert interface.app is not None
    
    def test_enhanced_main_interface_components(self, mock_components):
        """Test all enhanced main interface components are created."""
        interface = GradioInterface(
            mock_components['digipal_core'],
            mock_components['auth_manager']
        )
        
        with gr.Blocks() as test_interface:
            main_components = interface._create_digipal_main_interface()
        
        # Check all enhanced components exist
        enhanced_components = [
            'action_feedback', 'needs_display', 'auto_refresh',
            'strength_train_btn', 'speed_train_btn', 'brain_train_btn', 'defense_train_btn',
            'medicine_btn', 'clean_btn', 'audio_status', 'process_audio_btn',
            'quick_hello_btn', 'quick_status_btn', 'clear_history_btn', 'export_history_btn'
        ]
        
        for component in enhanced_components:
            assert component in main_components
            assert main_components[component] is not None
    
    def test_launch_interface_parameters(self, mock_components):
        """Test interface launch with different parameters."""
        interface = GradioInterface(
            mock_components['digipal_core'],
            mock_components['auth_manager']
        )
        
        # Create interface first
        interface.create_interface()
        
        # Mock the launch method to avoid actually starting server
        with patch.object(interface.app, 'launch') as mock_launch:
            interface.launch_interface(
                share=True,
                server_name="0.0.0.0",
                server_port=8080,
                debug=True
            )
            
            mock_launch.assert_called_once_with(
                share=True,
                server_name="0.0.0.0",
                server_port=8080,
                debug=True,
                show_error=True
            )


class TestEnhancedInteractionComponents:
    """Test enhanced interaction components and features."""
    
    @pytest.fixture
    def mock_digipal_core(self):
        """Create mock DigiPal core with enhanced methods."""
        mock_core = Mock(spec=DigiPalCore)
        return mock_core
    
    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock authentication manager."""
        mock_auth = Mock(spec=AuthManager)
        return mock_auth
    
    @pytest.fixture
    def gradio_interface(self, mock_digipal_core, mock_auth_manager):
        """Create GradioInterface instance."""
        return GradioInterface(mock_digipal_core, mock_auth_manager)
    
    def test_handle_quick_message(self, gradio_interface, mock_digipal_core):
        """Test quick message handling."""
        # Setup mocks
        mock_interaction = Mock()
        mock_interaction.pet_response = "Hello there!"
        mock_interaction.success = True
        
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.CHILD
        mock_pet_state.age_hours = 5.0
        mock_pet_state.status_summary = "Happy"
        mock_pet_state.hp = 85
        mock_pet_state.energy = 80
        mock_pet_state.happiness = 90
        mock_pet_state.weight = 30
        mock_pet_state.needs_attention = False
        
        mock_digipal_core.process_interaction.return_value = (True, mock_interaction)
        mock_digipal_core.get_pet_state.return_value = mock_pet_state
        mock_digipal_core.get_pet_statistics.return_value = {
            'interaction_summary': {'recent_interactions': []}
        }
        
        # Test quick message
        result = gradio_interface._handle_quick_message("Hello!", "test_user")
        
        # Verify results
        response, status, attributes, history, feedback = result
        
        assert "Quick Message:</strong> Hello!" in response
        assert "Hello there!" in response
        assert "TestPal" in status
        assert "Quick message sent!" in feedback
        mock_digipal_core.process_interaction.assert_called_once_with("test_user", "Hello!")
    
    def test_handle_audio_interaction(self, gradio_interface, mock_digipal_core):
        """Test audio interaction handling."""
        # Setup mocks
        mock_interaction = Mock()
        mock_interaction.user_input = "Hello DigiPal"
        mock_interaction.pet_response = "Hi there!"
        mock_interaction.success = True
        
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.TEEN
        mock_pet_state.age_hours = 25.0
        mock_pet_state.status_summary = "Energetic"
        mock_pet_state.hp = 90
        mock_pet_state.energy = 85
        mock_pet_state.happiness = 75
        mock_pet_state.weight = 35
        
        mock_digipal_core.process_audio_interaction.return_value = (True, mock_interaction)
        mock_digipal_core.get_pet_state.return_value = mock_pet_state
        mock_digipal_core.get_pet_statistics.return_value = {
            'interaction_summary': {'recent_interactions': []}
        }
        
        # Test audio interaction
        import numpy as np
        fake_audio = np.array([0.1, 0.2, 0.3])
        
        result = gradio_interface._handle_audio_interaction(fake_audio, "test_user")
        
        # Verify results
        audio_status, response, status, attributes, history = result
        
        assert "Speech processed successfully" in audio_status
        assert "Speech Input:</strong> Hello DigiPal" in response
        assert "Hi there!" in response
        assert "TestPal" in status
        mock_digipal_core.process_audio_interaction.assert_called_once_with("test_user", fake_audio)
    
    def test_handle_audio_interaction_no_audio(self, gradio_interface):
        """Test audio interaction with no audio data."""
        result = gradio_interface._handle_audio_interaction(None, "test_user")
        
        audio_status, response, status, attributes, history = result
        
        assert "Please record some audio first" in audio_status
    
    def test_handle_training_sub_actions(self, gradio_interface, mock_digipal_core):
        """Test training sub-action handling."""
        # Setup mocks
        mock_interaction = Mock()
        mock_interaction.pet_response = "Great strength training!"
        mock_interaction.success = True
        
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.name = "TestPal"
        mock_pet_state.life_stage = LifeStage.YOUNG_ADULT
        mock_pet_state.age_hours = 100.0
        mock_pet_state.status_summary = "Strong"
        mock_pet_state.hp = 95
        mock_pet_state.energy = 70
        mock_pet_state.happiness = 80
        mock_pet_state.weight = 40
        mock_pet_state.needs_attention = False
        
        mock_digipal_core.apply_care_action.return_value = (True, mock_interaction)
        mock_digipal_core.get_pet_state.return_value = mock_pet_state
        
        # Test strength training
        result = gradio_interface._handle_care_action("strength_train", "test_user")
        
        # Verify results
        response, status, attributes, feedback, needs = result
        
        assert "Care Action:</strong> 🏋️ Strength Train" in response
        assert "Great strength training!" in response
        assert "Strength Train completed!" in feedback
        mock_digipal_core.apply_care_action.assert_called_once_with("test_user", "strength_train")
    
    def test_handle_clear_history(self, gradio_interface, mock_digipal_core):
        """Test conversation history clearing."""
        # Setup mock
        mock_digipal_core.clear_conversation_history.return_value = True
        
        # Test clearing history
        result = gradio_interface._handle_clear_history("test_user")
        
        # Verify results
        history, response = result
        
        assert "Conversation history cleared" in history
        assert "Conversation history has been cleared!" in response
        mock_digipal_core.clear_conversation_history.assert_called_once_with("test_user")
    
    def test_handle_export_history(self, gradio_interface, mock_digipal_core):
        """Test conversation history export."""
        # Setup mock
        mock_interactions = [
            {
                'user_input': 'Hello',
                'pet_response': 'Hi there!',
                'timestamp': '2024-01-01 12:00:00'
            },
            {
                'user_input': 'How are you?',
                'pet_response': 'I am doing great!',
                'timestamp': '2024-01-01 12:01:00'
            }
        ]
        
        mock_stats = {
            'interaction_summary': {
                'recent_interactions': mock_interactions
            }
        }
        
        mock_digipal_core.get_pet_statistics.return_value = mock_stats
        
        # Test export
        result = gradio_interface._handle_export_history("test_user")
        
        # Verify result is a file path
        assert result is not None
        assert isinstance(result, str)
        assert result.endswith('.txt')
        
        # Clean up temporary file
        import os
        if os.path.exists(result):
            os.unlink(result)
    
    def test_format_needs_display(self, gradio_interface):
        """Test needs display formatting."""
        # Create mock pet state with various needs
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.energy = 15  # Very low
        mock_pet_state.happiness = 25  # Low
        mock_pet_state.weight = 10  # Too thin
        mock_pet_state.needs_attention = True
        
        # Test formatting
        needs_html = gradio_interface._format_needs_display(mock_pet_state)
        
        # Verify content
        assert "Needs Attention" in needs_html
        assert "Very tired" in needs_html
        assert "Unhappy" in needs_html
        assert "Too thin" in needs_html
        assert "Hasn't been interacted with recently" in needs_html
    
    def test_format_needs_display_all_good(self, gradio_interface):
        """Test needs display when all needs are met."""
        # Create mock pet state with good stats
        mock_pet_state = Mock(spec=PetState)
        mock_pet_state.energy = 80
        mock_pet_state.happiness = 85
        mock_pet_state.weight = 35
        mock_pet_state.needs_attention = False
        
        # Test formatting
        needs_html = gradio_interface._format_needs_display(mock_pet_state)
        
        # Verify content
        assert "All needs are being met!" in needs_html
        assert "all-good" in needs_html
    
    def test_toggle_auto_refresh(self, gradio_interface):
        """Test auto-refresh toggle functionality."""
        # Test enabling auto-refresh
        result_enabled = gradio_interface._toggle_auto_refresh(True, "test_user")
        assert "Auto-refresh enabled" in result_enabled
        assert "every 30 seconds" in result_enabled
        
        # Test disabling auto-refresh
        result_disabled = gradio_interface._toggle_auto_refresh(False, "test_user")
        assert "Auto-refresh disabled" in result_disabled
        assert "only on interaction" in result_disabled
    
    def test_enhanced_css_generation(self, gradio_interface):
        """Test enhanced CSS includes new styles."""
        css = gradio_interface._get_custom_css()
        
        # Check for enhanced styles
        enhanced_styles = [
            ".action-feedback",
            ".interaction-feedback",
            ".care-feedback",
            ".quick-feedback",
            ".needs-display",
            ".sub-care-btn",
            ".speech-panel",
            ".audio-status",
            ".processing",
            ".text-input-panel",
            ".quick-msg-btn",
            "@keyframes fadeInOut",
            "@keyframes pulse"
        ]
        
        for style in enhanced_styles:
            assert style in css
    
    def test_offline_mode_functionality(self, gradio_interface, mock_auth_manager, mock_digipal_core):
        """Test offline mode functionality."""
        # Setup mocks for offline mode
        mock_user = User(
            id="offline_user",
            username="offlineuser",
            created_at=datetime.now()
        )
        
        mock_auth_result = AuthResult(
            status=AuthStatus.OFFLINE_MODE,
            user=mock_user
        )
        
        mock_auth_manager.authenticate.return_value = mock_auth_result
        mock_digipal_core.load_existing_pet.return_value = None
        
        # Test offline login
        result = gradio_interface._handle_login("any_token", True, None, None)
        
        # Verify results
        auth_status, user_id, token, tabs = result
        
        assert "Welcome, offlineuser" in auth_status
        assert "Running in offline mode" in auth_status
        assert user_id == "offline_user"
        assert mock_auth_manager.offline_mode is True


if __name__ == "__main__":
    pytest.main([__file__])