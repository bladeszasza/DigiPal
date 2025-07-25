"""
Gradio web interface for DigiPal application.

This module provides the main web interface using Gradio with custom CSS
for a game-style UI. Includes authentication, egg selection, and main
pet interaction interfaces.
"""

import gradio as gr
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import base64
import io

from ..core.digipal_core import DigiPalCore, PetState
from ..core.enums import EggType, LifeStage
from ..auth.auth_manager import AuthManager
from ..auth.models import AuthStatus
from ..storage.database import DatabaseConnection

logger = logging.getLogger(__name__)


class GradioInterface:
    """Main Gradio interface for DigiPal application."""
    
    def __init__(self, digipal_core: DigiPalCore, auth_manager: AuthManager):
        """
        Initialize Gradio interface.
        
        Args:
            digipal_core: DigiPal core engine
            auth_manager: Authentication manager
        """
        self.digipal_core = digipal_core
        self.auth_manager = auth_manager
        
        # Current session state
        self.current_user_id: Optional[str] = None
        self.current_token: Optional[str] = None
        
        # Interface components
        self.app = None
        
        logger.info("GradioInterface initialized")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Custom CSS for game-style UI
        custom_css = self._get_custom_css()
        
        with gr.Blocks(
            css=custom_css,
            title="DigiPal - Your Digital Companion",
            theme=gr.themes.Soft()
        ) as interface:
            
            # State variables for session management
            user_state = gr.State(None)
            token_state = gr.State(None)
            
            # Main interface tabs with proper tab switching
            with gr.Tabs(selected="auth_tab") as main_tabs:
                
                # Authentication Tab
                with gr.Tab("Login", id="auth_tab") as auth_tab:
                    auth_components = self._create_authentication_tab()
                
                # Egg Selection Tab
                with gr.Tab("Choose Your Egg", id="egg_tab") as egg_tab:
                    egg_components = self._create_egg_selection_interface()
                
                # Main DigiPal Interface
                with gr.Tab("Your DigiPal", id="main_tab") as main_tab:
                    main_components = self._create_digipal_main_interface()
            
            # Event handlers
            self._setup_event_handlers(
                auth_components, egg_components, main_components,
                user_state, token_state, main_tabs
            )
        
        self.app = interface
        return interface
    
    def _create_authentication_tab(self) -> Dict[str, Any]:
        """Create the authentication tab components."""
        
        with gr.Column(elem_classes=["auth-container"]):
            gr.HTML("""
                <div class="auth-header">
                    <h1>🥚 Welcome to DigiPal</h1>
                    <p>Your digital companion awaits!</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Empty column for centering
                
                with gr.Column(scale=2, elem_classes=["auth-form"]):
                    gr.HTML("<h3>Login with HuggingFace</h3>")
                    
                    token_input = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="Enter your HuggingFace token...",
                        type="password",
                        elem_classes=["token-input"]
                    )
                    
                    login_btn = gr.Button(
                        "Login",
                        variant="primary",
                        elem_classes=["login-btn"]
                    )
                    
                    auth_status = gr.HTML(
                        "",
                        elem_classes=["auth-status"]
                    )
                    
                    # Offline mode toggle for development (outside accordion to avoid context issues)
                    offline_toggle = gr.Checkbox(
                        label="Enable Offline Mode (Development)",
                        value=False,
                        info="For development without HuggingFace token"
                    )
                
                with gr.Column(scale=1):
                    pass  # Empty column for centering
            
            # Instructions
            gr.HTML("""
                <details class="instructions-details">
                    <summary>How to get a HuggingFace Token</summary>
                    <div class="instructions">
                        <ol>
                            <li>Go to <a href="https://huggingface.co/settings/tokens" target="_blank">HuggingFace Tokens</a></li>
                            <li>Click "New token"</li>
                            <li>Give it a name like "DigiPal"</li>
                            <li>Select "Read" permissions</li>
                            <li>Copy the token and paste it above</li>
                        </ol>
                    </div>
                </details>
            """)
        
        return {
            'token_input': token_input,
            'login_btn': login_btn,
            'auth_status': auth_status,
            'offline_toggle': offline_toggle
        }
    
    def _create_egg_selection_interface(self) -> Dict[str, Any]:
        """Create the egg selection interface."""
        
        with gr.Column(elem_classes=["egg-selection-container"]):
            gr.HTML("""
                <div class="egg-header">
                    <h2>🥚 Choose Your DigiPal Egg</h2>
                    <p>Each egg type gives your DigiPal different starting attributes!</p>
                </div>
            """)
            
            with gr.Row(elem_classes=["egg-options"]):
                
                # Red Egg
                with gr.Column(elem_classes=["egg-option"]):
                    red_egg_btn = gr.Button(
                        "🔥 Red Egg",
                        variant="secondary",
                        elem_classes=["egg-btn", "red-egg"]
                    )
                    gr.HTML("""
                        <div class="egg-description">
                            <h4>Fire Type</h4>
                            <p>Higher Attack & Offense</p>
                            <p>Brave and energetic personality</p>
                        </div>
                    """)
                
                # Blue Egg
                with gr.Column(elem_classes=["egg-option"]):
                    blue_egg_btn = gr.Button(
                        "💧 Blue Egg",
                        variant="secondary",
                        elem_classes=["egg-btn", "blue-egg"]
                    )
                    gr.HTML("""
                        <div class="egg-description">
                            <h4>Water Type</h4>
                            <p>Higher Defense & HP</p>
                            <p>Calm and protective personality</p>
                        </div>
                    """)
                
                # Green Egg
                with gr.Column(elem_classes=["egg-option"]):
                    green_egg_btn = gr.Button(
                        "🌱 Green Egg",
                        variant="secondary",
                        elem_classes=["egg-btn", "green-egg"]
                    )
                    gr.HTML("""
                        <div class="egg-description">
                            <h4>Earth Type</h4>
                            <p>Higher Health & Symbiosis</p>
                            <p>Gentle and nurturing personality</p>
                        </div>
                    """)
            
            egg_status = gr.HTML(
                "",
                elem_classes=["egg-status"]
            )
        
        return {
            'red_egg_btn': red_egg_btn,
            'blue_egg_btn': blue_egg_btn,
            'green_egg_btn': green_egg_btn,
            'egg_status': egg_status
        }
    
    def _create_digipal_main_interface(self) -> Dict[str, Any]:
        """Create the main DigiPal interaction interface."""
        
        with gr.Row():
            # Left column - Pet display and status
            with gr.Column(scale=2, elem_classes=["pet-display-column"]):
                
                # Pet display area with visual feedback
                with gr.Group(elem_classes=["pet-display-area"]):
                    pet_image = gr.Image(
                        label="Your DigiPal",
                        show_label=False,
                        elem_classes=["pet-image"],
                        interactive=False
                    )
                    
                    pet_name_display = gr.HTML(
                        "<h3>Your DigiPal</h3>",
                        elem_classes=["pet-name"]
                    )
                    
                    # Visual feedback area for actions
                    action_feedback = gr.HTML(
                        "",
                        elem_classes=["action-feedback"]
                    )
                
                # Enhanced status display with real-time updates
                with gr.Group(elem_classes=["status-display"]):
                    gr.HTML("<h4>📊 Real-Time Status</h4>")
                    
                    # Basic status info
                    status_info = gr.HTML(
                        "",
                        elem_classes=["status-info"]
                    )
                    
                    # Detailed attribute bars with animations
                    attributes_display = gr.HTML(
                        "",
                        elem_classes=["attributes-display"]
                    )
                    
                    # Needs and alerts
                    needs_display = gr.HTML(
                        "",
                        elem_classes=["needs-display"]
                    )
                    
                    # Auto-refresh toggle
                    auto_refresh = gr.Checkbox(
                        label="Auto-refresh status (every 30s)",
                        value=True,
                        elem_classes=["auto-refresh-toggle"]
                    )
            
            # Right column - Enhanced interaction controls
            with gr.Column(scale=1, elem_classes=["interaction-column"]):
                
                # Enhanced care actions panel with detailed controls
                with gr.Group(elem_classes=["care-actions-panel"]):
                    gr.HTML("<h4>🎮 Care Actions</h4>")
                    
                    # Primary care actions
                    with gr.Row():
                        feed_btn = gr.Button("🍖 Feed", elem_classes=["care-btn", "feed-btn"])
                        train_btn = gr.Button("💪 Train", elem_classes=["care-btn", "train-btn"])
                    
                    # Training sub-options
                    with gr.Accordion("Training Options", open=False):
                        with gr.Row():
                            strength_train_btn = gr.Button("🏋️ Strength", elem_classes=["sub-care-btn"])
                            speed_train_btn = gr.Button("🏃 Speed", elem_classes=["sub-care-btn"])
                        with gr.Row():
                            brain_train_btn = gr.Button("🧠 Brains", elem_classes=["sub-care-btn"])
                            defense_train_btn = gr.Button("🛡️ Defense", elem_classes=["sub-care-btn"])
                    
                    # Emotional care
                    with gr.Row():
                        praise_btn = gr.Button("👍 Praise", elem_classes=["care-btn", "praise-btn"])
                        scold_btn = gr.Button("👎 Scold", elem_classes=["care-btn", "scold-btn"])
                    
                    # Rest and play
                    with gr.Row():
                        rest_btn = gr.Button("😴 Rest", elem_classes=["care-btn", "rest-btn"])
                        play_btn = gr.Button("🎾 Play", elem_classes=["care-btn", "play-btn"])
                    
                    # Advanced care options
                    with gr.Accordion("Advanced Care", open=False):
                        with gr.Row():
                            medicine_btn = gr.Button("💊 Medicine", elem_classes=["sub-care-btn"])
                            clean_btn = gr.Button("🧼 Clean", elem_classes=["sub-care-btn"])
                
                # Enhanced speech interaction interface
                with gr.Group(elem_classes=["speech-panel"]):
                    gr.HTML("<h4>🎤 Voice Interaction</h4>")
                    
                    # Audio recording with enhanced controls
                    audio_input = gr.Audio(
                        label="Record your voice (click to start/stop)",
                        type="numpy",
                        elem_classes=["audio-input"]
                    )
                    
                    # Audio processing status
                    audio_status = gr.HTML(
                        "",
                        elem_classes=["audio-status"]
                    )
                    
                    # Process audio button
                    process_audio_btn = gr.Button(
                        "🎵 Process Speech",
                        variant="secondary",
                        elem_classes=["process-audio-btn"]
                    )
                
                # Text input as alternative with enhanced features
                with gr.Group(elem_classes=["text-input-panel"]):
                    gr.HTML("<h4>💬 Text Chat</h4>")
                    
                    text_input = gr.Textbox(
                        label="Type a message",
                        placeholder="Say something to your DigiPal...",
                        elem_classes=["text-input"],
                        lines=2
                    )
                    
                    # Quick message buttons
                    with gr.Row():
                        quick_hello_btn = gr.Button("👋 Hello", elem_classes=["quick-msg-btn"])
                        quick_status_btn = gr.Button("❓ How are you?", elem_classes=["quick-msg-btn"])
                    
                    send_btn = gr.Button(
                        "Send Message",
                        variant="primary",
                        elem_classes=["send-btn"]
                    )
                
                # Enhanced response display with conversation history
                with gr.Group(elem_classes=["response-panel"]):
                    gr.HTML("<h4>💬 DigiPal Response</h4>")
                    
                    response_display = gr.HTML(
                        "<p>Your DigiPal is waiting for you to say something!</p>",
                        elem_classes=["response-display"]
                    )
                    
                    # Live conversation history (always visible)
                    conversation_history = gr.HTML(
                        "",
                        elem_classes=["conversation-history"]
                    )
                    
                    # Conversation controls
                    with gr.Row():
                        clear_history_btn = gr.Button("🗑️ Clear History", elem_classes=["clear-btn"])
                        export_history_btn = gr.Button("📥 Export Chat", elem_classes=["export-btn"])
        
        return {
            'pet_image': pet_image,
            'pet_name_display': pet_name_display,
            'action_feedback': action_feedback,
            'status_info': status_info,
            'attributes_display': attributes_display,
            'needs_display': needs_display,
            'auto_refresh': auto_refresh,
            'feed_btn': feed_btn,
            'train_btn': train_btn,
            'strength_train_btn': strength_train_btn,
            'speed_train_btn': speed_train_btn,
            'brain_train_btn': brain_train_btn,
            'defense_train_btn': defense_train_btn,
            'praise_btn': praise_btn,
            'scold_btn': scold_btn,
            'rest_btn': rest_btn,
            'play_btn': play_btn,
            'medicine_btn': medicine_btn,
            'clean_btn': clean_btn,
            'audio_input': audio_input,
            'audio_status': audio_status,
            'process_audio_btn': process_audio_btn,
            'text_input': text_input,
            'quick_hello_btn': quick_hello_btn,
            'quick_status_btn': quick_status_btn,
            'send_btn': send_btn,
            'response_display': response_display,
            'conversation_history': conversation_history,
            'clear_history_btn': clear_history_btn,
            'export_history_btn': export_history_btn
        }
    
    def _setup_event_handlers(self, auth_components: Dict, egg_components: Dict, 
                            main_components: Dict, user_state: gr.State, 
                            token_state: gr.State, main_tabs: gr.Tabs):
        """Set up event handlers for all interface components."""
        
        # Authentication handlers
        auth_components['login_btn'].click(
            fn=self._handle_login,
            inputs=[
                auth_components['token_input'],
                auth_components['offline_toggle'],
                user_state,
                token_state
            ],
            outputs=[
                auth_components['auth_status'],
                user_state,
                token_state,
                main_tabs
            ]
        )
        
        # Egg selection handlers
        for egg_type, btn in [
            (EggType.RED, egg_components['red_egg_btn']),
            (EggType.BLUE, egg_components['blue_egg_btn']),
            (EggType.GREEN, egg_components['green_egg_btn'])
        ]:
            btn.click(
                fn=lambda user_state_val, egg=egg_type: self._handle_egg_selection(egg, user_state_val),
                inputs=[user_state],
                outputs=[
                    egg_components['egg_status'],
                    main_tabs
                ]
            )
        
        # Text interaction handlers
        main_components['send_btn'].click(
            fn=self._handle_text_interaction,
            inputs=[
                main_components['text_input'],
                user_state
            ],
            outputs=[
                main_components['response_display'],
                main_components['text_input'],
                main_components['status_info'],
                main_components['attributes_display'],
                main_components['conversation_history'],
                main_components['action_feedback'],
                main_components['needs_display']
            ]
        )
        
        # Quick message handlers
        main_components['quick_hello_btn'].click(
            fn=self._handle_quick_message,
            inputs=[gr.State("Hello!"), user_state],
            outputs=[
                main_components['response_display'],
                main_components['status_info'],
                main_components['attributes_display'],
                main_components['conversation_history'],
                main_components['action_feedback']
            ]
        )
        
        main_components['quick_status_btn'].click(
            fn=self._handle_quick_message,
            inputs=[gr.State("How are you?"), user_state],
            outputs=[
                main_components['response_display'],
                main_components['status_info'],
                main_components['attributes_display'],
                main_components['conversation_history'],
                main_components['action_feedback']
            ]
        )
        
        # Audio processing handler
        main_components['process_audio_btn'].click(
            fn=self._handle_audio_interaction,
            inputs=[
                main_components['audio_input'],
                user_state
            ],
            outputs=[
                main_components['audio_status'],
                main_components['response_display'],
                main_components['status_info'],
                main_components['attributes_display'],
                main_components['conversation_history']
            ]
        )
        
        # Primary care action handlers
        primary_care_actions = {
            'feed': main_components['feed_btn'],
            'train': main_components['train_btn'],
            'praise': main_components['praise_btn'],
            'scold': main_components['scold_btn'],
            'rest': main_components['rest_btn'],
            'play': main_components['play_btn']
        }
        
        for action_name, btn in primary_care_actions.items():
            btn.click(
                fn=lambda user_state_val, action=action_name: self._handle_care_action(action, user_state_val),
                inputs=[user_state],
                outputs=[
                    main_components['response_display'],
                    main_components['status_info'],
                    main_components['attributes_display'],
                    main_components['action_feedback'],
                    main_components['needs_display']
                ]
            )
        
        # Training sub-action handlers
        training_actions = {
            'strength_train': main_components['strength_train_btn'],
            'speed_train': main_components['speed_train_btn'],
            'brain_train': main_components['brain_train_btn'],
            'defense_train': main_components['defense_train_btn']
        }
        
        for action_name, btn in training_actions.items():
            btn.click(
                fn=lambda user_state_val, action=action_name: self._handle_care_action(action, user_state_val),
                inputs=[user_state],
                outputs=[
                    main_components['response_display'],
                    main_components['status_info'],
                    main_components['attributes_display'],
                    main_components['action_feedback']
                ]
            )
        
        # Advanced care action handlers
        advanced_care_actions = {
            'medicine': main_components['medicine_btn'],
            'clean': main_components['clean_btn']
        }
        
        for action_name, btn in advanced_care_actions.items():
            btn.click(
                fn=lambda user_state_val, action=action_name: self._handle_care_action(action, user_state_val),
                inputs=[user_state],
                outputs=[
                    main_components['response_display'],
                    main_components['status_info'],
                    main_components['attributes_display'],
                    main_components['action_feedback']
                ]
            )
        
        # Conversation management handlers
        main_components['clear_history_btn'].click(
            fn=self._handle_clear_history,
            inputs=[user_state],
            outputs=[
                main_components['conversation_history'],
                main_components['response_display']
            ]
        )
        
        main_components['export_history_btn'].click(
            fn=self._handle_export_history,
            inputs=[user_state],
            outputs=[gr.File()]
        )
        
        # Auto-refresh handler (periodic update)
        main_components['auto_refresh'].change(
            fn=self._toggle_auto_refresh,
            inputs=[main_components['auto_refresh'], user_state],
            outputs=[main_components['status_info']]
        )
    
    def _handle_login(self, token: str, offline_mode: bool, 
                     current_user: Optional[str], current_token: Optional[str]) -> Tuple:
        """Handle user login."""
        if not token:
            return (
                '<div class="error">Please enter a token</div>',
                current_user,
                current_token,
                gr.update(selected="auth_tab")  # Stay on auth tab
            )
        
        # Set offline mode if requested
        if offline_mode:
            self.auth_manager.offline_mode = True
        
        # Authenticate user
        auth_result = self.auth_manager.authenticate(token)
        
        if auth_result.status in [AuthStatus.SUCCESS, AuthStatus.OFFLINE_MODE]:
            self.current_user_id = auth_result.user.id
            self.current_token = token
            
            # Check if user has existing DigiPal
            existing_pet = self.digipal_core.load_existing_pet(auth_result.user.id)
            
            if existing_pet:
                # User has existing pet, go to main interface
                status_msg = f'<div class="success">Welcome back, {auth_result.user.username}!</div>'
                if auth_result.status == AuthStatus.OFFLINE_MODE:
                    status_msg += '<div class="info">Running in offline mode</div>'
                
                return (
                    status_msg,
                    auth_result.user.id,
                    token,
                    gr.update(selected="main_tab")  # Go to main tab
                )
            else:
                # New user, go to egg selection
                status_msg = f'<div class="success">Welcome, {auth_result.user.username}! Choose your first egg.</div>'
                if auth_result.status == AuthStatus.OFFLINE_MODE:
                    status_msg += '<div class="info">Running in offline mode</div>'
                
                return (
                    status_msg,
                    auth_result.user.id,
                    token,
                    gr.update(selected="egg_tab")  # Go to egg tab
                )
        else:
            error_msg = f'<div class="error">Login failed: {auth_result.error_message}</div>'
            return (
                error_msg,
                current_user,
                current_token,
                gr.update(selected="auth_tab")  # Stay on auth tab
            )
    
    def _handle_egg_selection(self, egg_type: EggType, user_id: Optional[str]) -> Tuple:
        """Handle egg selection."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                gr.update(selected="auth_tab")  # Go back to auth tab
            )
        
        try:
            # Create new DigiPal with selected egg type
            pet = self.digipal_core.create_new_pet(egg_type, user_id)
            
            success_msg = f'''
                <div class="success">
                    <h4>🎉 Egg Selected!</h4>
                    <p>You chose the {egg_type.value} egg. Your DigiPal journey begins!</p>
                    <p>Go to the main interface to start caring for your digital companion.</p>
                </div>
            '''
            
            return (
                success_msg,
                gr.update(selected="main_tab")  # Go to main tab
            )
            
        except Exception as e:
            logger.error(f"Error creating new pet: {e}")
            return (
                f'<div class="error">Failed to create DigiPal: {str(e)}</div>',
                gr.update(selected="egg_tab")  # Stay on egg tab
            )
    
    def _handle_text_interaction(self, text: str, user_id: Optional[str]) -> Tuple:
        """Handle text interaction with DigiPal."""
        if not user_id or not text.strip():
            return (
                '<div class="error">Please enter a message</div>',
                text,
                "",
                "",
                "",
                "",
                ""
            )
        
        try:
            # Process interaction
            success, interaction = self.digipal_core.process_interaction(user_id, text.strip())
            
            if success:
                response_html = f'''
                    <div class="pet-response">
                        <p><strong>DigiPal:</strong> {interaction.pet_response}</p>
                    </div>
                '''
                
                # Get updated pet state
                pet_state = self.digipal_core.get_pet_state(user_id)
                status_html, attributes_html = self._format_pet_status(pet_state)
                needs_html = self._format_needs_display(pet_state)
                
                # Update conversation history
                history_html = self._format_conversation_history(user_id)
                
                # Visual feedback for interaction
                feedback_html = f'''
                    <div class="interaction-feedback">
                        <span class="feedback-icon">💬</span>
                        <span class="feedback-text">Message sent!</span>
                    </div>
                '''
                
                return (
                    response_html,
                    "",  # Clear text input
                    status_html,
                    attributes_html,
                    history_html,
                    feedback_html,
                    needs_html
                )
            else:
                return (
                    f'<div class="error">Interaction failed: {interaction.pet_response}</div>',
                    text,
                    "",
                    "",
                    "",
                    "",
                    ""
                )
                
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return (
                f'<div class="error">Error: {str(e)}</div>',
                text,
                "",
                "",
                "",
                "",
                ""
            )
    
    def _handle_care_action(self, action: str, user_id: Optional[str]) -> Tuple:
        """Handle care action with enhanced feedback."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                "",
                "",
                "",
                ""
            )
        
        try:
            # Apply care action
            success, interaction = self.digipal_core.apply_care_action(user_id, action)
            
            if success:
                # Action-specific icons and colors
                action_icons = {
                    'feed': '🍖',
                    'train': '💪',
                    'strength_train': '🏋️',
                    'speed_train': '🏃',
                    'brain_train': '🧠',
                    'defense_train': '🛡️',
                    'praise': '👍',
                    'scold': '👎',
                    'rest': '😴',
                    'play': '🎾',
                    'medicine': '💊',
                    'clean': '🧼'
                }
                
                icon = action_icons.get(action, '🎮')
                
                response_html = f'''
                    <div class="care-response">
                        <p><strong>Care Action:</strong> {icon} {action.replace('_', ' ').title()}</p>
                        <p><strong>DigiPal:</strong> {interaction.pet_response}</p>
                    </div>
                '''
                
                # Get updated pet state
                pet_state = self.digipal_core.get_pet_state(user_id)
                status_html, attributes_html = self._format_pet_status(pet_state)
                needs_html = self._format_needs_display(pet_state)
                
                # Visual feedback for care action
                feedback_html = f'''
                    <div class="care-feedback">
                        <span class="feedback-icon">{icon}</span>
                        <span class="feedback-text">{action.replace('_', ' ').title()} completed!</span>
                    </div>
                '''
                
                return (
                    response_html,
                    status_html,
                    attributes_html,
                    feedback_html,
                    needs_html
                )
            else:
                return (
                    f'<div class="error">Care action failed: {interaction.pet_response}</div>',
                    "",
                    "",
                    "",
                    ""
                )
                
        except Exception as e:
            logger.error(f"Error applying care action: {e}")
            return (
                f'<div class="error">Error: {str(e)}</div>',
                "",
                "",
                "",
                ""
            )
    
    def _format_pet_status(self, pet_state: Optional[PetState]) -> Tuple[str, str]:
        """Format pet status for display."""
        if not pet_state:
            return "", ""
        
        # Status info
        status_html = f'''
            <div class="status-grid">
                <div class="status-item">
                    <span class="label">Name:</span>
                    <span class="value">{pet_state.name}</span>
                </div>
                <div class="status-item">
                    <span class="label">Stage:</span>
                    <span class="value">{pet_state.life_stage.value.title()}</span>
                </div>
                <div class="status-item">
                    <span class="label">Age:</span>
                    <span class="value">{pet_state.age_hours:.1f} hours</span>
                </div>
                <div class="status-item">
                    <span class="label">Status:</span>
                    <span class="value">{pet_state.status_summary}</span>
                </div>
            </div>
        '''
        
        # Attributes display with progress bars
        attributes_html = f'''
            <div class="attributes-grid">
                <div class="attribute-bar">
                    <span class="attr-label">HP</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(pet_state.hp, 100)}%"></div>
                    </div>
                    <span class="attr-value">{pet_state.hp}</span>
                </div>
                <div class="attribute-bar">
                    <span class="attr-label">Energy</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {pet_state.energy}%"></div>
                    </div>
                    <span class="attr-value">{pet_state.energy}</span>
                </div>
                <div class="attribute-bar">
                    <span class="attr-label">Happiness</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {pet_state.happiness}%"></div>
                    </div>
                    <span class="attr-value">{pet_state.happiness}</span>
                </div>
                <div class="attribute-bar">
                    <span class="attr-label">Weight</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(pet_state.weight, 100)}%"></div>
                    </div>
                    <span class="attr-value">{pet_state.weight}</span>
                </div>
            </div>
        '''
        
        return status_html, attributes_html
    
    def _format_conversation_history(self, user_id: str) -> str:
        """Format conversation history for display."""
        try:
            pet_stats = self.digipal_core.get_pet_statistics(user_id)
            interactions = pet_stats.get('interaction_summary', {}).get('recent_interactions', [])
            
            if not interactions:
                return "<p>No conversation history yet.</p>"
            
            history_html = "<div class='history-list'>"
            for interaction in interactions[-5:]:  # Show last 5 interactions
                history_html += f'''
                    <div class="history-item">
                        <div class="user-msg"><strong>You:</strong> {interaction.get('user_input', '')}</div>
                        <div class="pet-msg"><strong>DigiPal:</strong> {interaction.get('pet_response', '')}</div>
                    </div>
                '''
            history_html += "</div>"
            
            return history_html
            
        except Exception as e:
            logger.error(f"Error formatting conversation history: {e}")
            return "<p>Error loading conversation history.</p>"
    
    def _format_needs_display(self, pet_state: Optional[PetState]) -> str:
        """Format pet needs and alerts for display."""
        if not pet_state:
            return ""
        
        needs_html = "<div class='needs-alerts'>"
        
        # Check for urgent needs
        urgent_needs = []
        if pet_state.energy < 20:
            urgent_needs.append("🔴 Very tired - needs rest!")
        elif pet_state.energy < 40:
            urgent_needs.append("🟡 Getting tired")
            
        if pet_state.happiness < 30:
            urgent_needs.append("🔴 Unhappy - needs attention!")
        elif pet_state.happiness < 50:
            urgent_needs.append("🟡 Could be happier")
            
        if pet_state.weight < 15:
            urgent_needs.append("🔴 Too thin - needs food!")
        elif pet_state.weight > 75:
            urgent_needs.append("🔴 Overweight - needs exercise!")
            
        if pet_state.needs_attention:
            urgent_needs.append("⚠️ Hasn't been interacted with recently")
        
        if urgent_needs:
            needs_html += "<h5>🚨 Needs Attention:</h5><ul>"
            for need in urgent_needs:
                needs_html += f"<li>{need}</li>"
            needs_html += "</ul>"
        else:
            needs_html += "<p class='all-good'>✅ All needs are being met!</p>"
        
        needs_html += "</div>"
        return needs_html
    
    def _handle_quick_message(self, message: str, user_id: Optional[str]) -> Tuple:
        """Handle quick message buttons."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                "",
                "",
                "",
                ""
            )
        
        try:
            # Process the quick message
            success, interaction = self.digipal_core.process_interaction(user_id, message)
            
            if success:
                response_html = f'''
                    <div class="pet-response quick-response">
                        <p><strong>Quick Message:</strong> {message}</p>
                        <p><strong>DigiPal:</strong> {interaction.pet_response}</p>
                    </div>
                '''
                
                # Get updated pet state
                pet_state = self.digipal_core.get_pet_state(user_id)
                status_html, attributes_html = self._format_pet_status(pet_state)
                
                # Update conversation history
                history_html = self._format_conversation_history(user_id)
                
                # Visual feedback
                feedback_html = f'''
                    <div class="quick-feedback">
                        <span class="feedback-icon">⚡</span>
                        <span class="feedback-text">Quick message sent!</span>
                    </div>
                '''
                
                return (
                    response_html,
                    status_html,
                    attributes_html,
                    history_html,
                    feedback_html
                )
            else:
                return (
                    f'<div class="error">Quick message failed: {interaction.pet_response}</div>',
                    "",
                    "",
                    "",
                    ""
                )
                
        except Exception as e:
            logger.error(f"Error processing quick message: {e}")
            return (
                f'<div class="error">Error: {str(e)}</div>',
                "",
                "",
                "",
                ""
            )
    
    def _handle_audio_interaction(self, audio_data, user_id: Optional[str]) -> Tuple:
        """Handle audio input and speech processing."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                "",
                "",
                "",
                ""
            )
        
        if audio_data is None:
            return (
                '<div class="info">Please record some audio first</div>',
                "",
                "",
                "",
                ""
            )
        
        try:
            # Update audio status
            status_html = '<div class="processing">🎵 Processing speech...</div>'
            
            # Process audio through speech recognition
            # Note: This will use the AI communication layer's speech processor
            success, interaction = self.digipal_core.process_audio_interaction(user_id, audio_data)
            
            if success:
                response_html = f'''
                    <div class="pet-response audio-response">
                        <p><strong>Speech Input:</strong> {interaction.user_input}</p>
                        <p><strong>DigiPal:</strong> {interaction.pet_response}</p>
                    </div>
                '''
                
                # Get updated pet state
                pet_state = self.digipal_core.get_pet_state(user_id)
                status_html, attributes_html = self._format_pet_status(pet_state)
                
                # Update conversation history
                history_html = self._format_conversation_history(user_id)
                
                audio_status_final = '<div class="success">✅ Speech processed successfully!</div>'
                
                return (
                    audio_status_final,
                    response_html,
                    status_html,
                    attributes_html,
                    history_html
                )
            else:
                return (
                    f'<div class="error">Speech processing failed: {interaction.pet_response}</div>',
                    "",
                    "",
                    "",
                    ""
                )
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return (
                f'<div class="error">Audio processing error: {str(e)}</div>',
                "",
                "",
                "",
                ""
            )
    
    def _handle_clear_history(self, user_id: Optional[str]) -> Tuple:
        """Handle clearing conversation history."""
        if not user_id:
            return (
                "<p>Please login first.</p>",
                '<div class="error">Please login first</div>'
            )
        
        try:
            # Clear conversation history in the core
            success = self.digipal_core.clear_conversation_history(user_id)
            
            if success:
                return (
                    "<p>Conversation history cleared.</p>",
                    '<div class="success">Conversation history has been cleared!</div>'
                )
            else:
                return (
                    "<p>Error clearing conversation history.</p>",
                    '<div class="error">Failed to clear conversation history</div>'
                )
                
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return (
                "<p>Error clearing conversation history.</p>",
                f'<div class="error">Error: {str(e)}</div>'
            )
    
    def _handle_export_history(self, user_id: Optional[str]):
        """Handle exporting conversation history."""
        if not user_id:
            return None
        
        try:
            # Get conversation history
            pet_stats = self.digipal_core.get_pet_statistics(user_id)
            interactions = pet_stats.get('interaction_summary', {}).get('recent_interactions', [])
            
            if not interactions:
                return None
            
            # Create export content
            export_content = "DigiPal Conversation History\n"
            export_content += "=" * 30 + "\n\n"
            
            for i, interaction in enumerate(interactions, 1):
                export_content += f"Interaction {i}:\n"
                export_content += f"You: {interaction.get('user_input', '')}\n"
                export_content += f"DigiPal: {interaction.get('pet_response', '')}\n"
                export_content += f"Timestamp: {interaction.get('timestamp', '')}\n\n"
            
            # Save to temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(export_content)
                temp_path = f.name
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return None
    
    def _toggle_auto_refresh(self, auto_refresh_enabled: bool, user_id: Optional[str]) -> str:
        """Handle auto-refresh toggle."""
        if not user_id:
            return ""
        
        if auto_refresh_enabled:
            status_msg = '<div class="info">Auto-refresh enabled - status will update every 30 seconds</div>'
        else:
            status_msg = '<div class="info">Auto-refresh disabled - status updates only on interaction</div>'
        
        return status_msg
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for game-style UI."""
        return """
        /* Game-style UI CSS */
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Authentication styles */
        .auth-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        
        .auth-header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .auth-form {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .token-input input {
            border-radius: 10px;
            border: 2px solid #ddd;
            padding: 12px;
            font-size: 16px;
        }
        
        .login-btn {
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border: none;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        /* Egg selection styles */
        .egg-selection-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .egg-header {
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        
        .egg-options {
            display: flex;
            gap: 2rem;
            justify-content: center;
        }
        
        .egg-option {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .egg-option:hover {
            transform: translateY(-5px);
        }
        
        .egg-btn {
            font-size: 2rem;
            padding: 1rem 2rem;
            border-radius: 15px;
            border: 3px solid;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .red-egg {
            border-color: #ff6b6b;
            background: linear-gradient(45deg, #ff6b6b, #ff5252);
        }
        
        .blue-egg {
            border-color: #4ecdc4;
            background: linear-gradient(45deg, #4ecdc4, #26a69a);
        }
        
        .green-egg {
            border-color: #95e1d3;
            background: linear-gradient(45deg, #95e1d3, #66bb6a);
        }
        
        .egg-description h4 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .egg-description p {
            color: #666;
            margin: 0.25rem 0;
        }
        
        /* Main interface styles */
        .pet-display-column {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            margin-right: 1rem;
        }
        
        .interaction-column {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
        }
        
        .pet-display-area {
            text-align: center;
            margin-bottom: 1.5rem;
            padding: 1rem;
            border: 2px dashed #ddd;
            border-radius: 10px;
        }
        
        .pet-image {
            max-width: 300px;
            max-height: 300px;
            border-radius: 10px;
        }
        
        .pet-name {
            color: #333;
            margin-top: 1rem;
        }
        
        .status-display {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
        }
        
        .label {
            font-weight: bold;
            color: #555;
        }
        
        .value {
            color: #333;
        }
        
        .attributes-display {
            margin-top: 1rem;
        }
        
        .attributes-grid {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .attribute-bar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .attr-label {
            min-width: 80px;
            font-weight: bold;
            color: #555;
        }
        
        .progress-bar {
            flex: 1;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }
        
        .attr-value {
            min-width: 40px;
            text-align: right;
            font-weight: bold;
            color: #333;
        }
        
        .care-actions-panel {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .care-btn {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 14px;
            margin: 0.25rem;
            transition: all 0.2s ease;
        }
        
        .care-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .speech-panel {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .response-panel {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
        }
        
        .pet-response {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        
        .care-response {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        
        .conversation-history {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .history-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .history-item {
            background: white;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #eee;
        }
        
        .user-msg {
            color: #1976d2;
            margin-bottom: 0.25rem;
        }
        
        .pet-msg {
            color: #388e3c;
        }
        
        /* Enhanced feedback and visual effects */
        .action-feedback {
            text-align: center;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            min-height: 30px;
        }
        
        .interaction-feedback, .care-feedback, .quick-feedback {
            background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
            border: 1px solid #4caf50;
            animation: fadeInOut 3s ease-in-out;
        }
        
        .care-feedback {
            background: linear-gradient(45deg, #fff3e0, #ffcc02);
            border: 1px solid #ff9800;
        }
        
        .quick-feedback {
            background: linear-gradient(45deg, #e3f2fd, #bbdefb);
            border: 1px solid #2196f3;
        }
        
        @keyframes fadeInOut {
            0% { opacity: 0; transform: translateY(-10px); }
            20% { opacity: 1; transform: translateY(0); }
            80% { opacity: 1; transform: translateY(0); }
            100% { opacity: 0; transform: translateY(-10px); }
        }
        
        .feedback-icon {
            font-size: 1.2em;
            margin-right: 0.5rem;
        }
        
        .feedback-text {
            font-weight: bold;
        }
        
        /* Enhanced needs display */
        .needs-display {
            background: #fff8e1;
            border: 1px solid #ffcc02;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .needs-alerts h5 {
            color: #e65100;
            margin: 0 0 0.5rem 0;
        }
        
        .needs-alerts ul {
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .needs-alerts li {
            margin-bottom: 0.25rem;
            color: #bf360c;
        }
        
        .all-good {
            color: #2e7d32;
            font-weight: bold;
            text-align: center;
            margin: 0;
        }
        
        /* Enhanced care action buttons */
        .sub-care-btn {
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            font-size: 12px;
            margin: 0.2rem;
            transition: all 0.2s ease;
            background: linear-gradient(45deg, #f5f5f5, #e0e0e0);
        }
        
        .sub-care-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
        }
        
        .feed-btn {
            background: linear-gradient(45deg, #ffeb3b, #ffc107);
        }
        
        .train-btn {
            background: linear-gradient(45deg, #ff9800, #f57c00);
        }
        
        .praise-btn {
            background: linear-gradient(45deg, #4caf50, #388e3c);
        }
        
        .scold-btn {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .rest-btn {
            background: linear-gradient(45deg, #9c27b0, #7b1fa2);
        }
        
        .play-btn {
            background: linear-gradient(45deg, #2196f3, #1976d2);
        }
        
        /* Enhanced speech panel */
        .speech-panel {
            background: #f3e5f5;
            border: 2px solid #9c27b0;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .audio-input {
            border: 2px dashed #9c27b0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        
        .audio-status {
            text-align: center;
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            min-height: 25px;
        }
        
        .processing {
            background: #fff3e0;
            color: #e65100;
            border: 1px solid #ffcc02;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .process-audio-btn {
            background: linear-gradient(45deg, #9c27b0, #7b1fa2);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        
        /* Enhanced text input panel */
        .text-input-panel {
            background: #e8f5e8;
            border: 2px solid #4caf50;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .quick-msg-btn {
            background: linear-gradient(45deg, #e3f2fd, #bbdefb);
            border: 1px solid #2196f3;
            border-radius: 15px;
            padding: 0.3rem 0.8rem;
            font-size: 12px;
            margin: 0.2rem;
            transition: all 0.2s ease;
        }
        
        .quick-msg-btn:hover {
            background: linear-gradient(45deg, #2196f3, #1976d2);
            color: white;
            transform: scale(1.05);
        }
        
        /* Enhanced response display */
        .pet-response.audio-response {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .pet-response.quick-response {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        /* Conversation controls */
        .clear-btn, .export-btn {
            background: linear-gradient(45deg, #757575, #616161);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            font-size: 12px;
            margin: 0.2rem;
        }
        
        .clear-btn:hover {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .export-btn:hover {
            background: linear-gradient(45deg, #4caf50, #388e3c);
        }
        
        /* Auto-refresh toggle */
        .auto-refresh-toggle {
            margin-top: 1rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        /* Status messages */
        .success {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #c3e6cb;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
        }
        
        .info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #bee5eb;
            margin-top: 0.5rem;
        }
        
        .instructions {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        
        .instructions ol {
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .instructions li {
            margin-bottom: 0.5rem;
        }
        
        .instructions a {
            color: #1976d2;
            text-decoration: none;
        }
        
        .instructions a:hover {
            text-decoration: underline;
        }
        
        /* Responsive design improvements */
        @media (max-width: 768px) {
            .pet-display-column, .interaction-column {
                margin: 0.5rem 0;
            }
            
            .care-btn, .sub-care-btn {
                font-size: 12px;
                padding: 0.3rem 0.6rem;
            }
            
            .egg-options {
                flex-direction: column;
                gap: 1rem;
            }
            
            .auth-form {
                padding: 1rem;
            }
        }
        """
    
    def launch_interface(self, share: bool = False, server_name: str = "127.0.0.1", 
                        server_port: int = 7860, debug: bool = False,
                        ssl_keyfile: Optional[str] = None, ssl_certfile: Optional[str] = None) -> None:
        """
        Launch the Gradio interface with health check endpoints.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
            ssl_keyfile: SSL key file path
            ssl_certfile: SSL certificate file path
        """
        if not self.app:
            self.app = self.create_interface()
        
        # Add health check endpoints
        self._add_health_endpoints()
        
        logger.info(f"Launching DigiPal interface on {server_name}:{server_port}")
        
        # Launch configuration
        launch_kwargs = {
            'share': share,
            'server_name': server_name,
            'server_port': server_port,
            'debug': debug,
            'show_error': True
        }
        
        # Add SSL configuration if provided
        if ssl_keyfile and ssl_certfile:
            launch_kwargs.update({
                'ssl_keyfile': ssl_keyfile,
                'ssl_certfile': ssl_certfile
            })
        
        self.app.launch(**launch_kwargs)
    
    def _add_health_endpoints(self):
        """Add health check and metrics endpoints to the Gradio app."""
        try:
            from ..monitoring import health_checker, get_metrics
            import json
            import time
            
            # Health check endpoint
            def health_check_handler():
                """Handle health check requests."""
                try:
                    health_status = health_checker.get_health_status()
                    return {
                        "status": health_status["status"],
                        "timestamp": health_status["timestamp"],
                        "checks": health_status["checks"],
                        "version": "0.1.0",
                        "environment": getattr(self, 'env', 'unknown')
                    }
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    return {
                        "status": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    }
            
            # Metrics endpoint
            def metrics_handler():
                """Handle metrics requests."""
                try:
                    return get_metrics()
                except Exception as e:
                    logger.error(f"Metrics endpoint failed: {e}")
                    return f"# Error: {e}"
            
            # Add custom routes if Gradio supports it
            # Note: This is a simplified approach - in production you might want
            # to use a proper web framework like FastAPI alongside Gradio
            if hasattr(self.app, 'add_route'):
                self.app.add_route('/health', health_check_handler, methods=['GET'])
                self.app.add_route('/metrics', metrics_handler, methods=['GET'])
            else:
                # Fallback: log that endpoints couldn't be added
                logger.warning("Could not add health endpoints - Gradio version may not support custom routes")
                
        except ImportError:
            logger.warning("Monitoring module not available - health endpoints disabled")
        except Exception as e:
            logger.error(f"Failed to add health endpoints: {e}")