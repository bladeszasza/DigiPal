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
            
            # Main interface tabs
            with gr.Tabs() as main_tabs:
                
                # Authentication Tab
                with gr.Tab("Login", id="auth_tab") as auth_tab:
                    auth_components = self._create_authentication_tab()
                
                # Egg Selection Tab (hidden initially)
                with gr.Tab("Choose Your Egg", id="egg_tab", visible=False) as egg_tab:
                    egg_components = self._create_egg_selection_interface()
                
                # Main DigiPal Interface (hidden initially)
                with gr.Tab("Your DigiPal", id="main_tab", visible=False) as main_tab:
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
                
                # Pet display area
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
                
                # Status display
                with gr.Group(elem_classes=["status-display"]):
                    gr.HTML("<h4>📊 Status</h4>")
                    
                    status_info = gr.HTML(
                        "",
                        elem_classes=["status-info"]
                    )
                    
                    # Attribute bars (will be updated with JavaScript)
                    attributes_display = gr.HTML(
                        "",
                        elem_classes=["attributes-display"]
                    )
            
            # Right column - Interaction controls
            with gr.Column(scale=1, elem_classes=["interaction-column"]):
                
                # Care actions panel
                with gr.Group(elem_classes=["care-actions-panel"]):
                    gr.HTML("<h4>🎮 Care Actions</h4>")
                    
                    with gr.Row():
                        feed_btn = gr.Button("🍖 Feed", elem_classes=["care-btn"])
                        train_btn = gr.Button("💪 Train", elem_classes=["care-btn"])
                    
                    with gr.Row():
                        praise_btn = gr.Button("👍 Praise", elem_classes=["care-btn"])
                        scold_btn = gr.Button("👎 Scold", elem_classes=["care-btn"])
                    
                    with gr.Row():
                        rest_btn = gr.Button("😴 Rest", elem_classes=["care-btn"])
                        play_btn = gr.Button("🎾 Play", elem_classes=["care-btn"])
                
                # Speech interaction
                gr.HTML("<h4>🎤 Talk to Your DigiPal</h4>")
                
                # Audio input (will be implemented in future tasks)
                audio_input = gr.Audio(
                    label="Record your voice",
                    type="numpy",
                    elem_classes=["audio-input"]
                )
                
                # Text input as alternative
                text_input = gr.Textbox(
                    label="Or type a message",
                    placeholder="Say something to your DigiPal...",
                    elem_classes=["text-input"]
                )
                
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    elem_classes=["send-btn"]
                )
                
                # Response display
                with gr.Group(elem_classes=["response-panel"]):
                    gr.HTML("<h4>💬 DigiPal Response</h4>")
                    
                    response_display = gr.HTML(
                        "<p>Your DigiPal is waiting for you to say something!</p>",
                        elem_classes=["response-display"]
                    )
                    
                    # Conversation history
                    with gr.Accordion("Conversation History", open=False):
                        conversation_history = gr.HTML(
                            "",
                            elem_classes=["conversation-history"]
                        )
        
        return {
            'pet_image': pet_image,
            'pet_name_display': pet_name_display,
            'status_info': status_info,
            'attributes_display': attributes_display,
            'feed_btn': feed_btn,
            'train_btn': train_btn,
            'praise_btn': praise_btn,
            'scold_btn': scold_btn,
            'rest_btn': rest_btn,
            'play_btn': play_btn,
            'audio_input': audio_input,
            'text_input': text_input,
            'send_btn': send_btn,
            'response_display': response_display,
            'conversation_history': conversation_history
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
                fn=lambda egg=egg_type: self._handle_egg_selection(egg, user_state.value),
                inputs=[user_state],
                outputs=[
                    egg_components['egg_status'],
                    main_tabs
                ]
            )
        
        # Main interface handlers
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
                main_components['conversation_history']
            ]
        )
        
        # Care action handlers
        care_actions = {
            'feed': main_components['feed_btn'],
            'train': main_components['train_btn'],
            'praise': main_components['praise_btn'],
            'scold': main_components['scold_btn'],
            'rest': main_components['rest_btn'],
            'play': main_components['play_btn']
        }
        
        for action_name, btn in care_actions.items():
            btn.click(
                fn=lambda action=action_name: self._handle_care_action(action, user_state.value),
                inputs=[user_state],
                outputs=[
                    main_components['response_display'],
                    main_components['status_info'],
                    main_components['attributes_display']
                ]
            )
    
    def _handle_login(self, token: str, offline_mode: bool, 
                     current_user: Optional[str], current_token: Optional[str]) -> Tuple:
        """Handle user login."""
        if not token:
            return (
                '<div class="error">Please enter a token</div>',
                current_user,
                current_token,
                gr.Tabs(selected="auth_tab")
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
                    gr.Tabs(selected="main_tab")
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
                    gr.Tabs(selected="egg_tab")
                )
        else:
            error_msg = f'<div class="error">Login failed: {auth_result.error_message}</div>'
            return (
                error_msg,
                current_user,
                current_token,
                gr.Tabs(selected="auth_tab")
            )
    
    def _handle_egg_selection(self, egg_type: EggType, user_id: Optional[str]) -> Tuple:
        """Handle egg selection."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                gr.Tabs(selected="auth_tab")
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
                gr.Tabs(selected="main_tab")
            )
            
        except Exception as e:
            logger.error(f"Error creating new pet: {e}")
            return (
                f'<div class="error">Failed to create DigiPal: {str(e)}</div>',
                gr.Tabs(selected="egg_tab")
            )
    
    def _handle_text_interaction(self, text: str, user_id: Optional[str]) -> Tuple:
        """Handle text interaction with DigiPal."""
        if not user_id or not text.strip():
            return (
                '<div class="error">Please enter a message</div>',
                text,
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
                
                # Update conversation history
                history_html = self._format_conversation_history(user_id)
                
                return (
                    response_html,
                    "",  # Clear text input
                    status_html,
                    attributes_html,
                    history_html
                )
            else:
                return (
                    f'<div class="error">Interaction failed: {interaction.pet_response}</div>',
                    text,
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
                ""
            )
    
    def _handle_care_action(self, action: str, user_id: Optional[str]) -> Tuple:
        """Handle care action."""
        if not user_id:
            return (
                '<div class="error">Please login first</div>',
                "",
                ""
            )
        
        try:
            # Apply care action
            success, interaction = self.digipal_core.apply_care_action(user_id, action)
            
            if success:
                response_html = f'''
                    <div class="care-response">
                        <p><strong>Care Action:</strong> {action.title()}</p>
                        <p><strong>DigiPal:</strong> {interaction.pet_response}</p>
                    </div>
                '''
                
                # Get updated pet state
                pet_state = self.digipal_core.get_pet_state(user_id)
                status_html, attributes_html = self._format_pet_status(pet_state)
                
                return (
                    response_html,
                    status_html,
                    attributes_html
                )
            else:
                return (
                    f'<div class="error">Care action failed: {interaction.pet_response}</div>',
                    "",
                    ""
                )
                
        except Exception as e:
            logger.error(f"Error applying care action: {e}")
            return (
                f'<div class="error">Error: {str(e)}</div>',
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
        """
    
    def launch_interface(self, share: bool = False, server_name: str = "127.0.0.1", 
                        server_port: int = 7860, debug: bool = False) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
        """
        if not self.app:
            self.app = self.create_interface()
        
        logger.info(f"Launching DigiPal interface on {server_name}:{server_port}")
        
        self.app.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True
        )