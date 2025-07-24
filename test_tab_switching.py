#!/usr/bin/env python3
"""
Simple test to debug Gradio tab switching issue.
"""

import gradio as gr

def handle_login(token, offline_mode):
    """Test login handler that should switch tabs."""
    if not token:
        return "Please enter a token", 0  # Stay on login tab
    
    if offline_mode:
        return f"Logged in offline with token: {token[:10]}...", 1  # Go to tab 1
    else:
        return f"Logged in with token: {token[:10]}...", 2  # Go to tab 2

def create_test_interface():
    """Create a simple test interface to debug tab switching."""
    
    with gr.Blocks(title="Tab Switching Test") as interface:
        
        with gr.Tabs(selected=0) as main_tabs:
            
            # Tab 0 - Login
            with gr.Tab("Login", id=0):
                gr.HTML("<h2>Login Tab</h2>")
                token_input = gr.Textbox(label="Token", placeholder="Enter any token...")
                offline_toggle = gr.Checkbox(label="Offline Mode", value=False)
                login_btn = gr.Button("Login", variant="primary")
                status_display = gr.HTML("")
            
            # Tab 1 - Egg Selection
            with gr.Tab("Egg Selection", id=1):
                gr.HTML("<h2>Egg Selection Tab</h2>")
                gr.HTML("<p>This tab should show when offline mode is enabled.</p>")
            
            # Tab 2 - Main Interface
            with gr.Tab("Main Interface", id=2):
                gr.HTML("<h2>Main Interface Tab</h2>")
                gr.HTML("<p>This tab should show when offline mode is disabled.</p>")
        
        # Event handler
        login_btn.click(
            fn=handle_login,
            inputs=[token_input, offline_toggle],
            outputs=[status_display, main_tabs]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_test_interface()
    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,
        debug=True
    )