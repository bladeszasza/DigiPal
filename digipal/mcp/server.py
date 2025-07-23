"""
MCP Server implementation for DigiPal.

This module provides the MCPServer class that implements the Model Context Protocol
for external system integration with DigiPal functionality.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    CallToolResult,
    ListToolsResult
)

from ..core.digipal_core import DigiPalCore, PetState
from ..core.models import DigiPal, Interaction
from ..core.enums import EggType, LifeStage, CareActionType

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server implementation for DigiPal.
    
    Provides MCP protocol compliance for external system integration,
    allowing other AI systems to interact with DigiPal pets.
    """
    
    def __init__(self, digipal_core: DigiPalCore, server_name: str = "digipal-mcp-server"):
        """
        Initialize MCP Server.
        
        Args:
            digipal_core: DigiPal core engine instance
            server_name: Name of the MCP server
        """
        self.digipal_core = digipal_core
        self.server_name = server_name
        self.server = Server(server_name)
        
        # Authentication and permissions
        self.authenticated_users: Dict[str, bool] = {}
        self.user_permissions: Dict[str, List[str]] = {}
        
        # Register MCP handlers
        self._register_handlers()
        
        logger.info(f"MCPServer initialized: {server_name}")
    
    def _register_handlers(self):
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available DigiPal interaction tools."""
            tools = [
                Tool(
                    name="get_pet_status",
                    description="Get the current status and attributes of a user's DigiPal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID to get pet status for"
                            }
                        },
                        "required": ["user_id"]
                    }
                ),
                Tool(
                    name="interact_with_pet",
                    description="Send a text message to interact with a user's DigiPal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID whose pet to interact with"
                            },
                            "message": {
                                "type": "string",
                                "description": "Message to send to the DigiPal"
                            }
                        },
                        "required": ["user_id", "message"]
                    }
                ),
                Tool(
                    name="apply_care_action",
                    description="Apply a care action to a user's DigiPal (feeding, training, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID whose pet to care for"
                            },
                            "action": {
                                "type": "string",
                                "description": "Care action to apply",
                                "enum": [
                                    "meat", "fish", "vegetables", "protein", "vitamins",
                                    "strength_training", "speed_training", "defense_training",
                                    "brain_training", "rest", "play", "praise", "scold"
                                ]
                            }
                        },
                        "required": ["user_id", "action"]
                    }
                ),
                Tool(
                    name="create_new_pet",
                    description="Create a new DigiPal for a user",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID to create pet for"
                            },
                            "egg_type": {
                                "type": "string",
                                "description": "Type of egg to create",
                                "enum": ["red", "blue", "green"]
                            },
                            "name": {
                                "type": "string",
                                "description": "Name for the new DigiPal",
                                "default": "DigiPal"
                            }
                        },
                        "required": ["user_id", "egg_type"]
                    }
                ),
                Tool(
                    name="get_pet_statistics",
                    description="Get comprehensive statistics and analysis for a user's DigiPal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID to get statistics for"
                            }
                        },
                        "required": ["user_id"]
                    }
                ),
                Tool(
                    name="trigger_evolution",
                    description="Manually trigger evolution for a user's DigiPal if eligible",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID whose pet to evolve"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force evolution regardless of requirements",
                                "default": False
                            }
                        },
                        "required": ["user_id"]
                    }
                ),
                Tool(
                    name="get_available_actions",
                    description="Get list of available care actions for a user's DigiPal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID to get available actions for"
                            }
                        },
                        "required": ["user_id"]
                    }
                )
            ]
            
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls from MCP clients."""
            try:
                # Validate user authentication and permissions
                user_id = arguments.get("user_id")
                if not user_id:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Error: user_id is required for all DigiPal operations"
                        )],
                        isError=True
                    )
                
                # Check authentication (simplified for now)
                if not self._is_user_authenticated(user_id):
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Error: User not authenticated"
                        )],
                        isError=True
                    )
                
                # Check permissions
                if not self._has_permission(user_id, name):
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Error: User does not have permission for action: {name}"
                        )],
                        isError=True
                    )
                
                # Route to appropriate handler
                if name == "get_pet_status":
                    return await self._handle_get_pet_status(arguments)
                elif name == "interact_with_pet":
                    return await self._handle_interact_with_pet(arguments)
                elif name == "apply_care_action":
                    return await self._handle_apply_care_action(arguments)
                elif name == "create_new_pet":
                    return await self._handle_create_new_pet(arguments)
                elif name == "get_pet_statistics":
                    return await self._handle_get_pet_statistics(arguments)
                elif name == "trigger_evolution":
                    return await self._handle_trigger_evolution(arguments)
                elif name == "get_available_actions":
                    return await self._handle_get_available_actions(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Error: Unknown tool: {name}"
                        )],
                        isError=True
                    )
                    
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: {str(e)}"
                    )],
                    isError=True
                )
    
    async def _handle_get_pet_status(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_pet_status tool call."""
        user_id = arguments["user_id"]
        
        pet_state = self.digipal_core.get_pet_state(user_id)
        if not pet_state:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No DigiPal found for user: {user_id}"
                )],
                isError=True
            )
        
        status_dict = pet_state.to_dict()
        status_text = self._format_pet_status(status_dict)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=status_text
            )]
        )
    
    async def _handle_interact_with_pet(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle interact_with_pet tool call."""
        user_id = arguments["user_id"]
        message = arguments["message"]
        
        success, interaction = self.digipal_core.process_interaction(user_id, message)
        
        if not success:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Interaction failed: {interaction.pet_response}"
                )],
                isError=True
            )
        
        response_text = f"DigiPal Response: {interaction.pet_response}"
        if interaction.attribute_changes:
            changes = ", ".join([f"{attr}: {change:+d}" for attr, change in interaction.attribute_changes.items()])
            response_text += f"\nAttribute Changes: {changes}"
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=response_text
            )]
        )
    
    async def _handle_apply_care_action(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle apply_care_action tool call."""
        user_id = arguments["user_id"]
        action = arguments["action"]
        
        success, interaction = self.digipal_core.apply_care_action(user_id, action)
        
        if not success:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Care action failed: {interaction.pet_response}"
                )],
                isError=True
            )
        
        response_text = f"Care Action Applied: {action}\nResult: {interaction.pet_response}"
        if interaction.attribute_changes:
            changes = ", ".join([f"{attr}: {change:+d}" for attr, change in interaction.attribute_changes.items()])
            response_text += f"\nAttribute Changes: {changes}"
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=response_text
            )]
        )
    
    async def _handle_create_new_pet(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle create_new_pet tool call."""
        user_id = arguments["user_id"]
        egg_type_str = arguments["egg_type"]
        name = arguments.get("name", "DigiPal")
        
        # Check if user already has a pet
        existing_pet = self.digipal_core.load_existing_pet(user_id)
        if existing_pet:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"User {user_id} already has a DigiPal: {existing_pet.name}"
                )],
                isError=True
            )
        
        # Convert egg type string to enum
        try:
            egg_type = EggType(egg_type_str.upper())
        except ValueError:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Invalid egg type: {egg_type_str}. Must be red, blue, or green."
                )],
                isError=True
            )
        
        try:
            new_pet = self.digipal_core.create_new_pet(egg_type, user_id, name)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Successfully created new {egg_type_str} DigiPal '{name}' for user {user_id}. Pet ID: {new_pet.id}"
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Failed to create new pet: {str(e)}"
                )],
                isError=True
            )
    
    async def _handle_get_pet_statistics(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_pet_statistics tool call."""
        user_id = arguments["user_id"]
        
        statistics = self.digipal_core.get_pet_statistics(user_id)
        if not statistics:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No DigiPal found for user: {user_id}"
                )],
                isError=True
            )
        
        stats_text = self._format_pet_statistics(statistics)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=stats_text
            )]
        )
    
    async def _handle_trigger_evolution(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle trigger_evolution tool call."""
        user_id = arguments["user_id"]
        force = arguments.get("force", False)
        
        success, evolution_result = self.digipal_core.trigger_evolution(user_id, force)
        
        if not success:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Evolution failed: {evolution_result.message}"
                )],
                isError=True
            )
        
        response_text = f"Evolution successful!\nFrom: {evolution_result.old_stage.value}\nTo: {evolution_result.new_stage.value}"
        if evolution_result.attribute_changes:
            changes = ", ".join([f"{attr}: {change:+d}" for attr, change in evolution_result.attribute_changes.items()])
            response_text += f"\nAttribute Changes: {changes}"
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=response_text
            )]
        )
    
    async def _handle_get_available_actions(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_available_actions tool call."""
        user_id = arguments["user_id"]
        
        actions = self.digipal_core.get_care_actions(user_id)
        if not actions:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No DigiPal found for user: {user_id}"
                )],
                isError=True
            )
        
        actions_text = "Available Care Actions:\n" + "\n".join([f"- {action}" for action in actions])
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=actions_text
            )]
        )
    
    def _format_pet_status(self, status_dict: Dict[str, Any]) -> str:
        """Format pet status dictionary into readable text."""
        basic = status_dict.get('basic_info', {})
        attrs = status_dict.get('attributes', {})
        status = status_dict.get('status', {})
        
        text = f"DigiPal Status Report\n"
        text += f"=====================\n"
        text += f"Name: {status_dict.get('name', 'Unknown')}\n"
        text += f"Life Stage: {basic.get('life_stage', 'Unknown')}\n"
        text += f"Generation: {basic.get('generation', 0)}\n"
        text += f"Age: {status.get('age_hours', 0):.1f} hours\n"
        text += f"Status: {status.get('status_summary', 'Unknown')}\n"
        text += f"Needs Attention: {'Yes' if status.get('needs_attention', False) else 'No'}\n"
        text += f"Evolution Ready: {'Yes' if status.get('evolution_ready', False) else 'No'}\n\n"
        
        text += f"Attributes:\n"
        text += f"-----------\n"
        text += f"HP: {attrs.get('hp', 0)}\n"
        text += f"MP: {attrs.get('mp', 0)}\n"
        text += f"Offense: {attrs.get('offense', 0)}\n"
        text += f"Defense: {attrs.get('defense', 0)}\n"
        text += f"Speed: {attrs.get('speed', 0)}\n"
        text += f"Brains: {attrs.get('brains', 0)}\n"
        text += f"Discipline: {attrs.get('discipline', 0)}\n"
        text += f"Happiness: {attrs.get('happiness', 0)}\n"
        text += f"Weight: {attrs.get('weight', 0)}\n"
        text += f"Energy: {attrs.get('energy', 0)}\n"
        text += f"Care Mistakes: {attrs.get('care_mistakes', 0)}\n"
        
        return text
    
    def _format_pet_statistics(self, statistics: Dict[str, Any]) -> str:
        """Format pet statistics dictionary into readable text."""
        basic = statistics.get('basic_info', {})
        attrs = statistics.get('attributes', {})
        care = statistics.get('care_assessment', {})
        evolution = statistics.get('evolution_status', {})
        
        text = f"DigiPal Statistics Report\n"
        text += f"=========================\n"
        text += f"Name: {basic.get('name', 'Unknown')}\n"
        text += f"ID: {basic.get('id', 'Unknown')}\n"
        text += f"Life Stage: {basic.get('life_stage', 'Unknown')}\n"
        text += f"Generation: {basic.get('generation', 0)}\n"
        text += f"Age: {basic.get('age_hours', 0):.1f} hours\n"
        text += f"Egg Type: {basic.get('egg_type', 'Unknown')}\n\n"
        
        text += f"Current Attributes:\n"
        text += f"------------------\n"
        for attr, value in attrs.items():
            text += f"{attr.capitalize()}: {value}\n"
        
        text += f"\nCare Assessment:\n"
        text += f"---------------\n"
        text += f"Care Quality: {care.get('care_quality', 'Unknown')}\n"
        text += f"Overall Score: {care.get('overall_score', 0)}/100\n"
        
        text += f"\nEvolution Status:\n"
        text += f"----------------\n"
        text += f"Eligible for Evolution: {'Yes' if evolution.get('eligible', False) else 'No'}\n"
        if evolution.get('next_stage'):
            text += f"Next Stage: {evolution['next_stage']}\n"
        
        personality = statistics.get('personality_traits', {})
        if personality:
            text += f"\nPersonality Traits:\n"
            text += f"------------------\n"
            for trait, value in personality.items():
                text += f"{trait.capitalize()}: {value:.2f}\n"
        
        learned = statistics.get('learned_commands', [])
        if learned:
            text += f"\nLearned Commands:\n"
            text += f"----------------\n"
            text += ", ".join(learned)
        
        return text
    
    def _is_user_authenticated(self, user_id: str) -> bool:
        """
        Check if user is authenticated.
        
        For now, this is a simplified implementation.
        In production, this would integrate with proper authentication.
        """
        # For development/demo purposes, allow all users
        # In production, implement proper authentication
        return True
    
    def _has_permission(self, user_id: str, action: str) -> bool:
        """
        Check if user has permission for specific action.
        
        Args:
            user_id: User ID to check
            action: Action name to check permission for
            
        Returns:
            True if user has permission
        """
        # For development/demo purposes, allow all actions for authenticated users
        # In production, implement proper permission system
        if not self._is_user_authenticated(user_id):
            return False
        
        # Check user-specific permissions
        user_perms = self.user_permissions.get(user_id, [])
        if user_perms and action not in user_perms:
            return False
        
        return True
    
    def authenticate_user(self, user_id: str, token: Optional[str] = None) -> bool:
        """
        Authenticate a user for MCP access.
        
        Args:
            user_id: User ID to authenticate
            token: Authentication token (optional for now)
            
        Returns:
            True if authentication successful
        """
        # Simplified authentication for development
        # In production, validate token against HuggingFace or other auth provider
        self.authenticated_users[user_id] = True
        logger.info(f"User {user_id} authenticated for MCP access")
        return True
    
    def set_user_permissions(self, user_id: str, permissions: List[str]):
        """
        Set permissions for a user.
        
        Args:
            user_id: User ID
            permissions: List of allowed action names
        """
        self.user_permissions[user_id] = permissions
        logger.info(f"Set permissions for user {user_id}: {permissions}")
    
    def revoke_user_access(self, user_id: str):
        """
        Revoke access for a user.
        
        Args:
            user_id: User ID to revoke access for
        """
        self.authenticated_users.pop(user_id, None)
        self.user_permissions.pop(user_id, None)
        logger.info(f"Revoked access for user {user_id}")
    
    async def start_server(self, host: str = "localhost", port: int = 8000):
        """
        Start the MCP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"Starting MCP server on {host}:{port}")
        
        # Start background updates for DigiPal core
        self.digipal_core.start_background_updates()
        
        try:
            # Run the MCP server
            await self.server.run(
                transport="stdio",  # Use stdio transport for MCP
                options=InitializationOptions(
                    server_name=self.server_name,
                    server_version="1.0.0"
                )
            )
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
            raise
        finally:
            # Stop background updates
            self.digipal_core.stop_background_updates()
    
    def shutdown(self):
        """Shutdown the MCP server and cleanup resources."""
        logger.info("Shutting down MCP server")
        
        # Stop DigiPal core background updates
        self.digipal_core.stop_background_updates()
        
        # Shutdown DigiPal core
        self.digipal_core.shutdown()
        
        # Clear authentication data
        self.authenticated_users.clear()
        self.user_permissions.clear()
        
        logger.info("MCP server shutdown complete")