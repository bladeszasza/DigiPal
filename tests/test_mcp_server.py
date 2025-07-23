"""
Tests for DigiPal MCP server functionality.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from digipal.mcp.server import MCPServer
from digipal.core.digipal_core import DigiPalCore, PetState
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication

from mcp.types import CallToolResult, TextContent


class TestMCPServer:
    """Test cases for MCPServer class."""
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        return Mock(spec=StorageManager)
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Create mock AI communication."""
        return Mock(spec=AICommunication)
    
    @pytest.fixture
    def mock_digipal_core(self, mock_storage_manager, mock_ai_communication):
        """Create mock DigiPal core."""
        core = Mock(spec=DigiPalCore)
        core.storage_manager = mock_storage_manager
        core.ai_communication = mock_ai_communication
        return core
    
    @pytest.fixture
    def mcp_server(self, mock_digipal_core):
        """Create MCP server instance."""
        return MCPServer(mock_digipal_core, "test-server")
    
    @pytest.fixture
    def sample_pet(self):
        """Create sample DigiPal for testing."""
        return DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.BABY
        )
    
    @pytest.fixture
    def sample_pet_state(self, sample_pet):
        """Create sample PetState for testing."""
        return PetState(sample_pet)
    
    def test_mcp_server_initialization(self, mock_digipal_core):
        """Test MCP server initialization."""
        server = MCPServer(mock_digipal_core, "test-server")
        
        assert server.digipal_core == mock_digipal_core
        assert server.server_name == "test-server"
        assert server.server is not None
        assert isinstance(server.authenticated_users, dict)
        assert isinstance(server.user_permissions, dict)
    
    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test listing available tools."""
        # Get the list_tools handler
        list_tools_handler = None
        for handler in mcp_server.server._tool_list_handlers:
            list_tools_handler = handler
            break
        
        assert list_tools_handler is not None
        
        # Call the handler
        result = await list_tools_handler()
        
        # Verify tools are returned
        assert hasattr(result, 'tools')
        assert len(result.tools) > 0
        
        # Check for expected tools
        tool_names = [tool.name for tool in result.tools]
        expected_tools = [
            "get_pet_status",
            "interact_with_pet", 
            "apply_care_action",
            "create_new_pet",
            "get_pet_statistics",
            "trigger_evolution",
            "get_available_actions"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    @pytest.mark.asyncio
    async def test_get_pet_status_success(self, mcp_server, sample_pet_state):
        """Test successful get_pet_status tool call."""
        # Mock the core method
        mcp_server.digipal_core.get_pet_state.return_value = sample_pet_state
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        assert call_tool_handler is not None
        
        # Call the handler
        result = await call_tool_handler("get_pet_status", {"user_id": "test_user"})
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert "TestPal" in result.content[0].text
        assert "BABY" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_get_pet_status_no_pet(self, mcp_server):
        """Test get_pet_status when no pet exists."""
        # Mock the core method to return None
        mcp_server.digipal_core.get_pet_state.return_value = None
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("get_pet_status", {"user_id": "test_user"})
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "No DigiPal found" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_interact_with_pet_success(self, mcp_server):
        """Test successful interact_with_pet tool call."""
        # Mock interaction result
        interaction = Interaction(
            user_input="hello",
            pet_response="Hello there!",
            success=True,
            result=InteractionResult.SUCCESS,
            attribute_changes={"happiness": 5}
        )
        
        mcp_server.digipal_core.process_interaction.return_value = (True, interaction)
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("interact_with_pet", {
            "user_id": "test_user",
            "message": "hello"
        })
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "Hello there!" in result.content[0].text
        assert "happiness: +5" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_interact_with_pet_failure(self, mcp_server):
        """Test failed interact_with_pet tool call."""
        # Mock failed interaction
        interaction = Interaction(
            user_input="invalid",
            pet_response="I don't understand",
            success=False,
            result=InteractionResult.FAILURE
        )
        
        mcp_server.digipal_core.process_interaction.return_value = (False, interaction)
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("interact_with_pet", {
            "user_id": "test_user",
            "message": "invalid"
        })
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "I don't understand" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_apply_care_action_success(self, mcp_server):
        """Test successful apply_care_action tool call."""
        # Mock care action result
        interaction = Interaction(
            user_input="meat",
            pet_response="Yummy! I feel stronger!",
            success=True,
            result=InteractionResult.SUCCESS,
            attribute_changes={"hp": 10, "happiness": 3}
        )
        
        mcp_server.digipal_core.apply_care_action.return_value = (True, interaction)
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("apply_care_action", {
            "user_id": "test_user",
            "action": "meat"
        })
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "Care Action Applied: meat" in result.content[0].text
        assert "Yummy! I feel stronger!" in result.content[0].text
        assert "hp: +10" in result.content[0].text
        assert "happiness: +3" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_create_new_pet_success(self, mcp_server, sample_pet):
        """Test successful create_new_pet tool call."""
        # Mock no existing pet and successful creation
        mcp_server.digipal_core.load_existing_pet.return_value = None
        mcp_server.digipal_core.create_new_pet.return_value = sample_pet
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("create_new_pet", {
            "user_id": "test_user",
            "egg_type": "red",
            "name": "TestPal"
        })
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "Successfully created new red DigiPal 'TestPal'" in result.content[0].text
        assert sample_pet.id in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_create_new_pet_already_exists(self, mcp_server, sample_pet):
        """Test create_new_pet when pet already exists."""
        # Mock existing pet
        mcp_server.digipal_core.load_existing_pet.return_value = sample_pet
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("create_new_pet", {
            "user_id": "test_user",
            "egg_type": "red"
        })
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "already has a DigiPal" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_create_new_pet_invalid_egg_type(self, mcp_server):
        """Test create_new_pet with invalid egg type."""
        # Mock no existing pet
        mcp_server.digipal_core.load_existing_pet.return_value = None
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("create_new_pet", {
            "user_id": "test_user",
            "egg_type": "purple"  # Invalid egg type
        })
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "Invalid egg type" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_get_pet_statistics_success(self, mcp_server):
        """Test successful get_pet_statistics tool call."""
        # Mock statistics
        mock_stats = {
            'basic_info': {
                'name': 'TestPal',
                'id': 'test-id',
                'life_stage': 'baby',
                'generation': 1,
                'age_hours': 5.5,
                'egg_type': 'red'
            },
            'attributes': {
                'hp': 100,
                'mp': 50,
                'offense': 15,
                'defense': 8
            },
            'care_assessment': {
                'care_quality': 'good',
                'overall_score': 75
            },
            'evolution_status': {
                'eligible': False,
                'next_stage': None
            },
            'personality_traits': {
                'playful': 0.7,
                'curious': 0.8
            },
            'learned_commands': ['eat', 'sleep', 'good']
        }
        
        mcp_server.digipal_core.get_pet_statistics.return_value = mock_stats
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("get_pet_statistics", {"user_id": "test_user"})
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "DigiPal Statistics Report" in result.content[0].text
        assert "TestPal" in result.content[0].text
        assert "Care Quality: good" in result.content[0].text
        assert "Overall Score: 75/100" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_trigger_evolution_success(self, mcp_server):
        """Test successful trigger_evolution tool call."""
        # Mock evolution result
        from digipal.core.evolution_controller import EvolutionResult
        
        evolution_result = EvolutionResult(
            success=True,
            old_stage=LifeStage.BABY,
            new_stage=LifeStage.CHILD,
            message="Evolution successful!",
            attribute_changes={"hp": 20, "offense": 5}
        )
        
        mcp_server.digipal_core.trigger_evolution.return_value = (True, evolution_result)
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("trigger_evolution", {
            "user_id": "test_user",
            "force": False
        })
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "Evolution successful!" in result.content[0].text
        assert "From: baby" in result.content[0].text
        assert "To: child" in result.content[0].text
        assert "hp: +20" in result.content[0].text
        assert "offense: +5" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_get_available_actions_success(self, mcp_server):
        """Test successful get_available_actions tool call."""
        # Mock available actions
        mock_actions = ["meat", "fish", "rest", "play", "praise"]
        mcp_server.digipal_core.get_care_actions.return_value = mock_actions
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler
        result = await call_tool_handler("get_available_actions", {"user_id": "test_user"})
        
        # Verify result
        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert "Available Care Actions:" in result.content[0].text
        for action in mock_actions:
            assert f"- {action}" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_missing_user_id(self, mcp_server):
        """Test tool call without user_id."""
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call the handler without user_id
        result = await call_tool_handler("get_pet_status", {})
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "user_id is required" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self, mcp_server):
        """Test call to unknown tool."""
        # Get the call_tool handler
        call_tool_handler = None
        for handler in mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # Call unknown tool
        result = await call_tool_handler("unknown_tool", {"user_id": "test_user"})
        
        # Verify error result
        assert isinstance(result, CallToolResult)
        assert result.isError
        assert "Unknown tool: unknown_tool" in result.content[0].text
    
    def test_authentication_methods(self, mcp_server):
        """Test authentication and permission methods."""
        user_id = "test_user"
        
        # Test authentication
        assert mcp_server.authenticate_user(user_id)
        assert mcp_server._is_user_authenticated(user_id)
        
        # Test permissions
        permissions = ["get_pet_status", "interact_with_pet"]
        mcp_server.set_user_permissions(user_id, permissions)
        assert mcp_server._has_permission(user_id, "get_pet_status")
        assert mcp_server._has_permission(user_id, "interact_with_pet")
        
        # Test revoke access
        mcp_server.revoke_user_access(user_id)
        assert not mcp_server._is_user_authenticated(user_id)
    
    def test_format_pet_status(self, mcp_server, sample_pet_state):
        """Test pet status formatting."""
        status_dict = sample_pet_state.to_dict()
        formatted = mcp_server._format_pet_status(status_dict)
        
        assert "DigiPal Status Report" in formatted
        assert "TestPal" in formatted
        assert "baby" in formatted.lower()
        assert "HP:" in formatted
        assert "Attributes:" in formatted
    
    def test_format_pet_statistics(self, mcp_server):
        """Test pet statistics formatting."""
        mock_stats = {
            'basic_info': {
                'name': 'TestPal',
                'id': 'test-id',
                'life_stage': 'baby',
                'generation': 1,
                'age_hours': 5.5,
                'egg_type': 'red'
            },
            'attributes': {
                'hp': 100,
                'mp': 50
            },
            'care_assessment': {
                'care_quality': 'good',
                'overall_score': 75
            },
            'evolution_status': {
                'eligible': False
            },
            'personality_traits': {
                'playful': 0.7
            },
            'learned_commands': ['eat', 'sleep']
        }
        
        formatted = mcp_server._format_pet_statistics(mock_stats)
        
        assert "DigiPal Statistics Report" in formatted
        assert "TestPal" in formatted
        assert "Care Quality: good" in formatted
        assert "Overall Score: 75/100" in formatted
        assert "Playful: 0.70" in formatted
        assert "eat, sleep" in formatted
    
    def test_shutdown(self, mcp_server):
        """Test server shutdown."""
        # Add some test data
        mcp_server.authenticate_user("test_user")
        mcp_server.set_user_permissions("test_user", ["test_permission"])
        
        # Shutdown
        mcp_server.shutdown()
        
        # Verify cleanup
        mcp_server.digipal_core.stop_background_updates.assert_called_once()
        mcp_server.digipal_core.shutdown.assert_called_once()
        assert len(mcp_server.authenticated_users) == 0
        assert len(mcp_server.user_permissions) == 0


class TestMCPServerIntegration:
    """Integration tests for MCP server with real components."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def real_storage_manager(self, temp_db_path):
        """Create real storage manager with temporary database."""
        return StorageManager(temp_db_path, "test_assets")
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Create mock AI communication for integration tests."""
        ai_comm = Mock(spec=AICommunication)
        
        # Mock process_interaction method
        def mock_process_interaction(text, pet):
            return Interaction(
                user_input=text,
                interpreted_command="test_command",
                pet_response=f"I heard you say: {text}",
                success=True,
                result=InteractionResult.SUCCESS,
                attribute_changes={"happiness": 1}
            )
        
        ai_comm.process_interaction = mock_process_interaction
        ai_comm.unload_all_models = Mock()
        
        return ai_comm
    
    @pytest.fixture
    def real_digipal_core(self, real_storage_manager, mock_ai_communication):
        """Create real DigiPal core with mocked AI."""
        return DigiPalCore(real_storage_manager, mock_ai_communication)
    
    @pytest.fixture
    def integration_mcp_server(self, real_digipal_core):
        """Create MCP server with real DigiPal core."""
        return MCPServer(real_digipal_core, "integration-test-server")
    
    @pytest.mark.asyncio
    async def test_full_pet_lifecycle_via_mcp(self, integration_mcp_server):
        """Test complete pet lifecycle through MCP interface."""
        user_id = "integration_test_user"
        
        # Authenticate user
        integration_mcp_server.authenticate_user(user_id)
        
        # Get the call_tool handler
        call_tool_handler = None
        for handler in integration_mcp_server.server._tool_call_handlers:
            call_tool_handler = handler
            break
        
        # 1. Create new pet
        result = await call_tool_handler("create_new_pet", {
            "user_id": user_id,
            "egg_type": "red",
            "name": "IntegrationPal"
        })
        
        assert not result.isError
        assert "Successfully created new red DigiPal 'IntegrationPal'" in result.content[0].text
        
        # 2. Get pet status
        result = await call_tool_handler("get_pet_status", {"user_id": user_id})
        
        assert not result.isError
        assert "IntegrationPal" in result.content[0].text
        assert "EGG" in result.content[0].text
        
        # 3. Interact with pet (should trigger hatching)
        result = await call_tool_handler("interact_with_pet", {
            "user_id": user_id,
            "message": "Hello little one!"
        })
        
        assert not result.isError
        assert "I heard you say: Hello little one!" in result.content[0].text
        
        # 4. Check status after interaction
        result = await call_tool_handler("get_pet_status", {"user_id": user_id})
        
        assert not result.isError
        # Pet should still be in egg stage until first speech interaction
        
        # 5. Apply care action
        result = await call_tool_handler("apply_care_action", {
            "user_id": user_id,
            "action": "meat"
        })
        
        assert not result.isError
        assert "Care Action Applied: meat" in result.content[0].text
        
        # 6. Get available actions
        result = await call_tool_handler("get_available_actions", {"user_id": user_id})
        
        assert not result.isError
        assert "Available Care Actions:" in result.content[0].text
        
        # 7. Get comprehensive statistics
        result = await call_tool_handler("get_pet_statistics", {"user_id": user_id})
        
        assert not result.isError
        assert "DigiPal Statistics Report" in result.content[0].text
        assert "IntegrationPal" in result.content[0].text
    
    def test_mcp_server_with_real_storage(self, integration_mcp_server):
        """Test MCP server with real storage persistence."""
        user_id = "persistence_test_user"
        
        # Create pet through core (simulating previous session)
        pet = integration_mcp_server.digipal_core.create_new_pet(
            EggType.BLUE, user_id, "PersistentPal"
        )
        
        # Verify pet was saved
        loaded_pet = integration_mcp_server.digipal_core.load_existing_pet(user_id)
        assert loaded_pet is not None
        assert loaded_pet.name == "PersistentPal"
        assert loaded_pet.egg_type == EggType.BLUE
        
        # Get pet state through MCP
        pet_state = integration_mcp_server.digipal_core.get_pet_state(user_id)
        assert pet_state is not None
        assert pet_state.name == "PersistentPal"


if __name__ == "__main__":
    pytest.main([__file__])