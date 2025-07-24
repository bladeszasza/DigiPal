#!/usr/bin/env python3
"""
Debug evolution issue in MCP server.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer


async def debug_evolution():
    """Debug evolution issue."""
    print("Debugging Evolution Issue")
    print("=" * 30)
    
    # Create temporary database for test
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        temp_db_path = f.name
    
    try:
        # Initialize components
        storage_manager = StorageManager(temp_db_path, "test_assets")
        ai_communication = AICommunication()
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        mcp_server = MCPServer(digipal_core, "debug-server")
        
        # Create user
        user_id = "debug_user"
        storage_manager.create_user(user_id, "debug_user")
        mcp_server.authenticate_user(user_id)
        
        # Create pet
        result = await mcp_server._handle_create_new_pet({
            "user_id": user_id,
            "egg_type": "red",
            "name": "DebugPal"
        })
        print(f"Pet creation: {result.content[0].text}")
        
        # Get initial status
        result = await mcp_server._handle_get_pet_status({"user_id": user_id})
        print(f"\nInitial status:")
        print(result.content[0].text)
        
        # Try evolution
        result = await mcp_server._handle_trigger_evolution({
            "user_id": user_id,
            "force": True
        })
        print(f"\nEvolution result:")
        print(f"Error: {result.isError}")
        print(f"Content: {result.content[0].text}")
        
        # Get status after evolution attempt
        result = await mcp_server._handle_get_pet_status({"user_id": user_id})
        print(f"\nStatus after evolution:")
        print(result.content[0].text)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)


if __name__ == "__main__":
    asyncio.run(debug_evolution())