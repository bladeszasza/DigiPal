#!/usr/bin/env python3
"""
Demo script showing how to use the DigiPal MCP server.

This script demonstrates the basic functionality of the MCP server
by creating a pet and performing various interactions.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer


async def demo_mcp_server():
    """Demonstrate MCP server functionality."""
    print("DigiPal MCP Server Demo")
    print("=" * 30)
    
    # Create temporary database for demo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        temp_db_path = f.name
    
    try:
        # Initialize components
        print("Initializing DigiPal components...")
        storage_manager = StorageManager(temp_db_path, "demo_assets")
        ai_communication = AICommunication()
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize MCP server
        mcp_server = MCPServer(digipal_core, "digipal-demo-server")
        print("MCP server initialized successfully!")
        
        # Demo user
        user_id = "demo_user"
        
        # Create user (required for database foreign key constraints)
        storage_manager.create_user(user_id, "demo_user")
        mcp_server.authenticate_user(user_id)
        print(f"User '{user_id}' created and authenticated")
        
        # 1. Create a new pet
        print("\n1. Creating a new DigiPal...")
        result = await mcp_server._handle_create_new_pet({
            "user_id": user_id,
            "egg_type": "red",
            "name": "DemoPal"
        })
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
            return
        
        print(f"Success: {result.content[0].text}")
        
        # 2. Get pet status
        print("\n2. Getting pet status...")
        result = await mcp_server._handle_get_pet_status({"user_id": user_id})
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Pet Status:")
            print(result.content[0].text)
        
        # 3. Interact with pet
        print("\n3. Interacting with pet...")
        result = await mcp_server._handle_interact_with_pet({
            "user_id": user_id,
            "message": "Hello, little one!"
        })
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Interaction Result:")
            print(result.content[0].text)
        
        # 4. Apply care action
        print("\n4. Feeding the pet...")
        result = await mcp_server._handle_apply_care_action({
            "user_id": user_id,
            "action": "meat"
        })
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Care Action Result:")
            print(result.content[0].text)
        
        # 5. Get available actions
        print("\n5. Getting available care actions...")
        result = await mcp_server._handle_get_available_actions({"user_id": user_id})
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Available Actions:")
            print(result.content[0].text)
        
        # 6. Get comprehensive statistics
        print("\n6. Getting pet statistics...")
        result = await mcp_server._handle_get_pet_statistics({"user_id": user_id})
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Pet Statistics:")
            print(result.content[0].text)
        
        # 7. Get updated status
        print("\n7. Getting updated pet status...")
        result = await mcp_server._handle_get_pet_status({"user_id": user_id})
        
        if result.isError:
            print(f"Error: {result.content[0].text}")
        else:
            print("Updated Pet Status:")
            print(result.content[0].text)
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        print("Cleanup completed.")


if __name__ == "__main__":
    asyncio.run(demo_mcp_server())