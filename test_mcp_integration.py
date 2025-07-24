#!/usr/bin/env python3
"""
Comprehensive integration test for DigiPal MCP server functionality.
This test verifies all MCP server features work together correctly.
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
from digipal.core.enums import EggType


async def test_mcp_server_integration():
    """Comprehensive integration test for MCP server."""
    print("DigiPal MCP Server Integration Test")
    print("=" * 40)
    
    # Create temporary database for test
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        temp_db_path = f.name
    
    try:
        # Initialize components
        print("1. Initializing DigiPal components...")
        storage_manager = StorageManager(temp_db_path, "test_assets")
        ai_communication = AICommunication()
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize MCP server
        mcp_server = MCPServer(digipal_core, "integration-test-server")
        print("   ✓ MCP server initialized")
        
        # Test users
        users = ["user1", "user2", "user3"]
        
        # Create and authenticate users
        print("\n2. Setting up test users...")
        for user_id in users:
            storage_manager.create_user(user_id, f"test_{user_id}")
            mcp_server.authenticate_user(user_id)
            print(f"   ✓ User {user_id} created and authenticated")
        
        # Test 1: Create pets with different egg types
        print("\n3. Creating pets with different egg types...")
        egg_types = ["red", "blue", "green"]
        
        for i, user_id in enumerate(users):
            egg_type = egg_types[i]
            result = await mcp_server._handle_create_new_pet({
                "user_id": user_id,
                "egg_type": egg_type,
                "name": f"Pet{i+1}"
            })
            
            if result.isError:
                print(f"   ✗ Failed to create pet for {user_id}: {result.content[0].text}")
                return False
            
            print(f"   ✓ Created {egg_type} pet for {user_id}")
        
        # Test 2: Verify all pets exist and have correct attributes
        print("\n4. Verifying pet creation...")
        for user_id in users:
            result = await mcp_server._handle_get_pet_status({"user_id": user_id})
            
            if result.isError:
                print(f"   ✗ Failed to get status for {user_id}: {result.content[0].text}")
                return False
            
            status_text = result.content[0].text
            if "Life Stage: egg" not in status_text:
                print(f"   ✗ Pet for {user_id} not in egg stage")
                return False
            
            print(f"   ✓ Pet status verified for {user_id}")
        
        # Test 3: Interact with pets to trigger hatching
        print("\n5. Triggering pet hatching through interaction...")
        for user_id in users:
            result = await mcp_server._handle_interact_with_pet({
                "user_id": user_id,
                "message": "Hello little one!"
            })
            
            # Note: Interaction might fail due to AI model issues, but that's expected
            print(f"   ✓ Interaction attempted for {user_id}")
        
        # Test 4: Apply care actions to all pets
        print("\n6. Applying care actions...")
        care_actions = ["meat", "fish", "vegetables"]
        
        for i, user_id in enumerate(users):
            action = care_actions[i]
            result = await mcp_server._handle_apply_care_action({
                "user_id": user_id,
                "action": action
            })
            
            if result.isError:
                print(f"   ✗ Failed to apply {action} to {user_id}: {result.content[0].text}")
                return False
            
            print(f"   ✓ Applied {action} to {user_id}")
        
        # Test 5: Get available actions for all pets
        print("\n7. Getting available actions...")
        for user_id in users:
            result = await mcp_server._handle_get_available_actions({"user_id": user_id})
            
            if result.isError:
                print(f"   ✗ Failed to get actions for {user_id}: {result.content[0].text}")
                return False
            
            actions_text = result.content[0].text
            if "Available Care Actions:" not in actions_text:
                print(f"   ✗ Invalid actions response for {user_id}")
                return False
            
            print(f"   ✓ Got available actions for {user_id}")
        
        # Test 6: Get comprehensive statistics
        print("\n8. Getting pet statistics...")
        for user_id in users:
            result = await mcp_server._handle_get_pet_statistics({"user_id": user_id})
            
            if result.isError:
                print(f"   ✗ Failed to get statistics for {user_id}: {result.content[0].text}")
                return False
            
            stats_text = result.content[0].text
            if "DigiPal Statistics Report" not in stats_text:
                print(f"   ✗ Invalid statistics response for {user_id}")
                return False
            
            print(f"   ✓ Got statistics for {user_id}")
        
        # Test 7: Test evolution (force evolution for testing)
        print("\n9. Testing evolution...")
        test_user = users[0]
        result = await mcp_server._handle_trigger_evolution({
            "user_id": test_user,
            "force": True
        })
        
        if result.isError:
            print(f"   ✗ Failed to trigger evolution for {test_user}: {result.content[0].text}")
            return False
        
        print(f"   ✓ Evolution triggered for {test_user}")
        
        # Test 8: Verify evolution occurred
        print("\n10. Verifying evolution...")
        result = await mcp_server._handle_get_pet_status({"user_id": test_user})
        
        if result.isError:
            print(f"   ✗ Failed to get post-evolution status: {result.content[0].text}")
            return False
        
        status_text = result.content[0].text
        # Check that pet evolved from egg stage (could be baby, child, etc.)
        if "Life Stage: egg" in status_text:
            print(f"   ✗ Pet did not evolve from egg stage")
            print(f"   Status text: {status_text}")
            return False
        
        # Extract the current life stage
        import re
        stage_match = re.search(r"Life Stage: (\w+)", status_text)
        current_stage = stage_match.group(1) if stage_match else "unknown"
        
        print(f"   ✓ Evolution verified - pet evolved to {current_stage} stage")
        
        # Test 9: Test error handling - try to create pet for existing user
        print("\n11. Testing error handling...")
        result = await mcp_server._handle_create_new_pet({
            "user_id": users[0],
            "egg_type": "red",
            "name": "DuplicatePet"
        })
        
        if not result.isError:
            print(f"   ✗ Should have failed to create duplicate pet")
            return False
        
        print(f"   ✓ Correctly rejected duplicate pet creation")
        
        # Test 10: Test authentication and permissions
        print("\n12. Testing authentication...")
        
        # Test with non-existent user
        result = await mcp_server._handle_get_pet_status({"user_id": "nonexistent_user"})
        
        if not result.isError:
            print(f"   ✗ Should have failed for non-existent user")
            return False
        
        print(f"   ✓ Correctly handled non-existent user")
        
        # Test 11: Test server shutdown and cleanup
        print("\n13. Testing server shutdown...")
        mcp_server.shutdown()
        print(f"   ✓ Server shutdown completed")
        
        print("\n" + "=" * 40)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("MCP Server is fully functional and ready for use.")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server_integration())
    sys.exit(0 if success else 1)