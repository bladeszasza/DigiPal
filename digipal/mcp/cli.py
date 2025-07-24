#!/usr/bin/env python3
"""
CLI tool for DigiPal MCP server.

This module provides a command-line interface for starting and managing
the DigiPal MCP server.
"""

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def start_server(args):
    """Start the MCP server."""
    print("Starting DigiPal MCP Server...")
    print(f"Database: {args.database}")
    print(f"Assets: {args.assets}")
    print(f"Server name: {args.name}")
    
    try:
        # Initialize components
        storage_manager = StorageManager(args.database, args.assets)
        ai_communication = AICommunication()
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize MCP server
        mcp_server = MCPServer(digipal_core, args.name)
        
        print("MCP Server initialized successfully!")
        print("Available tools:")
        print("- get_pet_status: Get current status of a user's DigiPal")
        print("- interact_with_pet: Send text message to DigiPal")
        print("- apply_care_action: Apply care actions (feeding, training, etc.)")
        print("- create_new_pet: Create new DigiPal for a user")
        print("- get_pet_statistics: Get comprehensive pet statistics")
        print("- trigger_evolution: Manually trigger evolution")
        print("- get_available_actions: Get available care actions")
        print()
        print("Server is ready to accept MCP connections...")
        
        # Start the server
        await mcp_server.start_server()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def demo_mode(args):
    """Run interactive demo of MCP server functionality."""
    print("DigiPal MCP Server Demo Mode")
    print("=" * 40)
    
    # Use temporary database for demo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        temp_db_path = f.name
    
    try:
        # Initialize components
        print("Initializing DigiPal components...")
        storage_manager = StorageManager(temp_db_path, "demo_assets")
        ai_communication = AICommunication()
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize MCP server
        mcp_server = MCPServer(digipal_core, "demo-server")
        
        # Create demo user
        demo_user = "demo_user"
        storage_manager.create_user(demo_user, "Demo User")
        mcp_server.authenticate_user(demo_user)
        
        print(f"✓ Demo user '{demo_user}' created and authenticated")
        
        while True:
            print("\nAvailable MCP Tools:")
            print("1. create_new_pet - Create a new DigiPal")
            print("2. get_pet_status - Get pet status")
            print("3. interact_with_pet - Send message to pet")
            print("4. apply_care_action - Apply care action")
            print("5. get_pet_statistics - Get detailed statistics")
            print("6. trigger_evolution - Force evolution")
            print("7. get_available_actions - List care actions")
            print("0. Exit demo")
            
            choice = input("\nSelect tool (0-7): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await demo_create_pet(mcp_server, demo_user)
            elif choice == "2":
                await demo_get_status(mcp_server, demo_user)
            elif choice == "3":
                await demo_interact(mcp_server, demo_user)
            elif choice == "4":
                await demo_care_action(mcp_server, demo_user)
            elif choice == "5":
                await demo_statistics(mcp_server, demo_user)
            elif choice == "6":
                await demo_evolution(mcp_server, demo_user)
            elif choice == "7":
                await demo_available_actions(mcp_server, demo_user)
            else:
                print("Invalid choice. Please select 0-7.")
        
        print("\nDemo completed. Cleaning up...")
        mcp_server.shutdown()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import os
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)


async def demo_create_pet(mcp_server, user_id):
    """Demo create_new_pet tool."""
    print("\n--- Create New Pet ---")
    
    # Check if pet already exists
    existing = await mcp_server._handle_get_pet_status({"user_id": user_id})
    if not existing.isError:
        print("User already has a pet. Cannot create another.")
        return
    
    egg_type = input("Enter egg type (red/blue/green): ").strip().lower()
    if egg_type not in ["red", "blue", "green"]:
        print("Invalid egg type. Using 'red'.")
        egg_type = "red"
    
    name = input("Enter pet name (or press Enter for 'DigiPal'): ").strip()
    if not name:
        name = "DigiPal"
    
    result = await mcp_server._handle_create_new_pet({
        "user_id": user_id,
        "egg_type": egg_type,
        "name": name
    })
    
    print(f"\nResult: {result.content[0].text}")


async def demo_get_status(mcp_server, user_id):
    """Demo get_pet_status tool."""
    print("\n--- Get Pet Status ---")
    
    result = await mcp_server._handle_get_pet_status({"user_id": user_id})
    print(f"\nResult:\n{result.content[0].text}")


async def demo_interact(mcp_server, user_id):
    """Demo interact_with_pet tool."""
    print("\n--- Interact with Pet ---")
    
    message = input("Enter message to send to pet: ").strip()
    if not message:
        message = "Hello!"
    
    result = await mcp_server._handle_interact_with_pet({
        "user_id": user_id,
        "message": message
    })
    
    print(f"\nResult: {result.content[0].text}")


async def demo_care_action(mcp_server, user_id):
    """Demo apply_care_action tool."""
    print("\n--- Apply Care Action ---")
    
    # Show available actions first
    actions_result = await mcp_server._handle_get_available_actions({"user_id": user_id})
    if not actions_result.isError:
        print("Available actions:")
        print(actions_result.content[0].text)
    
    action = input("\nEnter care action: ").strip().lower()
    if not action:
        action = "meat"
    
    result = await mcp_server._handle_apply_care_action({
        "user_id": user_id,
        "action": action
    })
    
    print(f"\nResult: {result.content[0].text}")


async def demo_statistics(mcp_server, user_id):
    """Demo get_pet_statistics tool."""
    print("\n--- Get Pet Statistics ---")
    
    result = await mcp_server._handle_get_pet_statistics({"user_id": user_id})
    print(f"\nResult:\n{result.content[0].text}")


async def demo_evolution(mcp_server, user_id):
    """Demo trigger_evolution tool."""
    print("\n--- Trigger Evolution ---")
    
    force = input("Force evolution regardless of requirements? (y/n): ").strip().lower()
    force_bool = force.startswith('y')
    
    result = await mcp_server._handle_trigger_evolution({
        "user_id": user_id,
        "force": force_bool
    })
    
    print(f"\nResult: {result.content[0].text}")


async def demo_available_actions(mcp_server, user_id):
    """Demo get_available_actions tool."""
    print("\n--- Get Available Actions ---")
    
    result = await mcp_server._handle_get_available_actions({"user_id": user_id})
    print(f"\nResult:\n{result.content[0].text}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DigiPal MCP Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start MCP server with default settings
  python -m digipal.mcp.cli start
  
  # Start with custom database and assets
  python -m digipal.mcp.cli start --database /path/to/digipal.db --assets /path/to/assets
  
  # Run interactive demo
  python -m digipal.mcp.cli demo
  
  # Start with debug logging
  python -m digipal.mcp.cli start --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start server command
    start_parser = subparsers.add_parser("start", help="Start MCP server")
    start_parser.add_argument(
        "--database",
        default="digipal.db",
        help="Path to SQLite database file (default: digipal.db)"
    )
    start_parser.add_argument(
        "--assets",
        default="assets",
        help="Path to assets directory (default: assets)"
    )
    start_parser.add_argument(
        "--name",
        default="digipal-mcp-server",
        help="MCP server name (default: digipal-mcp-server)"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.command == "start":
        asyncio.run(start_server(args))
    elif args.command == "demo":
        asyncio.run(demo_mode(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()