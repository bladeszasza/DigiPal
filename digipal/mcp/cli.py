#!/usr/bin/env python3
"""
CLI script to run the DigiPal MCP server.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)  # Use stderr to avoid interfering with MCP stdio
        ]
    )


async def main():
    """Main entry point for MCP server."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize storage manager
        storage_path = os.getenv("DIGIPAL_STORAGE_PATH", "digipal_data.db")
        assets_path = os.getenv("DIGIPAL_ASSETS_PATH", "demo_assets")
        
        logger.info(f"Initializing storage: {storage_path}")
        storage_manager = StorageManager(storage_path, assets_path)
        
        # Initialize AI communication
        logger.info("Initializing AI communication")
        ai_communication = AICommunication()
        
        # Initialize DigiPal core
        logger.info("Initializing DigiPal core")
        digipal_core = DigiPalCore(storage_manager, ai_communication)
        
        # Initialize MCP server
        server_name = os.getenv("DIGIPAL_MCP_SERVER_NAME", "digipal-mcp-server")
        logger.info(f"Initializing MCP server: {server_name}")
        mcp_server = MCPServer(digipal_core, server_name)
        
        # Start the server
        logger.info("Starting MCP server...")
        await mcp_server.start_server()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)
    finally:
        if 'mcp_server' in locals():
            mcp_server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())