"""
MCP (Model Context Protocol) server implementation for DigiPal.

This module provides MCP server functionality for external system integration
with DigiPal pets, allowing other AI systems to interact with and manage
digital pets through the Model Context Protocol.
"""

from .server import MCPServer

__all__ = ['MCPServer']