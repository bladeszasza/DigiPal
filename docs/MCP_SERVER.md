# DigiPal MCP Server

The DigiPal MCP (Model Context Protocol) Server provides external system integration capabilities, allowing other AI systems and tools to interact with DigiPal pets through a standardized protocol.

## Overview

The MCP server implements the Model Context Protocol specification, exposing DigiPal functionality as a set of tools that can be called by MCP clients. This enables integration with various AI systems, development tools, and automation platforms.

## Features

- **Full Pet Management**: Create, load, and manage DigiPal pets
- **Real-time Interaction**: Send messages and commands to pets
- **Care Actions**: Apply feeding, training, and care actions
- **Status Monitoring**: Get detailed pet status and statistics
- **Evolution Control**: Trigger and monitor pet evolution
- **Authentication**: User authentication and permission management
- **Error Handling**: Comprehensive error handling and validation

## Available Tools

### 1. get_pet_status
Get the current status and attributes of a user's DigiPal.

**Parameters:**
- `user_id` (string, required): User ID to get pet status for

**Returns:**
- Formatted status report with pet attributes, life stage, and current condition

### 2. interact_with_pet
Send a text message to interact with a user's DigiPal.

**Parameters:**
- `user_id` (string, required): User ID whose pet to interact with
- `message` (string, required): Message to send to the DigiPal

**Returns:**
- Pet response and any attribute changes from the interaction

### 3. apply_care_action
Apply a care action to a user's DigiPal (feeding, training, etc.).

**Parameters:**
- `user_id` (string, required): User ID whose pet to care for
- `action` (string, required): Care action to apply
  - Food: `meat`, `fish`, `vegetables`, `protein`, `vitamins`
  - Training: `strength_training`, `speed_training`, `defense_training`, `brain_training`
  - Care: `rest`, `play`, `praise`, `scold`

**Returns:**
- Action result and attribute changes

### 4. create_new_pet
Create a new DigiPal for a user.

**Parameters:**
- `user_id` (string, required): User ID to create pet for
- `egg_type` (string, required): Type of egg to create (`red`, `blue`, `green`)
- `name` (string, optional): Name for the new DigiPal (default: "DigiPal")

**Returns:**
- Success confirmation with pet ID

### 5. get_pet_statistics
Get comprehensive statistics and analysis for a user's DigiPal.

**Parameters:**
- `user_id` (string, required): User ID to get statistics for

**Returns:**
- Detailed statistics including care assessment, evolution status, personality traits, and learned commands

### 6. trigger_evolution
Manually trigger evolution for a user's DigiPal if eligible.

**Parameters:**
- `user_id` (string, required): User ID whose pet to evolve
- `force` (boolean, optional): Force evolution regardless of requirements (default: false)

**Returns:**
- Evolution result with stage changes and attribute modifications

### 7. get_available_actions
Get list of available care actions for a user's DigiPal.

**Parameters:**
- `user_id` (string, required): User ID to get available actions for

**Returns:**
- List of available care actions based on pet's current state

## Authentication and Permissions

The MCP server includes authentication and permission management:

- **User Authentication**: Users must be authenticated before accessing tools
- **Permission Control**: Fine-grained permissions for different actions
- **Session Management**: Secure session handling for authenticated users

### Development Mode
In development mode, authentication is simplified for testing purposes. All users are automatically authenticated, and all actions are permitted.

### Production Mode
In production, proper authentication integration with HuggingFace or other providers should be implemented.

## Usage Examples

### Starting the MCP Server

```bash
# Start with default settings
python -m digipal.mcp.cli start

# Start with custom database and assets
python -m digipal.mcp.cli start --database /path/to/digipal.db --assets /path/to/assets

# Start with debug logging
python -m digipal.mcp.cli start --log-level DEBUG
```

### Interactive Demo

```bash
# Run interactive demo
python -m digipal.mcp.cli demo
```

### Programmatic Usage

```python
import asyncio
from digipal.core.digipal_core import DigiPalCore
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer

async def example_usage():
    # Initialize components
    storage_manager = StorageManager("digipal.db", "assets")
    ai_communication = AICommunication()
    digipal_core = DigiPalCore(storage_manager, ai_communication)
    
    # Create MCP server
    mcp_server = MCPServer(digipal_core, "my-digipal-server")
    
    # Authenticate user
    user_id = "example_user"
    storage_manager.create_user(user_id, "Example User")
    mcp_server.authenticate_user(user_id)
    
    # Create a new pet
    result = await mcp_server._handle_create_new_pet({
        "user_id": user_id,
        "egg_type": "red",
        "name": "MyPal"
    })
    
    # Get pet status
    status = await mcp_server._handle_get_pet_status({"user_id": user_id})
    print(status.content[0].text)
    
    # Interact with pet
    interaction = await mcp_server._handle_interact_with_pet({
        "user_id": user_id,
        "message": "Hello!"
    })
    print(interaction.content[0].text)

# Run example
asyncio.run(example_usage())
```

## Integration with MCP Clients

The DigiPal MCP server can be integrated with various MCP clients:

### Claude Desktop
Add to your MCP configuration:

```json
{
  "mcpServers": {
    "digipal": {
      "command": "python",
      "args": ["-m", "digipal.mcp.cli", "start"],
      "env": {
        "DIGIPAL_DB": "/path/to/digipal.db"
      }
    }
  }
}
```

### Custom MCP Client
Use the standard MCP protocol to connect and call tools:

```python
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters

# Connect to DigiPal MCP server
server_params = StdioServerParameters(
    command="python",
    args=["-m", "digipal.mcp.cli", "start"]
)

async with ClientSession(server_params) as session:
    # List available tools
    tools = await session.list_tools()
    
    # Call a tool
    result = await session.call_tool("get_pet_status", {"user_id": "my_user"})
    print(result.content[0].text)
```

## Error Handling

The MCP server provides comprehensive error handling with automated recovery:

### Error Categories
- **Validation Errors**: Invalid parameters or missing required fields
- **Authentication Errors**: Unauthenticated users or insufficient permissions
- **Pet Not Found**: Attempts to interact with non-existent pets
- **System Errors**: Database failures, AI model issues, network problems
- **Recovery Errors**: Issues during automated recovery attempts

### Automated Recovery
The MCP server integrates with the DigiPal error handling and recovery system:

```python
from digipal.core.recovery_strategies import get_system_recovery_orchestrator

# Automatic error recovery in MCP operations
try:
    result = await mcp_operation()
except DigiPalException as e:
    orchestrator = get_system_recovery_orchestrator()
    recovery_result = orchestrator.execute_comprehensive_recovery(e)
    
    if recovery_result.success:
        # Retry operation after successful recovery
        result = await mcp_operation()
    else:
        # Return user-friendly error with recovery suggestions
        return create_error_response(e, recovery_result)
```

### Error Response Format
All errors are returned as MCP-compliant error responses with:
- **Error Code**: Standardized error classification
- **Error Message**: User-friendly description
- **Recovery Suggestions**: Actionable recommendations
- **Context Information**: Additional debugging details (in development mode)

### Graceful Degradation
The MCP server supports graceful degradation:
- **AI Model Failures**: Fallback to simple response templates
- **Network Issues**: Offline mode with cached data
- **Storage Problems**: Alternative storage locations and read-only mode
- **Authentication Issues**: Guest mode with limited functionality

## Testing

### Unit Tests
```bash
python -m pytest tests/test_mcp_server.py -v
```

### Integration Tests
```bash
python test_mcp_integration.py
```

The integration test suite validates complete MCP server functionality including:
- Pet creation and lifecycle management
- Evolution system with flexible stage progression
- Care action application and attribute changes
- Error handling and authentication
- Multi-user scenarios and data persistence

### Demo Script
```bash
python examples/mcp_server_demo.py
```

## Configuration

### Environment Variables
- `DIGIPAL_DB`: Path to SQLite database file
- `DIGIPAL_ASSETS`: Path to assets directory
- `DIGIPAL_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Server Configuration
The MCP server can be configured through the `MCPServer` constructor:

```python
mcp_server = MCPServer(
    digipal_core=digipal_core,
    server_name="custom-server-name"
)
```

## Performance Considerations

- **Caching**: Active pets are cached in memory for faster access
- **Background Updates**: Automatic pet state updates run in background
- **Model Loading**: AI models are loaded on-demand and cached
- **Database Optimization**: Efficient queries with proper indexing

## Security

- **Input Validation**: All user inputs are validated and sanitized
- **Permission Checks**: Actions are authorized before execution
- **Error Messages**: Secure error messages that don't expose sensitive data
- **Session Management**: Secure session handling and cleanup

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure database file is accessible and writable
   - Check file permissions and disk space

2. **AI Model Loading Failures**
   - Verify model dependencies are installed
   - Check available memory and GPU resources

3. **Authentication Issues**
   - Verify user exists in database
   - Check authentication token validity

4. **Pet Not Found Errors**
   - Ensure pet was created successfully
   - Check user ID matches exactly

### Debug Mode
Enable debug logging for detailed troubleshooting:

```bash
python -m digipal.mcp.cli start --log-level DEBUG
```

## Contributing

When contributing to the MCP server:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for API changes
4. Ensure MCP protocol compliance
5. Test with real MCP clients when possible

## License

The DigiPal MCP Server is part of the DigiPal project and follows the same licensing terms.