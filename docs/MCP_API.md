# DigiPal MCP Server API Documentation

## Overview

The DigiPal MCP (Model Context Protocol) Server provides a standardized interface for AI systems to interact with DigiPal digital pets. This API allows external systems to query pet status, perform care actions, and integrate DigiPal functionality into larger AI workflows.

## Base Configuration

### Server Information
- **Protocol**: MCP (Model Context Protocol)
- **Version**: 1.0
- **Server Name**: digipal-mcp-server
- **Description**: Digital pet interaction and management system

### Connection Details
- **Host**: localhost (configurable)
- **Port**: 8080 (configurable)
- **Transport**: HTTP/WebSocket
- **Authentication**: Token-based (HuggingFace integration)

## Available Tools

### 1. get_pet_status

Retrieves the current status and attributes of a user's DigiPal.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user

**Returns:**
```json
{
  "pet_id": "string",
  "name": "string",
  "egg_type": "red|blue|green",
  "life_stage": "egg|baby|child|teen|young_adult|adult|elderly",
  "generation": "integer",
  "attributes": {
    "hp": "integer",
    "mp": "integer", 
    "offense": "integer",
    "defense": "integer",
    "speed": "integer",
    "brains": "integer",
    "discipline": "integer",
    "happiness": "integer",
    "weight": "integer",
    "energy": "integer"
  },
  "status": {
    "age_hours": "float",
    "last_interaction": "timestamp",
    "needs_attention": "boolean",
    "evolution_ready": "boolean",
    "care_mistakes": "integer"
  },
  "capabilities": {
    "understood_commands": ["string"],
    "available_actions": ["string"]
  }
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("get_pet_status", {"user_id": "user123"})
```

### 2. perform_care_action

Executes a care action on the user's DigiPal.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user
- `action` (string, required): Care action to perform
- `parameters` (object, optional): Additional action parameters

**Available Actions:**
- `feed` - Feed the DigiPal (parameters: `food_type`)
- `train` - Train the DigiPal (parameters: `training_type`)
- `praise` - Praise the DigiPal
- `scold` - Scold the DigiPal
- `rest` - Let the DigiPal rest
- `play` - Play with the DigiPal
- `clean` - Clean the DigiPal
- `medicine` - Give medicine to the DigiPal

**Training Types:**
- `strength_training` - Increases offense and HP
- `defense_training` - Increases defense and HP
- `speed_training` - Increases speed
- `brain_training` - Increases brains and MP
- `endurance_training` - Increases HP and defense
- `agility_training` - Increases speed and offense

**Food Types:**
- `meat` - Increases weight, HP, offense, happiness
- `fish` - Increases weight, brains, MP, happiness
- `vegetables` - Increases weight, defense, happiness
- `protein_shake` - Increases weight, offense, HP
- `energy_drink` - Restores energy, increases speed, MP

**Returns:**
```json
{
  "success": "boolean",
  "message": "string",
  "attribute_changes": {
    "hp": "integer",
    "mp": "integer",
    "offense": "integer",
    "defense": "integer",
    "speed": "integer",
    "brains": "integer",
    "discipline": "integer",
    "happiness": "integer",
    "weight": "integer",
    "energy": "integer"
  },
  "new_status": {
    "needs_attention": "boolean",
    "evolution_ready": "boolean"
  }
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("perform_care_action", {
    "user_id": "user123",
    "action": "train",
    "parameters": {"training_type": "strength_training"}
})
```

### 3. communicate_with_pet

Send a message to the DigiPal and receive a response.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user
- `message` (string, required): Message to send to the pet
- `input_type` (string, optional): Type of input ("text" or "speech", default: "text")

**Returns:**
```json
{
  "success": "boolean",
  "pet_response": "string",
  "understood_command": "string",
  "action_taken": "string",
  "attribute_changes": {
    "happiness": "integer",
    "energy": "integer"
  },
  "conversation_context": {
    "interaction_count": "integer",
    "last_topics": ["string"]
  }
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("communicate_with_pet", {
    "user_id": "user123",
    "message": "Good job! Let's train together!",
    "input_type": "text"
})
```

### 4. trigger_evolution

Manually trigger evolution if the DigiPal meets the requirements.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user
- `force` (boolean, optional): Force evolution regardless of requirements (default: false)

**Returns:**
```json
{
  "success": "boolean",
  "evolved": "boolean",
  "from_stage": "string",
  "to_stage": "string",
  "evolution_bonuses": {
    "hp": "integer",
    "mp": "integer",
    "offense": "integer",
    "defense": "integer",
    "speed": "integer",
    "brains": "integer"
  },
  "new_capabilities": ["string"],
  "message": "string"
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("trigger_evolution", {
    "user_id": "user123",
    "force": false
})
```

### 5. get_care_recommendations

Get AI-powered recommendations for caring for the DigiPal.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user

**Returns:**
```json
{
  "recommendations": [
    {
      "action": "string",
      "priority": "high|medium|low",
      "reason": "string",
      "expected_outcome": "string"
    }
  ],
  "urgent_needs": ["string"],
  "evolution_progress": {
    "current_stage": "string",
    "next_stage": "string",
    "requirements_met": "boolean",
    "missing_requirements": ["string"]
  }
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("get_care_recommendations", {
    "user_id": "user123"
})
```

### 6. get_pet_history

Retrieve interaction history and statistics for the DigiPal.

**Parameters:**
- `user_id` (string, required): Unique identifier for the user
- `limit` (integer, optional): Maximum number of interactions to return (default: 50)
- `include_stats` (boolean, optional): Include statistical summary (default: true)

**Returns:**
```json
{
  "interactions": [
    {
      "timestamp": "string",
      "type": "string",
      "action": "string",
      "success": "boolean",
      "attribute_changes": "object",
      "pet_response": "string"
    }
  ],
  "statistics": {
    "total_interactions": "integer",
    "care_actions_performed": "integer",
    "training_sessions": "integer",
    "feeding_sessions": "integer",
    "evolution_count": "integer",
    "average_happiness": "float",
    "care_quality_score": "float"
  },
  "milestones": [
    {
      "achievement": "string",
      "timestamp": "string",
      "description": "string"
    }
  ]
}
```

**Example Usage:**
```python
result = mcp_client.call_tool("get_pet_history", {
    "user_id": "user123",
    "limit": 100,
    "include_stats": true
})
```

## Error Handling

All MCP tool calls return standardized error responses when failures occur:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object"
  }
}
```

### Common Error Codes

- `USER_NOT_FOUND` - User ID does not exist
- `PET_NOT_FOUND` - User has no active DigiPal
- `INVALID_ACTION` - Requested action is not valid
- `INSUFFICIENT_ENERGY` - Pet lacks energy for the action
- `EVOLUTION_NOT_READY` - Evolution requirements not met
- `AUTHENTICATION_FAILED` - Invalid or expired authentication
- `RATE_LIMIT_EXCEEDED` - Too many requests in time window
- `INTERNAL_ERROR` - Server-side error occurred

## Authentication

The MCP server uses token-based authentication integrated with HuggingFace:

1. Obtain a HuggingFace token
2. Include the token in MCP connection headers
3. The server validates the token and creates/loads user session
4. All subsequent tool calls are associated with the authenticated user

**Connection Headers:**
```json
{
  "Authorization": "Bearer hf_your_token_here",
  "User-Agent": "your-client-name/1.0"
}
```

## Rate Limiting

The MCP server implements rate limiting to ensure fair usage:

- **Default Limit**: 60 requests per minute per user
- **Burst Limit**: 10 requests per second
- **Headers**: Rate limit information included in response headers

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## WebSocket Events

For real-time updates, the MCP server supports WebSocket connections with event streaming:

### Event Types

- `pet_status_update` - Pet attributes or status changed
- `evolution_occurred` - Pet evolved to new life stage
- `attention_needed` - Pet requires user attention
- `milestone_achieved` - Pet reached a significant milestone

### Event Format
```json
{
  "event": "string",
  "timestamp": "string",
  "user_id": "string",
  "data": "object"
}
```

## Integration Examples

### Python Client Example

```python
import asyncio
from mcp_client import MCPClient

async def main():
    client = MCPClient("ws://localhost:8080")
    await client.connect(token="hf_your_token_here")
    
    # Get pet status
    status = await client.call_tool("get_pet_status", {"user_id": "user123"})
    print(f"Pet: {status['name']} ({status['life_stage']})")
    
    # Perform care action
    result = await client.call_tool("perform_care_action", {
        "user_id": "user123",
        "action": "train",
        "parameters": {"training_type": "strength_training"}
    })
    
    if result['success']:
        print(f"Training successful! Changes: {result['attribute_changes']}")
    
    # Communicate with pet
    response = await client.call_tool("communicate_with_pet", {
        "user_id": "user123",
        "message": "How are you feeling today?"
    })
    
    print(f"Pet says: {response['pet_response']}")
    
    await client.disconnect()

asyncio.run(main())
```

### JavaScript Client Example

```javascript
const MCPClient = require('mcp-client');

async function main() {
    const client = new MCPClient('ws://localhost:8080');
    await client.connect({ token: 'hf_your_token_here' });
    
    // Get pet status
    const status = await client.callTool('get_pet_status', { user_id: 'user123' });
    console.log(`Pet: ${status.name} (${status.life_stage})`);
    
    // Subscribe to events
    client.on('pet_status_update', (event) => {
        console.log('Pet status updated:', event.data);
    });
    
    // Perform care action
    const result = await client.callTool('perform_care_action', {
        user_id: 'user123',
        action: 'feed',
        parameters: { food_type: 'meat' }
    });
    
    if (result.success) {
        console.log('Feeding successful!', result.attribute_changes);
    }
    
    await client.disconnect();
}

main().catch(console.error);
```

## Best Practices

1. **Regular Status Checks**: Poll pet status every 5-10 minutes to monitor needs
2. **Batch Operations**: Group multiple care actions when possible
3. **Error Handling**: Always handle potential errors gracefully
4. **Rate Limiting**: Respect rate limits to avoid service disruption
5. **WebSocket Events**: Use event streaming for real-time updates
6. **Authentication**: Securely store and refresh authentication tokens
7. **Logging**: Log all interactions for debugging and analytics

## Support and Troubleshooting

For issues with the MCP API:

1. Check server logs for detailed error information
2. Verify authentication token validity
3. Ensure proper parameter formatting
4. Monitor rate limiting headers
5. Test with minimal examples first

For additional support, refer to the main DigiPal documentation or contact the development team.