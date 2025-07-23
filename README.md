# DigiPal

**Version:** 0.1.0  
**Author:** DigiPal Team

DigiPal is a sophisticated digital pet application that combines modern AI technologies with classic virtual pet mechanics. Built as an entry for the [Code with Kiro Hackathon](https://kiro.devpost.com/?ref_feature=challenge&ref_medium=homepage-recommended-hackathons), DigiPal serves dual purposes: providing an engaging Gradio web interface for users to interact with their digital companions, and functioning as an MCP (Model Context Protocol) server for integration with other AI systems.

## 🎮 Features

### Core Digital Pet Experience
- **Egg Selection**: Choose from three egg types (Red, Blue, Green) with unique attribute bonuses
- **Life Stages**: Watch your DigiPal evolve through 7 life stages: Egg → Baby → Child → Teen → Young Adult → Adult → Elderly
- **Attribute System**: Digimon World 1-inspired attributes including HP, MP, Offense, Defense, Speed, and Brains
- **Care Mechanics**: Train, feed, praise, scold, and rest to influence your pet's development
- **Generational Inheritance**: Pass traits to new generations when your DigiPal reaches the end of its lifecycle

### AI-Powered Interaction
- **Natural Language Communication**: Powered by Qwen3-0.6B for contextual conversations
- **Speech Recognition**: Kyutai integration for voice-based interaction
- **Memory System**: Persistent conversation history and learned behaviors
- **Stage-Appropriate Commands**: Command understanding evolves with your pet's life stage

### Technical Integration
- **MCP Server**: Full Model Context Protocol compliance for AI system integration
- **HuggingFace Authentication**: Secure user authentication and progress saving
- **Gradio Web Interface**: Game-style UI for intuitive interaction
- **Persistent Storage**: SQLite-based data persistence with backup systems

## 🏗️ Project Structure

```
digipal/
├── __init__.py              # Package initialization (v0.1.0)
├── core/                    # Core game logic and data models
│   ├── __init__.py
│   ├── models.py           # DigiPal, Interaction, Command data models
│   └── enums.py            # EggType, LifeStage, AttributeType enums
├── ai/                     # AI communication layer
│   └── __init__.py
├── mcp/                    # MCP server implementation
│   └── __init__.py
├── storage/                # Data persistence layer
│   └── __init__.py
└── ui/                     # Gradio interface components
    └── __init__.py

tests/                      # Comprehensive test suite
├── __init__.py
└── test_models.py          # Unit tests for core models

.kiro/                      # Kiro IDE configuration
└── specs/                  # Project specifications
    └── digipal-mcp-server/
        ├── requirements.md  # Detailed requirements
        ├── design.md       # Architecture and design
        └── tasks.md        # Implementation roadmap
```

## 🚀 Current Implementation Status

### ✅ Completed (Task 1)
- **Core Data Models**: Complete DigiPal model with all attributes and lifecycle properties
- **Enum System**: EggType, LifeStage, AttributeType, and other constants
- **Serialization**: Full JSON serialization/deserialization support
- **Unit Tests**: Comprehensive test coverage for data models
- **Attribute System**: Bounds checking, modification, and egg-type initialization

### 🔄 In Progress
Following the [implementation roadmap](.kiro/specs/digipal-mcp-server/tasks.md), the next phases include:
- Storage and persistence layer (SQLite integration)
- Core attribute system and care mechanics
- Evolution and lifecycle management
- AI communication layer foundation

## 📋 Requirements

The project implements 11 core requirements covering:
1. **Authentication**: HuggingFace integration for user management
2. **Pet Loading**: Automatic restoration of existing DigiPals
3. **Egg Selection**: Three egg types with unique attribute bonuses
4. **Hatching**: Speech-triggered hatching experience
5. **Communication**: Natural language and speech interaction
6. **Evolution**: Progressive life stage development
7. **Care Actions**: Training, feeding, and care mechanics
8. **Attributes**: Persistent attribute system affecting behavior
9. **Inheritance**: Generational trait passing
10. **MCP Integration**: Full MCP server functionality
11. **Memory**: Persistent interaction history and context

## 🧪 Testing

Run the test suite to validate core functionality:

```bash
python -m pytest tests/ -v
```

Current test coverage includes:
- DigiPal model initialization and attribute management
- Egg type attribute bonuses
- Attribute bounds checking and modification
- Command understanding by life stage
- Serialization/deserialization
- Interaction, Command, and CareAction models

## 🎯 Hackathon Category: Games & Entertainment

DigiPal exemplifies the Games & Entertainment category by creating an expressive, interactive experience that combines:
- **Nostalgic Gaming**: Classic Digimon World 1 mechanics
- **Modern AI**: Cutting-edge language models and speech recognition
- **Interactive Storytelling**: Evolving pet narratives based on user care
- **Technical Innovation**: MCP server integration for AI ecosystem compatibility

## 🔮 Future Development

The project roadmap includes 18 implementation tasks covering:
- Complete AI integration (Qwen3-0.6B + Kyutai)
- Full Gradio web interface with game-style UI
- MCP server with external system integration
- Image generation for pet visualization
- Comprehensive error handling and optimization
- Production deployment configuration

## 📄 License

See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the Code with Kiro Hackathon*