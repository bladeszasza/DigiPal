# DigiPal

**Version:** 0.1.0  
**Author:** DigiPal Team

DigiPal is a sophisticated digital pet application that combines modern AI technologies with classic virtual pet mechanics. Built as an entry for the [Code with Kiro Hackathon](https://kiro.devpost.com/?ref_feature=challenge&ref_medium=homepage-recommended-hackathons), DigiPal serves dual purposes: providing an engaging Gradio web interface for users to interact with their digital companions, and functioning as an MCP (Model Context Protocol) server for integration with other AI systems.

## 🎮 Features

### Core Digital Pet Experience
- **Egg Selection**: Choose from three egg types (Red, Blue, Green) with unique attribute bonuses
- **Life Stages**: Watch your DigiPal evolve through 7 life stages: Egg → Baby → Child → Teen → Young Adult → Adult → Elderly
- **Attribute System**: Digimon World 1-inspired attributes including HP, MP, Offense, Defense, Speed, and Brains
- **Care Mechanics**: Comprehensive training, feeding, and care system with 13+ different actions
- **Generational Inheritance**: Pass traits to new generations when your DigiPal reaches the end of its lifecycle

#### 🏋️ Training Actions
- **Strength Training**: Increases Offense (+3), HP (+2), reduces Weight (-1)
- **Defense Training**: Increases Defense (+3), HP (+2), reduces Weight (-1)  
- **Speed Training**: Increases Speed (+3), reduces Weight (-2)
- **Brain Training**: Increases Brains (+3), MP (+2)
- **Endurance Training**: Increases HP (+4), Defense (+2), reduces Weight (-2)
- **Agility Training**: Increases Speed (+4), Offense (+1), reduces Weight (-3)

#### 🍽️ Feeding Actions
- **Meat**: Increases Weight (+2), HP (+1), Offense (+1), Happiness (+5)
- **Fish**: Increases Weight (+1), Brains (+1), MP (+1), Happiness (+3)
- **Vegetables**: Increases Weight (+1), Defense (+1), Happiness (+2)
- **Protein Shake**: Increases Weight (+1), Offense (+2), HP (+1), Happiness (+1)
- **Energy Drink**: Restores Energy (+5), increases Speed (+1), MP (+1), Happiness (+3)

#### 💝 Care Actions
- **Praise**: Increases Happiness (+10), reduces Discipline (-2)
- **Scold**: Reduces Happiness (-8), increases Discipline (+5)
- **Rest**: Restores Energy (+30), increases Happiness (+3)
- **Play**: Increases Happiness (+8), reduces Weight (-1), costs Energy (-8)

#### 🦋 Evolution System
- **Time-Based Evolution**: Automatic progression through life stages based on age
- **Requirement-Based Evolution**: Meet specific attribute, care, and happiness thresholds
- **Stage Progression**: Egg (30min) → Baby (1 day) → Child (3 days) → Teen (5 days) → Young Adult (7 days) → Adult (10 days) → Elderly (3 days)
- **Evolution Bonuses**: Each stage grants significant attribute increases and new command understanding
- **Egg Type Specialization**: Red eggs gain offense bonuses, Blue eggs gain defense/MP bonuses, Green eggs gain HP bonuses
- **Care Quality Impact**: Better care leads to easier evolution and stronger attribute gains

#### 🧬 Generational Inheritance
- **DNA System**: Parent attributes and care quality determine offspring bonuses
- **Inheritance Rates**: Perfect care (25%), Excellent (20%), Good (15%), Fair (10%), Poor (5%)
- **Attribute Passing**: High-performing parents pass stat bonuses to next generation
- **Randomization**: Small random variations prevent identical offspring
- **Generation Tracking**: Each DigiPal knows its generation number and lineage

### AI-Powered Interaction
- **Natural Language Communication**: Powered by Qwen3-0.6B for contextual conversations
- **Speech Recognition**: Kyutai integration for voice-based interaction
- **Memory System**: Persistent conversation history and learned behaviors
- **Stage-Appropriate Commands**: Command understanding evolves with your pet's life stage
- **Command Interpretation**: Advanced regex-based command parsing with parameter extraction
- **Contextual Responses**: Life stage-specific response templates with personality-based selection
- **Conversation Memory**: Automatic interaction tracking with personality trait evolution

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
│   ├── enums.py            # EggType, LifeStage, AttributeType enums
│   ├── attribute_engine.py # Care mechanics and attribute management
│   └── evolution_controller.py # Evolution system and generational inheritance
├── ai/                     # AI communication layer
│   └── __init__.py
├── mcp/                    # MCP server implementation
│   └── __init__.py
├── storage/                # Data persistence layer
│   ├── __init__.py
│   ├── database.py         # SQLite schema and connection management
│   └── storage_manager.py  # High-level storage operations
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

### ✅ Completed (Tasks 1-4)
- **Core Data Models**: Complete DigiPal model with all attributes and lifecycle properties
- **Enum System**: EggType, LifeStage, AttributeType, and other constants
- **Serialization**: Full JSON serialization/deserialization support
- **Storage Layer**: SQLite database with full CRUD operations, backup/recovery system
- **Attribute Engine**: Complete Digimon World 1-inspired care mechanics with bounds checking
- **Care Actions**: Full training, feeding, and care action system
- **Evolution System**: Complete lifecycle management with time-based and requirement-based evolution
- **Generational Inheritance**: DNA-based attribute passing between generations
- **Unit Tests**: Comprehensive test coverage for all implemented components

### 🔄 In Progress
Following the [implementation roadmap](.kiro/specs/digipal-mcp-server/tasks.md), the next phases include:
- AI communication layer foundation
- Qwen3-0.6B and Kyutai integration
- DigiPal core engine orchestration
- Image generation system for pet visualization

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
- **Core Models**: DigiPal model initialization and attribute management
- **Egg Types**: Attribute bonuses and initialization behavior
- **Attribute System**: Bounds checking, modification, and validation
- **Life Stages**: Command understanding progression and evolution mechanics
- **Data Persistence**: Serialization/deserialization and storage operations
- **AI Integration**: Language model integration and response generation
- **Care Mechanics**: Training, feeding, and care action effects
- **Evolution System**: Life stage progression and inheritance mechanics

The test suite includes comprehensive unit tests with proper mocking for external dependencies like the Qwen3-0.6B language model and Kyutai speech processor, ensuring reliable testing without requiring actual model downloads. Tests include robust edge case handling and confidence threshold validation for speech processing components.

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