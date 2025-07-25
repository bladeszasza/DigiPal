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

### Visual Generation System
- **FLUX.1-dev Integration**: Professional image generation using state-of-the-art diffusion models
- **Dynamic Visualization**: Images automatically update as your DigiPal evolves through life stages
- **Attribute-Based Appearance**: Pet appearance reflects current stats (offense, defense, happiness, etc.)
- **Egg Type Specialization**: Visual traits match elemental affinities (fire, water, earth)
- **Intelligent Caching**: Generated images cached to optimize performance and reduce API calls
- **Fallback System**: Graceful degradation with placeholder images when generation fails
- **Professional Prompts**: Sophisticated prompt engineering for consistent, high-quality results

### Enhanced Web Interface
- **Three-Tab Navigation**: Seamless flow from authentication → egg selection → main interface
- **Real-Time Status Display**: Live pet status updates with visual feedback and needs assessment
- **Advanced Care Controls**: Comprehensive training options with specialized sub-actions
- **Interactive Communication**: Enhanced speech and text interaction with quick message buttons
- **Visual Feedback System**: Action feedback, needs alerts, and auto-refresh capabilities
- **Conversation Management**: Live conversation history with export and clear functions
- **Professional UI Design**: Game-style interface with responsive layout and animations
- **Complete Implementation**: Fully functional Gradio interface with all core features

#### 🎮 Enhanced Care Interface
The Gradio interface now features a comprehensive care system with multiple interaction methods:

**Primary Care Actions**:
- **Feed** (🍖): Basic feeding with visual feedback
- **Train** (💪): General training with expandable sub-options
- **Praise** (👍): Positive reinforcement for good behavior
- **Scold** (👎): Discipline for care mistakes
- **Rest** (😴): Energy restoration and relaxation
- **Play** (🎾): Interactive play sessions

**Advanced Training Options** (Expandable Accordion):
- **Strength Training** (🏋️): Focused offense and HP development
- **Speed Training** (🏃): Agility and speed enhancement
- **Brain Training** (🧠): Intelligence and MP development
- **Defense Training** (🛡️): Defensive capabilities improvement

**Advanced Care Options** (Expandable Accordion):
- **Medicine** (💊): Health restoration and status effect treatment
- **Clean** (🧼): Hygiene maintenance and happiness boost

#### 🎤 Enhanced Communication System
**Voice Interaction Panel**:
- **Audio Recording**: Click-to-record voice input with visual feedback
- **Processing Status**: Real-time audio processing status display
- **Process Speech Button**: Manual trigger for speech-to-text conversion

**Text Chat Panel**:
- **Message Input**: Multi-line text input with placeholder guidance
- **Quick Messages**: Pre-defined interaction buttons
  - **Hello** (👋): Quick greeting
  - **How are you?** (❓): Status inquiry
- **Send Button**: Primary action for text message submission

#### 📊 Real-Time Status Dashboard
**Pet Display Area**:
- **Pet Image**: Dynamic visual representation of your DigiPal
- **Pet Name**: Personalized display with current pet name
- **Action Feedback**: Visual confirmation of performed actions

**Status Information**:
- **Real-Time Updates**: Live pet status with automatic refresh
- **Attribute Display**: Detailed attribute bars with visual indicators
- **Needs Assessment**: Alert system for pet attention requirements
- **Auto-Refresh Toggle**: Optional 30-second automatic status updates

#### 💬 Conversation Management
**Response Display**:
- **DigiPal Responses**: Real-time pet communication feedback
- **Live History**: Always-visible conversation timeline
- **Conversation Controls**:
  - **Clear History** (🗑️): Reset conversation memory
  - **Export Chat** (📥): Download conversation log

#### 🎨 Visual Design Features
**Professional UI Elements**:
- **Game-Style Aesthetics**: Custom CSS with gaming-inspired design
- **Responsive Layout**: Two-column layout optimizing space usage
- **Visual Feedback**: Action confirmations and status animations
- **Accordion Panels**: Expandable sections for advanced features
- **Consistent Theming**: Unified color scheme and typography

### Technical Integration
- **DigiPal Core Engine**: Central orchestrator managing all pet operations and real-time updates
- **MCP Server**: Full Model Context Protocol compliance for AI system integration
- **HuggingFace Authentication**: Secure user authentication and progress saving
- **Enhanced Gradio Interface**: Feature-rich web UI with real-time updates and advanced controls
- **Persistent Storage**: SQLite-based data persistence with backup systems
- **Background Processing**: Automatic time-based updates and evolution monitoring
- **Error Handling & Recovery**: Comprehensive error management with automated recovery strategies

## 🎛️ DigiPal Core Engine

The `DigiPalCore` class serves as the central orchestrator for all DigiPal functionality, providing a unified interface for pet management, interaction processing, and real-time updates.

### Key Components

#### PetState
Represents the current state of a DigiPal for external systems:
- **Attributes**: All pet statistics (HP, MP, Offense, Defense, Speed, Brains, etc.)
- **Status**: Age, last interaction, evolution readiness, attention needs
- **Derived Metrics**: Status summary, needs assessment, evolution eligibility

#### InteractionProcessor
Handles user interactions through the AI communication layer:
- **Text Processing**: Natural language interpretation and response generation
- **Speech Processing**: Audio-to-text conversion with Kyutai integration
- **Command Effects**: Automatic application of care actions based on interpreted commands
- **Special Handling**: Egg hatching triggers and lifecycle events

### Core Engine API

#### Pet Management
```python
# Create new DigiPal
pet = core.create_new_pet(EggType.RED, user_id="user123", name="MyPal")

# Load existing pet
pet = core.load_existing_pet(user_id="user123")

# Get current pet state
state = core.get_pet_state(user_id="user123")
```

#### Interaction Processing
```python
# Process text interaction
success, interaction = core.process_interaction(user_id, "Let's train!")

# Process speech interaction
success, interaction = core.process_speech_interaction(user_id, audio_data)

# Apply direct care action
success, interaction = core.apply_care_action(user_id, "strength_training")
```

#### Evolution Management
```python
# Manual evolution trigger
success, result = core.trigger_evolution(user_id, force=False)

# Check evolution eligibility
eligible, next_stage, requirements = evolution_controller.check_evolution_eligibility(pet)
```

#### Background Processing
```python
# Start automatic updates
core.start_background_updates()

# Manual state update
core.update_pet_state(user_id, force_save=True)

# Stop background processing
core.stop_background_updates()
```

#### Statistics and Analytics
```python
# Get comprehensive pet statistics
stats = core.get_pet_statistics(user_id)

# Get available care actions
actions = core.get_care_actions(user_id)
```

### Real-Time Features

#### Automatic Updates
- **Time-Based Decay**: Energy and happiness naturally decrease over time
- **Evolution Monitoring**: Automatic progression through life stages
- **Death Handling**: End-of-life processing with inheritance preparation
- **Background Thread**: Non-blocking updates every 60 seconds

#### State Management
- **Active Pet Cache**: In-memory storage for frequently accessed pets
- **Lazy Loading**: Pets loaded from storage only when needed
- **Automatic Persistence**: Changes saved to database automatically
- **Memory Optimization**: Efficient caching with cleanup on shutdown

#### Event Handling
- **Egg Hatching**: First interaction triggers evolution to baby stage
- **Evolution Events**: Automatic attribute bonuses and command learning
- **Death Events**: Generational inheritance DNA creation
- **Care Mistakes**: Automatic tracking of poor care decisions

### Integration Points

The DigiPalCore integrates seamlessly with:
- **Storage Manager**: Persistent data operations
- **AI Communication**: Natural language and speech processing
- **Attribute Engine**: Care mechanics and attribute management
- **Evolution Controller**: Lifecycle and inheritance management

This architecture provides a robust foundation for both the Gradio web interface and MCP server implementations, ensuring consistent behavior across all interaction methods.

## 🎨 Image Generation System

The `ImageGenerator` class provides sophisticated visual representation for DigiPals using the FLUX.1-dev diffusion model, creating dynamic images that reflect each pet's unique characteristics and evolution.

### Key Features

#### Professional Prompt Engineering
- **Life Stage Templates**: Specialized prompts for each evolution stage (egg, baby, child, teen, young adult, adult, elderly)
- **Egg Type Specialization**: Elemental themes (fire/red, water/blue, earth/green) with appropriate colors and environments
- **Attribute Integration**: Pet stats influence visual appearance (high offense = fierce features, high defense = armored look)
- **Personality Reflection**: Happiness and other traits affect facial expressions and poses

#### Intelligent Caching System
- **MD5-Based Keys**: Generated images cached using prompt and parameter hashes
- **Performance Optimization**: Avoids redundant generation for identical configurations
- **Cache Management**: Automatic cleanup of old cached images (configurable age limit)
- **Storage Efficiency**: Organized cache directory structure with metadata tracking

#### Robust Fallback System
- **Graceful Degradation**: Automatic fallback to placeholder images when generation fails
- **Staged Fallbacks**: Life stage and egg type specific placeholders → generic placeholder → simple colored rectangles
- **Error Handling**: Comprehensive exception handling with detailed logging
- **Offline Support**: System remains functional without internet connectivity

### Image Generation API

#### Basic Generation
```python
# Initialize image generator
generator = ImageGenerator(
    model_name="black-forest-labs/FLUX.1-dev",
    cache_dir="demo_assets/images",
    fallback_dir="demo_assets/images/fallbacks"
)

# Generate image for DigiPal
image_path = generator.generate_image(digipal, force_regenerate=False)

# Update image after evolution
new_image_path = generator.update_image_for_evolution(digipal)
```

#### Advanced Configuration
```python
# Custom generation parameters
generator.generation_params = {
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512
}

# Generate professional prompt
prompt = generator.generate_prompt(digipal)

# Cache management
generator.cleanup_cache(max_age_days=30)
cache_info = generator.get_cache_info()
```

### Prompt Template System

#### Life Stage Characteristics
Each life stage has specific visual traits:
- **Egg**: Mystical egg with glowing patterns and magical runes
- **Baby**: Small cute creature with big eyes and soft features
- **Child**: Young creature with curious eyes and energetic poses
- **Teen**: Adolescent with developing strength and confident stance
- **Young Adult**: Athletic build with determined expression at full power
- **Adult**: Mature, powerful creature with commanding presence
- **Elderly**: Ancient wise creature with dignified, mystical aura

#### Egg Type Specializations
Visual themes based on elemental affinities:
- **Red (Fire)**: Fierce, energetic, blazing aura with volcanic environments
- **Blue (Water)**: Calm, protective, flowing aura with aquatic environments  
- **Green (Earth)**: Sturdy, wise, natural aura with forest environments

#### Attribute-Based Modifiers
Pet statistics influence visual appearance:
- **High Offense (>50)**: Fierce expression, sharp features
- **High Defense (>50)**: Armored appearance, protective stance
- **High Speed (>50)**: Sleek, agile build
- **High Brains (>50)**: Intelligent eyes, wise demeanor
- **High Happiness (>70)**: Happy, cheerful expression
- **Low Happiness (<30)**: Sad, tired expression

### Technical Implementation

#### Model Integration
- **FLUX.1-dev**: State-of-the-art diffusion model for high-quality image generation
- **CPU Offloading**: Memory-efficient model loading with automatic VRAM management
- **Consistent Seeds**: Deterministic generation based on DigiPal ID for reproducible results
- **Torch Integration**: Optimized PyTorch backend with proper device management

#### File Management
- **Organized Structure**: Separate directories for cache, fallbacks, and user assets
- **Path Resolution**: Robust path handling across different operating systems
- **Asset Tracking**: Integration with storage manager for persistent image references
- **Cleanup Utilities**: Automated maintenance of cache directories

#### Error Handling
- **Graceful Failures**: System continues functioning even when image generation fails
- **Detailed Logging**: Comprehensive error reporting for debugging and monitoring
- **Dependency Checks**: Proper handling of missing dependencies (diffusers, torch, PIL)
- **Resource Management**: Automatic cleanup of GPU memory and file handles

### Integration with Core Systems

#### DigiPal Core Engine
- **Automatic Updates**: Images regenerated when pets evolve to new life stages
- **State Synchronization**: Image paths stored in DigiPal model for persistence
- **Background Processing**: Image generation can be triggered by evolution events

#### Storage Manager
- **Asset Management**: Generated images stored and tracked in user asset directories
- **Backup Integration**: Image references included in backup and restore operations
- **Database Storage**: Image paths and generation prompts persisted in DigiPal records

This visual system enhances the DigiPal experience by providing dynamic, personalized imagery that evolves with each pet's unique journey and characteristics.

## 🛡️ Error Handling and Recovery System

The DigiPal system includes a comprehensive error handling and recovery framework designed to provide robust operation and graceful degradation when issues occur.

### Key Components

#### Exception Hierarchy
- **DigiPalException**: Base exception with severity levels and recovery suggestions
- **Specialized Exceptions**: StorageError, AIModelError, NetworkError, AuthenticationError, PetLifecycleError, ImageGenerationError, SpeechProcessingError, MCPProtocolError
- **Error Context**: Rich context information for debugging and recovery

#### Recovery Strategies
- **Storage Recovery**: Database corruption recovery, disk space cleanup, permission error handling
- **AI Model Recovery**: Memory cleanup, fallback response modes, model loading optimization
- **Network Recovery**: Offline mode activation, DNS failover, rate limiting management
- **Authentication Recovery**: Token refresh, guest mode activation, offline authentication
- **Pet Lifecycle Recovery**: Data restoration from backups, evolution failure handling

#### System Recovery Orchestrator
- **Comprehensive Recovery**: Coordinates recovery across all system components
- **Pre-Recovery Backups**: Automatic backup creation before critical recovery operations
- **Recovery Recommendations**: Intelligent suggestions for manual intervention when needed
- **Performance Monitoring**: Recovery success rates and performance impact tracking

### Error Handling API

#### Basic Error Handling
```python
from digipal.core.error_handler import error_handler

@error_handler.handle_errors
def risky_operation():
    # Operation that might fail
    pass

# Manual error handling
try:
    risky_operation()
except DigiPalException as e:
    print(f"Error: {e.message}")
    print(f"Suggestions: {e.recovery_suggestions}")
```

#### Recovery System Usage
```python
from digipal.core.recovery_strategies import (
    initialize_system_recovery,
    get_system_recovery_orchestrator
)

# Initialize recovery system
initialize_system_recovery(backup_manager)

# Execute recovery
orchestrator = get_system_recovery_orchestrator()
result = orchestrator.execute_comprehensive_recovery(error)

if result.success:
    print(f"Recovery successful: {result.message}")
else:
    recommendations = orchestrator.get_recovery_recommendations(error)
    for rec in recommendations:
        print(f"- {rec}")
```

### Graceful Degradation

The system provides multiple levels of graceful degradation:

#### AI Model Degradation
1. **Full AI**: Complete language model and speech processing
2. **Basic AI**: Simple response templates with limited processing
3. **Static Responses**: Pre-defined responses based on pet state
4. **Minimal Mode**: Basic pet status updates only

#### Network Degradation
1. **Online Mode**: Full cloud service integration
2. **Cached Mode**: Use cached responses and data
3. **Offline Mode**: Local-only operation
4. **Emergency Mode**: Core functionality only

#### Storage Degradation
1. **Full Storage**: Complete database functionality
2. **Backup Storage**: Alternative storage locations
3. **Memory Storage**: In-memory temporary storage
4. **Read-Only Mode**: Status viewing only

### Error Integration

The error system integrates seamlessly with all DigiPal components:
- **Core Integration**: Automatic error handling in pet operations
- **UI Integration**: User-friendly error messages and recovery progress
- **MCP Integration**: Protocol-compliant error responses
- **Background Processing**: Error handling in automatic updates

For detailed information, see [Error Handling Documentation](docs/ERROR_HANDLING.md).

## 🏗️ Project Structure

```
digipal/
├── __init__.py              # Package initialization (v0.1.0)
├── core/                    # Core game logic and data models
│   ├── __init__.py
│   ├── models.py           # DigiPal, Interaction, Command data models
│   ├── enums.py            # EggType, LifeStage, AttributeType enums
│   ├── digipal_core.py     # Central orchestrator and main engine
│   ├── attribute_engine.py # Care mechanics and attribute management
│   └── evolution_controller.py # Evolution system and generational inheritance
├── ai/                     # AI communication layer
│   ├── __init__.py
│   ├── communication.py    # AI communication orchestration
│   ├── language_model.py   # Qwen3-0.6B integration
│   ├── speech_processor.py # Kyutai speech recognition
│   └── image_generator.py  # FLUX.1-dev image generation system
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

### ✅ Completed (Tasks 1-16)
- **Core Data Models**: Complete DigiPal model with all attributes and lifecycle properties
- **Enum System**: EggType, LifeStage, AttributeType, and other constants
- **Serialization**: Full JSON serialization/deserialization support
- **Storage Layer**: SQLite database with full CRUD operations, backup/recovery system
- **Attribute Engine**: Complete Digimon World 1-inspired care mechanics with bounds checking
- **Care Actions**: Full training, feeding, and care action system
- **Evolution System**: Complete lifecycle management with time-based and requirement-based evolution
- **Generational Inheritance**: DNA-based attribute passing between generations
- **AI Communication Layer**: Complete Qwen3-0.6B and Kyutai integration with contextual responses
- **DigiPal Core Engine**: Central orchestrator with real-time updates and background processing
- **Image Generation System**: FLUX.1-dev integration with intelligent caching and fallback systems
- **MCP Server**: Full Model Context Protocol implementation with external system integration
- **HuggingFace Authentication**: Complete authentication system with offline mode support
- **Gradio Web Interface**: Feature-rich web UI with real-time updates and advanced controls
- **Error Handling & Recovery**: Comprehensive error management with automated recovery strategies
- **Memory Management & Performance**: Optimized memory usage with conversation memory and RAG system
- **Unit Tests**: Comprehensive test coverage for all implemented components

### 🔄 In Progress
Following the [implementation roadmap](.kiro/specs/digipal-mcp-server/tasks.md), the next phases include:
- Complete Gradio web interface implementation (UI components completed, event handling refined)
- MCP server development and integration
- Production deployment and optimization

### 🐛 Recent Fixes
- **Gradio Interface**: Fixed egg selection event handler parameter passing to ensure proper user state management
- **Care Action Handlers**: Fixed primary care action event handlers to properly pass user state parameters, ensuring reliable care action execution

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

### Test Coverage

#### Core System Tests
- **Core Models**: DigiPal model initialization and attribute management
- **Egg Types**: Attribute bonuses and initialization behavior
- **Attribute System**: Bounds checking, modification, and validation
- **Life Stages**: Command understanding progression and evolution mechanics
- **Data Persistence**: Serialization/deserialization and storage operations
- **AI Integration**: Language model integration and response generation
- **Care Mechanics**: Training, feeding, and care action effects
- **Evolution System**: Life stage progression and inheritance mechanics

#### Authentication System Tests
- **Unit Tests**: Individual component testing for AuthManager, SessionManager, and auth models
- **Integration Tests**: Complete authentication workflows from login to logout
- **Performance Tests**: Concurrent authentication and session validation benchmarks
- **Error Handling**: Comprehensive error scenario testing and recovery validation

#### Authentication Test Scenarios
The authentication integration tests cover:

**Online Authentication Flow**:
- HuggingFace API integration with token validation
- Session creation and persistence across manager instances
- Profile refresh and session extension
- Proper logout and session invalidation

**Offline Authentication Flow**:
- Development mode authentication with deterministic user creation
- Offline session persistence and validation
- Cross-instance session recovery
- Extended session management for offline development

**Network Resilience**:
- Automatic fallback from online to offline mode during network issues
- Cached session recovery after network failures
- Graceful degradation with maintained functionality

**Multi-User Support**:
- Concurrent authentication of multiple users
- Independent session management and validation
- Selective logout with other sessions remaining active
- Session cleanup and maintenance

**Performance Benchmarks**:
- 50+ concurrent authentications in under 5 seconds
- Session validation at 25+ operations per second
- Memory-efficient session caching and persistence

**Error Scenarios**:
- Invalid token handling (too short, malformed)
- Non-existent user and session management
- Database connection failures and recovery
- Session expiration and automatic cleanup

### Running Specific Test Suites

```bash
# Run all tests
python -m pytest tests/ -v

# Run only authentication tests
python -m pytest tests/test_auth_*.py -v

# Run integration tests specifically
python -m pytest tests/test_auth_integration.py -v

# Run with performance output
python -m pytest tests/test_auth_integration.py::TestAuthenticationPerformance -v -s

# Run core system tests
python -m pytest tests/test_models.py tests/test_digipal_core.py -v
```

The test suite includes comprehensive unit and integration tests with proper mocking for external dependencies like the Qwen3-0.6B language model and Kyutai speech processor, ensuring reliable testing without requiring actual model downloads. Tests include robust edge case handling, performance benchmarks, and confidence threshold validation for all system components.

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