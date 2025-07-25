# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_auth_*.py -v          # Authentication tests
python -m pytest tests/test_integration_*.py -v   # Integration tests
python -m pytest tests/test_models.py -v          # Core model tests

# Run with performance output
python -m pytest tests/test_auth_integration.py::TestAuthenticationPerformance -v -s
```

### Running the Application
```bash
# Launch main Gradio interface
python launch_digipal.py

# Run specific demos
python examples/digipal_core_demo.py
python examples/gradio_interface_demo.py
python examples/mcp_server_demo.py
```

### Testing MCP Integration
```bash
python test_mcp_integration.py
```

## High-Level Architecture

DigiPal is a sophisticated digital pet application with dual interfaces: a Gradio web UI and an MCP (Model Context Protocol) server. The architecture follows a layered approach:

### Core Engine (`digipal/core/`)
- **DigiPalCore**: Central orchestrator managing all pet operations, real-time updates, and background processing
- **Models**: Core data structures (DigiPal, Interaction, Command) with full lifecycle management
- **AttributeEngine**: Digimon World 1-inspired care mechanics with training/feeding/care actions
- **EvolutionController**: Manages 7 life stages (Egg → Elderly) with generational inheritance system

### AI Layer (`digipal/ai/`)
- **AICommunication**: Orchestrates natural language processing and speech recognition
- **LanguageModel**: Qwen3-0.6B integration for contextual conversations
- **SpeechProcessor**: Kyutai speech-to-text integration 
- **ImageGenerator**: FLUX.1-dev diffusion model for dynamic pet visualization with intelligent caching

### Storage Layer (`digipal/storage/`)
- **StorageManager**: High-level data operations with backup/recovery
- **Database**: SQLite schema with proper migrations and indexing

### Authentication (`digipal/auth/`)
- **AuthManager**: HuggingFace API integration with offline fallback
- **SessionManager**: Secure token storage and session validation

### Interfaces
- **Gradio UI** (`digipal/ui/`): Three-tab interface (auth → egg selection → main interaction)
- **MCP Server** (`digipal/mcp/`): Protocol-compliant server for AI system integration

## Key Development Patterns

### Pet State Management
The DigiPalCore uses an active pet cache with lazy loading. Pets are automatically persisted and have background updates every 60 seconds for time-based attribute decay and evolution monitoring.

### Error Handling Philosophy  
The system implements graceful degradation throughout - AI models can fail, image generation has fallbacks, and offline modes are supported for development.

### Real-Time Updates
Background threading handles automatic pet state updates, evolution progression, and death/inheritance processing without blocking the main interface.

### Generational System
When pets reach end-of-life, they create DNA for inheritance. Care quality determines inheritance rates (Perfect: 25%, Excellent: 20%, Good: 15%, Fair: 10%, Poor: 5%).

## Current Implementation Status

**Completed (Tasks 1-14)**: Core engine, AI integration, storage, authentication, Gradio interface, MCP server, image generation, and comprehensive testing.

**In Progress (Tasks 15-18)**: Error handling improvements, memory management optimization, expanded test coverage, and deployment configuration.

## Testing Approach

The test suite includes unit tests, integration tests, and performance benchmarks. Authentication tests cover both online (HuggingFace API) and offline modes with network resilience scenarios. All AI model dependencies are properly mocked for reliable testing.

## Important Files

- `launch_digipal.py`: Main application entry point
- `digipal/core/digipal_core.py`: Central engine orchestrator  
- `digipal/core/models.py`: Core data structures
- `.kiro/specs/digipal-mcp-server/tasks.md`: Detailed implementation roadmap
- `examples/`: Working demos of all major components