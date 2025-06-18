# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Second Opinion is an AI tool that provides "second opinion" functionality for AI responses via MCP (Model Context Protocol). It helps users optimize their AI model usage by comparing responses across different models, suggesting cost-effective alternatives, and tracking usage patterns.

## Implementation Guide

For detailed implementation specifications, design decisions, and development patterns, see **[IMPLEMENTATION.md](IMPLEMENTATION.md)**. This comprehensive guide contains:

- **Architecture deep-dive** with real-world implementation challenges
- **Phase-by-phase implementation plan** with detailed specifications
- **Component specifications** and interface definitions
- **Security implementation** patterns and best practices
- **Common solutions** and reusable code patterns
- **API design patterns** for consistent development
- **Testing strategies** including security and performance testing

Always refer to IMPLEMENTATION.md when:
- Starting work on a new component
- Making architectural decisions
- Implementing security features
- Adding new functionality
- Troubleshooting implementation issues

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run only security tests
uv run pytest -m security

# Run specific test file
uv run pytest tests/test_core/test_evaluator.py

# Skip slow tests
uv run pytest -m "not slow"
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Run all quality checks
uv run black . && uv run ruff check . && uv run mypy src/
```

### Running the Application
```bash
# CLI usage
uv run second-opinion --help
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" second_opinion "What's 2+2?"

# Start MCP server
uv run python -m second_opinion.mcp.server

# Development server with auto-reload
uv run python -m second_opinion.mcp.server --dev
```

## Architecture Overview

### Core Components

1. **Client System** (`src/second_opinion/clients/`)
   - Abstract base client for model interactions
   - OpenRouter client for multi-model access
   - LM Studio client for local development
   - Cost tracking and rate limiting built-in

2. **Configuration System** (`src/second_opinion/config/`)
   - YAML-based tool configurations
   - Primary model flexibility (CLI sets primary, MCP detects from client)
   - Security-focused settings management
   - Cost limits and budget controls

3. **Evaluation Engine** (`src/second_opinion/core/`)
   - Response comparison and similarity scoring
   - Model recommendation logic
   - Usage analytics and cost optimization
   - Encrypted local data storage

4. **MCP Integration** (`src/second_opinion/mcp/`)
   - FastMCP server implementation
   - Tools: `second_opinion`, `should_upgrade`, `should_downgrade`, `compare_responses`, `usage_analytics`
   - Automatic primary model detection from MCP client

5. **Prompt System** (`src/second_opinion/prompts/`)
   - Structured prompt templates in separate files
   - Parameter injection and dynamic routing
   - Tool-specific prompts with clear separation

## Security Guidelines

### Critical Security Practices

1. **Never commit API keys or secrets**
   - Use `.env` files (listed in `.gitignore`)
   - Validate environment variables on startup
   - Use descriptive names: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`

2. **Input validation and sanitization**
   - All external inputs must be validated with Pydantic models
   - Strip sensitive data before logging/storage
   - Use type hints and validation decorators

3. **Local data protection**
   - SQLite database encrypted at rest using SQLCipher
   - File permissions restricted to user only
   - Sanitize data before storage (remove potential API keys from responses)

4. **Network security**
   - HTTPS only for all API calls
   - Request timeout limits
   - Rate limiting to prevent API abuse
   - Cost limits to prevent runaway spending

### Testing Security
```bash
# Run security-focused tests
uv run pytest -m security

# Test with malicious inputs
uv run pytest tests/security/
```

## Configuration Management

### Primary Model Handling
- **CLI Mode**: User explicitly sets primary model via `--primary-model` flag
- **MCP Mode**: Primary model automatically detected from MCP client context
- **Tool Configuration**: Other models configured relative to primary in `config/model_profiles.yaml`

### Cost Controls
- Per-request cost limits in configuration
- Daily/monthly budget tracking
- Automatic fallback for expensive operations
- Cost estimation before API calls

### Model Profiles Example
```yaml
tools:
  second_opinion:
    comparison_models:
      - "openai/gpt-4o"
      - "anthropic/claude-3-5-sonnet"
    max_tokens: 1000
    temperature: 0.1
    cost_limit_per_request: 0.10
```

## Development Philosophy

### KISS Principles
- **Static over Dynamic**: Predictable configuration over runtime complexity
- **Explicit over Implicit**: Clear interfaces and obvious behavior
- **Simple over Clever**: Readable, maintainable code over premature optimization
- **Secure by Default**: Security built into every layer

### Code Organization
- **Separation of Concerns**: Clear boundaries between components
- **Dependency Injection**: Easy to test and extend
- **Type Safety**: Comprehensive type hints and validation
- **Error Handling**: Graceful failures with informative messages

### Testing Strategy
- **Security First**: Every feature tested with malicious inputs
- **Cost Awareness**: Tests verify budget limits and cost tracking
- **Integration Tests**: End-to-end MCP server testing
- **Mock External APIs**: Fast, reliable tests without API dependencies

## Common Development Tasks

### Adding a New Model Client
1. Implement `BaseClient` interface in `src/second_opinion/clients/`
2. Add configuration schema in `src/second_opinion/config/model_configs.py`
3. Register in client factory
4. Add comprehensive tests including security tests
5. Update configuration documentation

### Adding a New MCP Tool
1. Define tool schema in `src/second_opinion/mcp/tools.py`
2. Implement core logic in appropriate `src/second_opinion/core/` module
3. Add prompt templates in `prompts/templates/`
4. Add cost limits and security validation
5. Write tests for tool functionality and security
6. Update tool documentation

### Modifying Prompt Templates
1. Edit template files in `prompts/templates/`
2. Update parameter schemas if needed
3. Test with various inputs including edge cases
4. Verify no sensitive data leakage in prompts
5. Update prompt version tracking

## File Structure Notes

- **`src/second_opinion/`**: Main package code
- **`tests/`**: Test files mirroring source structure
- **`config/`**: YAML configuration files
- **`prompts/`**: Prompt template files
- **`data/`**: Local storage directory (encrypted, gitignored)

## Dependencies

### Core Dependencies
- `fastmcp`: MCP server framework
- `pydantic`: Data validation and models
- `httpx`: Async HTTP client
- `sqlalchemy`: Database ORM with encryption support
- `cryptography`: Data encryption utilities

### Development Dependencies
- `pytest`: Testing framework with async support
- `black`: Code formatting
- `ruff`: Fast Python linter with security rules
- `mypy`: Static type checking

## Environment Variables

Required environment variables (see `.env.example`):
- `OPENROUTER_API_KEY`: OpenRouter API access
- `ANTHROPIC_API_KEY`: Anthropic API access (optional)
- `OPENAI_API_KEY`: OpenAI API access (optional)
- `LMStudio_BASE_URL`: LM Studio server URL (optional)
- `DATABASE_ENCRYPTION_KEY`: Local database encryption key
- `DEFAULT_COST_LIMIT`: Default per-request cost limit