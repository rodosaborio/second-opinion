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

## 🚀 Current Implementation Status

### ✅ Completed Components (Phase 1)
- **Core Models** (`src/second_opinion/core/models.py`) - Comprehensive Pydantic V2 models with validation
- **Configuration System** (`src/second_opinion/config/`) - Environment variables, YAML configs, security
- **Test Infrastructure** - 90%+ coverage with security-focused testing

### ✅ Completed Components (Phase 2)
- **Security Utils** (`src/second_opinion/utils/sanitization.py`) - Multi-layer input validation and sanitization
- **Abstract Client Interface** (`src/second_opinion/clients/base.py`) - Standardized provider interface with error handling
- **Cost Tracking Framework** (`src/second_opinion/utils/cost_tracking.py`) - Comprehensive budget management system
- **Enhanced Testing** - 183 total tests with 95%+ functional coverage, extensive security testing

### ✅ Completed Components (Phase 3)
- **Prompt System** (`src/second_opinion/prompts/manager.py`) - Template loading, caching, parameter injection with security-aware rendering
- **Response Evaluation Engine** (`src/second_opinion/core/evaluator.py`) - Response comparison, quality scoring, cost-benefit analysis
- **Task Complexity Classification** - Intelligent algorithm with weighted scoring and technical term detection
- **Model Recommendation System** - Quality-first logic with cost-aware upgrade/downgrade/maintain recommendations
- **Enhanced Templates** - 6 comprehensive prompt templates including cost-benefit analysis and model recommendations
- **Integration Testing** - 84 total tests passing with comprehensive end-to-end verification

### ✅ Completed Components (Phase 4a: OpenRouter Client)
- **OpenRouter Client** (`src/second_opinion/clients/openrouter.py`) - Complete OpenRouter API integration with cost tracking, model discovery, and error handling
- **Client Factory System** (`src/second_opinion/clients/__init__.py`, `src/second_opinion/utils/client_factory.py`) - Dynamic provider instantiation and configuration-based client creation
- **Production-Ready Implementation** - Full OpenRouter API compliance, security validation, type safety, and comprehensive testing
- **Enhanced Testing** - 58 additional tests (39 OpenRouter + 19 factory) with 95%+ functional coverage and security scenarios

### ✅ Completed Components (Phase 4b: Dynamic Pricing Integration)
- **Dynamic Pricing Manager** (`src/second_opinion/utils/pricing.py`) - LiteLLM integration with 1,117+ models, real-time pricing, caching, and intelligent fallbacks
- **Pricing Configuration** (`src/second_opinion/config/settings.py`) - Full configuration system with TTL, auto-update, and backup settings
- **OpenRouter Integration** - Real-time cost estimates using comprehensive pricing data instead of hardcoded values
- **Cost Tracking Enhancement** - Updated to use pricing manager with backward compatibility for legacy systems
- **Comprehensive Testing** - 50+ additional tests covering pricing manager, integration scenarios, security, and edge cases

### 🔄 Next: CLI Interface & MCP Integration
Ready to implement CLI interface and MCP server using the complete OpenRouter client foundation with accurate dynamic pricing.

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
# Fast development testing (no coverage, 60s timeout protection)
uv run pytest

# Run tests with full coverage reporting (slower)
uv run pytest --cov=second_opinion --cov-report=term-missing --cov-report=html --cov-fail-under=85

# Run only security tests
uv run pytest -m security

# Run specific test file
uv run pytest tests/test_core/test_evaluator.py

# Skip slow tests for faster feedback
uv run pytest -m "not slow"

# Run with verbose output for debugging hanging tests
uv run pytest -v --tb=long

# Run integration tests
uv run pytest -m integration

# Continuous testing during development (fast feedback)
uv run pytest --timeout=30 -x --tb=short

# Debug test isolation issues
uv run pytest -v --setup-show
```

### Code Quality
```bash
# AUTOMATED: Pre-commit hooks handle this automatically!
# Install hooks once: pre-commit install

# Manual formatting (if needed)
uv run black .

# Manual linting (if needed)
uv run ruff check .
uv run ruff check . --fix

# Manual type checking (if needed)  
uv run mypy src/

# Run ALL quality checks manually
uv run black . && uv run ruff check . --fix && uv run mypy src/

# Update pre-commit hooks
pre-commit autoupdate

# Run pre-commit on all files (manual trigger)
pre-commit run --all-files
```

### Running the Application
```bash
# CLI usage - Basic model comparison
uv run second-opinion --help
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" "What's 2+2?"

# CLI usage - Advanced features (NEW!)
# Verbose mode for full responses (helpful for thinking models)
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" --verbose "Complex question here"

# Use existing response to save API calls and tokens
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" \
  --existing-response "The capital of France is Paris." \
  "What's the capital of France?"

# Multiple comparison models with context
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" \
  --comparison-model "openai/gpt-4o" \
  --comparison-model "google/gemini-pro" \
  --context "This is for academic research" \
  --verbose \
  "Analyze the economic impact of climate change"

# Start MCP server
uv run python -m second_opinion.mcp.server

# Development server with auto-reload
uv run python -m second_opinion.mcp.server --dev
```

## 🔧 Development Patterns & Learnings

### Lesson Learned: Blocking Calls and Test Behavior
- Discovered critical issues with blocking calls in async test environments
- Blocking calls can cause test deadlocks and unexpected timeouts
- Always use async equivalents or properly wrap blocking calls with `asyncio.to_thread()` or `run_in_executor()`
- Implement global timeout and resource cleanup mechanisms to prevent test hanging
- Use `@pytest.mark.asyncio` and ensure proper async test fixture management


## 🔧 Test Infrastructure & Performance Fixes

### Global State Management
The test suite now includes comprehensive global state reset to prevent test isolation issues:

```python
# Automatic global state reset between tests (tests/conftest.py)
@pytest.fixture(autouse=True, scope="function") 
def reset_global_state():
    # Resets: config_manager, pricing_manager, cost_guard
    # Clears: module caches, HTTP clients, async resources
    pass

# Isolated test environment with clean temp directories
@pytest.fixture(scope="function")
def isolated_temp_dir(tmp_path):
    # Sets up: isolated DATA_DIR, CONFIG_DIR, DATABASE paths
    # Prevents: test data pollution and cross-test interference
    pass
```

### Test Hanging Prevention
Multiple layers of protection against hanging tests:

```bash
# Global 60-second timeout for all tests
--timeout=60

# Async resource cleanup
@pytest.fixture(autouse=True)
def ensure_async_cleanup():
    # Force cleanup of pending async tasks
    # Prevents event loop pollution between tests

# Mock HTTP clients by default
@pytest.fixture
def mock_http_client():
    # Prevents real network calls during testing
    # Ensures consistent, fast test execution
```

### Performance Optimizations

```bash
# Development testing (fast feedback)
uv run pytest  # No coverage, optimized for speed

# Production testing (complete validation)  
uv run pytest --cov=second_opinion --cov-report=html --cov-fail-under=85

# Test execution improvements:
✅ Global state reset fixtures prevent hanging
✅ 60-second timeout protection for all tests
✅ Async resource cleanup prevents memory leaks
✅ Mock HTTP clients prevent network delays
✅ Isolated temp directories prevent data pollution
✅ Optimized pytest configuration for development
```

### Test Debugging Tools

```bash
# Debug hanging tests
uv run pytest -v --tb=long --setup-show

# Fast continuous testing
uv run pytest --timeout=30 -x --tb=short

# Monitor test isolation
uv run pytest -v --setup-show tests/test_clients/test_openrouter.py

# Profile test performance
uv run pytest --durations=10 --timeout=30
```

## Testing Best Practices

- Always use async-friendly testing techniques
- Implement comprehensive test isolation
- Prevent global state pollution
- Use mock objects to simulate complex scenarios
- Implement timeouts to catch hanging tests
- Maintain high test coverage (85%+)
- Write security-focused test scenarios
- Use fixtures for setup and teardown
- Leverage pytest markers for test categorization
- Profile and optimize test performance