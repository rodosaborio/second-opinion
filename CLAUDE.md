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

### ✅ Completed Components (Phase 5: CLI Interface Implementation)
- **CLI Main Interface** (`src/second_opinion/cli/main.py`) - Complete Typer-based CLI with rich output formatting and comparison model flag support
- **--existing-response Flag** - Users can provide existing primary model response to save API calls and tokens
- **--verbose Flag** - Full response display mode for detailed analysis (helpful for thinking models)
- **Multiple Comparison Models** - Support for `--comparison-model` flag used multiple times for comprehensive analysis
- **Rich User Experience** - Beautiful terminal UI with tables, panels, progress indicators, and cost transparency
- **Intelligent Model Selection** - Smart auto-selection based on primary model tier and task complexity with priority hierarchy
- **Cost Protection Integration** - Full integration with existing cost tracking and budget management systems
- **Comprehensive Testing** - 17+ test scenarios covering CLI functionality, model selection, error handling, and user experience

### ✅ Completed Components (Phase 6: Evaluation Engine Enhancements)
- **Real Evaluation API Integration** - Replaced simulation with actual model-based evaluation using OpenRouter client
- **Cost Integration** - Integrated real budget tracking from cost guard system with fallback protection
- **Task Complexity Detection** - Added intelligent task complexity classification to CLI workflow with user feedback
- **Think Tag Filtering** - Implemented filtering of `<think>`, `<thinking>`, and similar reasoning tags from responses
- **Enhanced Response Processing** - Applied filtering in both summary and verbose display modes for cleaner output
- **Robust Error Handling** - Graceful fallback to simulation when evaluation API calls fail
- **Comprehensive Testing** - All existing tests pass, new functionality validated

### ✅ Completed Components (Phase 7: MCP Tools & Configuration-Driven Design)
- **MCP Tool Implementation** (`src/second_opinion/mcp/tools/second_opinion.py`) - Fixed hardcoded providers, now uses `detect_model_provider()` for config-driven client selection
- **Security Enhancements** (`src/second_opinion/utils/sanitization.py`) - Updated prompt injection detection to allow code snippets while maintaining security
- **Budget Configuration Fixes** (`src/second_opinion/utils/cost_tracking.py`) - Fixed hierarchy to use config-based limits with per-request overrides  
- **Comprehensive Test Infrastructure** (`tests/test_mcp/`) - Created proper mock utilities and unit tests independent of real API keys
- **Enhanced Testing** - 15 MCP tool tests + existing test suite all passing with improved mock strategies

### ✅ Complete MCP Integration
MCP server tools fully implemented using the complete evaluation engine, CLI interface, and OpenRouter foundation.

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

### MCP Implementation Blueprint
For implementing new MCP tools, follow the proven patterns documented in **[IMPLEMENTATION.md](IMPLEMENTATION.md)**:

- **FastMCP Server Foundation**: Established server setup with lifecycle management and session tracking
- **Tool Implementation Pattern**: Standard 9-step flow for cost-efficient, secure tool development
- **Response Reuse Optimization**: 50-80% cost reduction through existing response evaluation
- **Configuration-Driven Design**: No hardcoding - use `detect_model_provider()` for dynamic provider selection
- **Comprehensive Testing**: Reusable mock strategies and fixtures for reliable test coverage

**Ready for Implementation**: `should_downgrade`, `should_upgrade`, `compare_responses`, `usage_analytics`

### Core Development Patterns

**Configuration-Driven Design**:
```python
# ✅ Good: Configuration-driven provider detection
provider = detect_model_provider(model)
client = create_client_from_config(provider)

# ✅ Good: Configuration hierarchy (CLI > config > defaults)
budget_check = await cost_guard.check_and_reserve_budget(
    estimated_cost, "tool", model, per_request_override=user_cost_limit
)
```

**Cost Optimization Strategy**:
```python
# Core pattern: Response reuse for significant cost savings
if primary_response:
    # Use provided response - zero additional API cost
    clean_response = filter_think_tags(primary_response)
    return clean_response, Decimal("0.0")
else:
    # Generate new response - incurs API cost
    response = await client.complete(request)
    return filter_think_tags(response.content), response.cost_estimate
```

**Security Context Awareness**:
- **USER_PROMPT**: Permissive for code snippets and technical content
- **API_REQUEST**: Strict validation for security
- **Balance**: Maintain security for serious threats while enabling legitimate use cases

**Testing Infrastructure**:
- **Complete mock implementation**: Mock classes must implement ALL abstract methods from base classes
- **Proper Pydantic objects**: Use actual model objects (`TokenUsage`, `Message`) instead of raw dictionaries
- **Global state reset**: Use `autouse=True` pytest fixtures for test isolation
- **Async resource cleanup**: Prevent test hanging with timeout and cleanup mechanisms


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

## Test Writing and Mocking Guidelines

### Configuration Independence
- Prioritize independence from actual configurations for tests
- Use dependency injection and mocking to isolate test components
- Create mock objects that simulate real dependencies without actual configurations
- Ensure tests are deterministic and do not rely on external state or configurations

### Comprehensive Mock Implementation
- **Complete abstract method implementation**: Mock classes must implement ALL abstract methods from base classes
```python
class MockClient(BaseClient):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    async def estimate_cost(self, request: ModelRequest) -> Decimal: ...
    async def list_models(self) -> List[ModelInfo]: ...
    async def get_model_info(self, model: str) -> ModelInfo: ...
    async def get_available_models(self) -> List[ModelInfo]: ...  # Often missed!
```

### Proper Object Creation in Mocks
- **Use actual Pydantic objects**: Don't use raw dictionaries for complex objects
```python
# Good: Proper object creation
return ModelResponse(
    content=mock_content,
    model=request.model,
    usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),  # Real object
    cost_estimate=self.mock_costs,
    provider=self.provider_name,
    metadata={'mock': True}
)

# Bad: Raw dictionary usage
return ModelResponse(
    usage={'input_tokens': 10, 'output_tokens': 20}  # Will fail Pydantic validation
)
```

### Dependency Injection Patterns  
- Prefer interfaces and abstract base classes for easier mocking
- Use `unittest.mock` or `pytest-mock` for creating mock objects
- Implement dependency injection to allow easy substitution of real implementations with mocks
- Use monkeypatch for comprehensive dependency replacement

### Test Design Principles
- Write tests that focus on behavior and logic rather than specific implementation details
- Use parametrization to test multiple scenarios without duplicating test code
- Mock external services, databases, and network calls to ensure test reliability and speed
- Create context-aware mocks that handle different scenarios (provider detection, model patterns, etc.)

## 🚀 MCP Tool Development Blueprint

### Reusable Architecture Components

The successful `second_opinion` tool provides a proven blueprint for implementing additional MCP tools efficiently:

**Core Infrastructure** (100% reusable):
- **FastMCP Server**: `src/second_opinion/mcp/server.py` - Production-ready server with lifecycle management
- **Session Management**: `src/second_opinion/mcp/session.py` - Cost tracking and conversation context
- **Provider Detection**: `src/second_opinion/utils/client_factory.py` - Configuration-driven client creation
- **Cost Protection**: `src/second_opinion/utils/cost_tracking.py` - Budget reservation and recording system
- **Security Framework**: `src/second_opinion/utils/sanitization.py` - Multi-layer input validation

**Implementation Pattern** (standard 9-step flow):
1. Parameter validation and defaults
2. Model provider detection (configuration-driven)
3. Cost estimation and budget check
4. Response reuse optimization (if applicable)
5. Core tool logic execution
6. Result processing and evaluation
7. Actual cost recording
8. Response formatting for MCP clients
9. Error handling with cost cleanup

### Next MCP Tools Pipeline

**High-Priority Tools Ready for Implementation**:

**1. `should_downgrade`** - Cost optimization through cheaper alternatives
```python
@mcp.tool(name="should_downgrade", description="Test if cheaper models could achieve similar quality")
async def should_downgrade(
    current_response: str,        # Response to analyze for cost savings
    task: str,                   # Original task/question
    current_model: str = None,   # Model that generated response
    test_local: bool = True      # Include LM Studio models
) -> str:
```

**2. `should_upgrade`** - Quality improvement analysis
```python
@mcp.tool(name="should_upgrade", description="Evaluate if premium models justify additional cost")
async def should_upgrade(
    current_response: str,        # Response to analyze for quality improvements
    task: str,                   # Original task/question  
    current_model: str = None,   # Current model
    upgrade_target: str = None   # Specific premium model to test
) -> str:
```

**3. `compare_responses`** - Detailed side-by-side analysis
```python
@mcp.tool(name="compare_responses", description="Detailed comparison across quality criteria")
async def compare_responses(
    response_a: str,             # First response
    response_b: str,             # Second response
    task: str,                   # Original task for context
    model_a: str = None,         # Model A for cost analysis
    model_b: str = None          # Model B for cost analysis
) -> str:
```

### Development Workflow for New Tools

**1. Tool Implementation** (reuse existing patterns):
- Copy `src/second_opinion/mcp/tools/second_opinion.py` as template
- Implement tool-specific logic while preserving standard flow
- Use existing `get_session()`, `get_cost_guard()`, and `detect_model_provider()` patterns

**2. Testing** (leverage existing infrastructure):
- Copy `tests/test_mcp/test_second_opinion_tool.py` as starting point
- Use existing mock fixtures and utilities from `tests/conftest.py`
- Test cost tracking, error handling, and parameter validation

**3. Integration** (minimal server changes):
- Register new tool in `src/second_opinion/mcp/server.py`
- Update `src/second_opinion/mcp/__init__.py` exports
- Add tool-specific configuration to `config/mcp_profiles.yaml`

**Benefits of This Approach**:
- **90%+ code reuse** from existing infrastructure
- **Consistent quality** through proven patterns
- **Reliable cost optimization** with established response reuse strategies
- **Comprehensive testing** using validated mock strategies
- **Fast development cycles** with minimal setup required

</invoke>