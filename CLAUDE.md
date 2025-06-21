# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Second Opinion is an AI tool that provides "second opinion" functionality for AI responses via MCP (Model Context Protocol). It helps users pull in new perspectives from different models and optimize their usage by comparing responses across them, suggesting cost-effective alternatives, and tracking usage patterns.

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

- Add new feature ideas to Future Features in IMPLEMENTATION.MD
- Add bugs, todos and hacks in the relevant sectin in IMPLEMENTATION.MD
- Do habitual cleanup of IMPLEMENTATION.MD. Keep detailed plans for the next feature or currently-being-implemented feature, but delete once completed save for important tidbits or insights learned from the features.

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

## Development Guidelines and Rules
* No Emoji use in code or prompts or output of any kind. It distracts from what we are trying to do and may not display properly on all devices and output modes.
* No adding TODOs or HACKS without explicit user approval and documentation as required work on IMPLEMENTATION.MD
* No hardcoding any model names on actual class implementations. Defaults must come from configuration files
* Breaking tests should neither simply get patched at the test layer or immediately change core implementations, but rather should be an opportunity to think hard, reflect and analyze why they broke. Some cases are easy and other not so much
## 🔧 Development Patterns & Learnings
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
### ✅ Completed MCP Tools

**Production-Ready Tools** (all implemented and tested):

**1. `second_opinion`** - ✅ Core comparison and model recommendation engine
**2. `should_downgrade`** - ✅ Cost optimization through cheaper alternatives
**3. `should_upgrade`** - ✅ Quality improvement analysis for premium models
**4. `compare_responses`** - ✅ Detailed side-by-side response analysis

### Next MCP Tools Pipeline

**High-Priority Tools Ready for Implementation**:

**1. `usage_analytics`** - Cost and usage tracking with insights
```python
@mcp.tool(name="usage_analytics", description="Analyze model usage patterns and cost optimization opportunities")
async def usage_analytics(
    time_period: str = "week",    # Analysis period
    breakdown_by: str = "model",  # Group by model, tool, or cost
    include_recommendations: bool = True
) -> str:
```

**2. `batch_comparison`** - Multiple response comparison
```python
@mcp.tool(name="batch_comparison", description="Compare multiple responses to the same task")
async def batch_comparison(
    responses: list[str],         # Multiple responses to compare
    task: str,                   # Original task
    models: list[str] = None,    # Corresponding models
    rank_by: str = "quality"     # Ranking criteria
) -> str:
```

**3. `model_benchmark`** - Comprehensive model testing
```python
@mcp.tool(name="model_benchmark", description="Benchmark models across different task types")
async def model_benchmark(
    models: list[str],           # Models to benchmark
    task_types: list[str] = None, # Task categories to test
    sample_size: int = 5         # Number of tasks per category
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
