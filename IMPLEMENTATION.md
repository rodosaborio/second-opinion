# Second Opinion Implementation Guide

This document serves as the master implementation guide for the Second Opinion AI tool, focusing on current architecture patterns and proven implementation strategies for building reliable MCP tools.

## Table of Contents

1. [Development Completion Checklist](#development-completion-checklist)
2. [Architecture Overview](#architecture-overview)
3. [MCP Implementation Blueprint](#mcp-implementation-blueprint)
4. [Core Infrastructure Components](#core-infrastructure-components)
5. [Implementation Status](#implementation-status)
6. [Development Patterns & Best Practices](#development-patterns--best-practices)
7. [Future Features](#future-features)

## Development Completion Checklist

Before declaring any coding task complete, run this comprehensive checklist to ensure production-ready code quality:

### ✅ Code Quality Gates

**1. Formatting & Linting**:
```bash
# Auto-format code
uv run ruff format .

# Fix linting issues
uv run ruff check . --fix

# Verify no remaining issues
uv run ruff check .
```

**2. Type Checking**:
```bash
# Run type checker on entire source
uvx ty check src/

# Address all type issues before completion
```

**3. Security Testing**:
```bash
# Run security-focused tests
uv run pytest -m security

# Run bandit security linter
uv run bandit -r src/

# Verify input sanitization tests pass
uv run pytest tests/test_core/test_sanitization.py -v
```

**4. Test Suite Validation**:
```bash
# Run full test suite with coverage
uv run pytest --cov=second_opinion --cov-report=html --cov-fail-under=85

# Run baseline regression tests
uv run pytest tests/test_conversation_storage/test_baseline_behavior_simple.py -v

# Verify no hanging tests (60s timeout)
uv run pytest --timeout=60
```

### ✅ Integration Verification

**5. MCP Tool Testing**:
```bash
# Test MCP server startup
uv run python -m second_opinion.mcp.server

# Test all tools respond correctly
# (Manual verification in MCP client)
```

**6. CLI Testing**:
```bash
# Test CLI commands
uv run second-opinion --help
uv run second-opinion "Test prompt" --primary-model "openai/gpt-4o-mini"
```

### 🚨 Critical Success Criteria

**Before merging/declaring complete:**
- [ ] **ALL tests pass** (`uv run pytest`)
- [ ] **Type checking clean** (`uvx ty check src/`)
- [ ] **Linting clean** (`uv run ruff check .`)
- [ ] **Security tests pass** (`uv run pytest -m security`)
- [ ] **Test count increased appropriately** (new features = new tests)
- [ ] **No hanging or slow tests** without proper marking
- [ ] **Error handling tested** with mock failure scenarios
- [ ] **Resource cleanup verified** (no connection leaks)

**Quality Gates:**
- [ ] **Code coverage ≥ 85%** for new components
- [ ] **No hardcoded values** that should be configurable
- [ ] **All public methods documented** with clear examples
- [ ] **Error messages user-friendly** and actionable

## Architecture Overview

### System Design Philosophy

**Core Principles:**
- **Configuration-driven design** over hardcoded values
- **Cost-efficiency first** with response reuse and explicit parameters
- **Security by default** at every layer
- **Explicit over implicit** for reliability and maintainability

### Current Production Architecture

```
User/MCP Client → FastMCP Server → Tool Router → Cost Guard → Provider Clients → Model APIs
                        ↓
                   Session Manager → Cost Tracking → Database Storage
```

### Proven Technology Stack

**Core Components:**
- **FastMCP**: MCP server framework with lifecycle management
- **OpenRouter**: Unified API for all cloud model providers (Anthropic, OpenAI, Google)
- **Pydantic V2**: Data validation and serialization
- **SQLite + Encryption**: Local storage with security
- **Rich + Typer**: Professional CLI interface

## MCP Implementation Blueprint

Based on the successful `second_opinion` tool implementation, this section provides a proven blueprint for building reliable MCP tools.

### FastMCP Server Foundation

**Server Setup** (`src/second_opinion/mcp/server.py`):
```python
from mcp import FastMCP
from contextlib import asynccontextmanager

@asynccontextmanager
async def mcp_lifespan():
    """Manage server lifecycle with proper cleanup."""
    # Initialize global resources
    yield
    # Cleanup resources

mcp = FastMCP(name="Second Opinion", lifespan=mcp_lifespan)

# Tool registration with comprehensive documentation
@mcp.tool(
    name="second_opinion",
    description="Compare AI responses across models for quality assessment and cost optimization"
)
async def second_opinion(
    prompt: str,                                # Required: Question to analyze
    primary_model: str | None = None,           # Model that generated original response
    primary_response: str | None = None,        # Existing response (saves API costs)
    context: str | None = None,                 # Additional task context
    comparison_models: list[str] | None = None, # Specific comparison models
    cost_limit: float | None = None             # Max cost in USD
) -> str:
    """Tool implementation with response reuse and cost optimization."""
    # Implementation follows standard 9-step pattern
```

**Key Architecture Decisions:**
1. **FastMCP Framework**: Proven reliability with excellent developer experience
2. **Explicit Parameter Design**: Client intelligence over complex detection logic
3. **Response Reuse Priority**: Cost optimization through existing response evaluation
4. **Comprehensive Documentation**: Rich parameter descriptions for excellent UX

### Standard Tool Implementation Pattern

**Core Tool Structure**:
```python
@mcp.tool(name="tool_name", description="...")
async def tool_implementation(
    # Required parameters first
    prompt: str,
    # Optional parameters with clear defaults
    primary_model: str | None = None,
    primary_response: str | None = None,
    context: str | None = None,
    cost_limit: float | None = None
) -> str:
    """
    Standard MCP tool implementation pattern.
    Returns: Structured markdown response with actionable insights.
    """

    # 1. Parameter validation and defaults
    session = get_session()
    cost_guard = get_cost_guard()

    # 2. Model provider detection (configuration-driven)
    if primary_model:
        provider = detect_model_provider(primary_model)
        primary_client = create_client_from_config(provider)

    # 3. Cost estimation and budget check
    estimated_cost = await estimate_total_cost(...)
    budget_check = await cost_guard.check_and_reserve_budget(
        estimated_cost, "tool_name", primary_model,
        per_request_override=cost_limit
    )

    try:
        # 4. Core tool logic with response reuse optimization
        if primary_response:
            # Use provided response - saves API costs
            clean_primary_response = filter_think_tags(primary_response)
        else:
            # Generate new primary response
            primary_response_obj = await primary_client.complete(request)
            clean_primary_response = filter_think_tags(primary_response_obj.content)

        # 5. Comparison model selection and evaluation
        comparison_models = select_comparison_models(
            primary_model, comparison_models, context
        )

        # 6. Response processing and evaluation
        results = await process_comparisons(...)

        # 7. Record actual costs
        await cost_guard.record_actual_cost(
            budget_check.reservation_id, actual_cost, primary_model, "tool_name"
        )

        # 8. Format response for MCP client
        return format_tool_response(results, session_cost_summary=True)

    except Exception as e:
        # 9. Error handling with cost cleanup
        await cost_guard.release_reservation(budget_check.reservation_id)
        raise MCPToolError(f"Tool execution failed: {str(e)}")
```

### Configuration-Driven Model Selection

**Provider Detection Pattern**:
```python
def detect_model_provider(model: str) -> str:
    """Detect model provider based on model name patterns."""
    # Local model patterns (without provider prefix)
    local_patterns = ['mlx', 'qwen', 'llama', 'mistral']

    if any(pattern in model.lower() for pattern in local_patterns) and "/" not in model:
        return "lmstudio"

    # All other models (including provider prefixes) use OpenRouter
    return "openrouter"

def create_client_from_config(provider: str) -> BaseClient:
    """Create client instance from configuration."""
    settings = get_settings()

    if provider == "openrouter":
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            pricing_manager=get_pricing_manager()
        )
    elif provider == "lmstudio":
        return LMStudioClient(
            base_url=settings.lmstudio.base_url,
            timeout=settings.lmstudio.timeout
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

## Core Infrastructure Components

### Session Management

**MCPSession** (`src/second_opinion/mcp/session.py`):
- **Per-call design**: Sessions created per tool invocation, not global storage
- **Database recovery**: Multi-turn continuity via conversation storage
- **Enhanced context**: 4x larger context window with smart truncation
- **Cost transparency**: Comprehensive cost tracking with model usage patterns

### Cost Protection

**CostGuard** (`src/second_opinion/utils/cost_tracking.py`):
- **Budget hierarchy**: Per-request > user config > global limits
- **Reservation system**: Reserve budget before API calls, record actual costs after
- **Cost estimation**: Accurate pre-flight cost calculations
- **Resource cleanup**: Automatic reservation release on errors

### Response Processing

**Think Tag Filtering**:
```python
def filter_think_tags(text: str) -> str:
    """Remove reasoning tags from model responses for clean display."""
    think_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        # ... additional patterns
    ]

    filtered_text = text
    for pattern in think_patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)

    return filtered_text.strip()
```

### Conversation Storage

**ConversationOrchestrator** coordinates:
- **Unified storage**: Both CLI and MCP interfaces with optional flags
- **Field-level encryption**: AES-256-GCM for sensitive data
- **Searchable history**: Find responses by content, model, cost, date
- **Export capabilities**: JSON, CSV, markdown formats

### Template Management

**Template Loader System** (`src/second_opinion/utils/template_loader.py`):
- **Externalized prompts**: All prompts moved from hardcoded strings to organized template files
- **Categorized structure**: Separate directories for evaluation, MCP, and system prompts
- **Cached loading**: Performance optimization with intelligent caching
- **Fallback handling**: Graceful degradation when templates are unavailable

**Prompt Organization**:
```
prompts/
├── evaluation/              # Core evaluation and comparison prompts
├── mcp/tool_descriptions/   # MCP tool descriptions externalized from server.py
└── system/                  # System-level prompts (followup evaluation, etc.)
```

## Implementation Status

### ✅ Completed MCP Tools (Production Ready)

**5/5 Tools Fully Functional**:
1. **`second_opinion`** - Core comparison and model recommendation engine
2. **`should_downgrade`** - Cost optimization through cheaper alternatives
3. **`should_upgrade`** - Quality improvement analysis for premium models
4. **`compare_responses`** - Detailed side-by-side response analysis
5. **`consult`** - AI-to-AI consultation for expert opinions and task delegation

### ✅ Core Infrastructure (Stable)

**Infrastructure Components**:
- **FastMCP Server**: Production-ready with lifecycle management
- **Session Management**: Per-call design with database recovery
- **Cost Protection**: Budget hierarchy with reservation system
- **Provider Detection**: Configuration-driven client creation
- **Conversation Storage**: Encrypted SQLite with export capabilities
- **Template Management**: Externalized prompt system with organized structure
- **CLI Interface**: Rich formatting with conversation history

### Current Status

**Overall Assessment**: **🟢 EXCELLENT** - All functionality working perfectly:
- CLI regression resolved, OpenRouter API integration restored
- Conversation storage fully functional with encryption and analytics
- All 5/5 MCP tools fully functional with zero regressions detected
- Cost optimization features working (response reuse saves 100% of API costs)
- Quality assessment reliable with consistent evaluation scores

## Development Patterns & Best Practices

### Template Management Patterns

**Externalized Prompt Architecture**:
```python
# ✅ Good: Use centralized template loading
from ..utils.template_loader import load_mcp_tool_description, load_system_template

# MCP tool descriptions
@mcp.tool(
    name="second_opinion",
    description=get_tool_description("second_opinion"),  # Loads from prompts/mcp/tool_descriptions/
)

# System prompts
class FollowUpEvaluator:
    def _get_evaluation_prompt(self) -> str:
        try:
            return load_system_template("followup_evaluation")  # Loads from prompts/system/
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            return fallback_prompt  # Always have fallback
```

**Template Organization Principles**:
- **Category separation**: evaluation/, mcp/, system/ for clear purpose
- **Hierarchical structure**: tool_descriptions/ subdirectory for MCP tools
- **Template caching**: Performance optimization with `use_cache=True`
- **Error resilience**: Graceful fallback when templates fail to load

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

### Testing Infrastructure

**Test Design Principles**:
- **Complete mock implementation**: Mock classes must implement ALL abstract methods
- **Proper Pydantic objects**: Use actual model objects instead of raw dictionaries
- **Global state reset**: Use `autouse=True` pytest fixtures for test isolation
- **Async resource cleanup**: Prevent test hanging with timeout mechanisms

**Test Performance**:
```bash
# Development testing (fast feedback)
uv run pytest  # No coverage, optimized for speed

# Production testing (complete validation)
uv run pytest --cov=second_opinion --cov-report=html --cov-fail-under=85
```

### Error Handling Patterns

**Graceful Degradation**:
- Comprehensive input validation for all parameters
- Circuit breaker patterns for external API reliability
- Rate limiting and resource protection with cost guards
- Graceful fallback when evaluation APIs fail

## Future Features

### Next MCP Tools Pipeline

**High-Priority Tools Ready for Implementation**:

1. **`usage_analytics`** - Cost and usage tracking with insights
2. **`batch_comparison`** - Multiple response comparison and ranking
3. **`model_benchmark`** - Comprehensive model testing across task types

### Development Workflow for New Tools

**1. Tool Implementation** (reuse existing patterns):
- Copy `src/second_opinion/mcp/tools/second_opinion.py` as template
- Implement tool-specific logic while preserving standard 9-step flow
- Use existing `get_session()`, `get_cost_guard()`, and `detect_model_provider()` patterns

**2. Testing** (leverage existing infrastructure):
- Copy `tests/test_mcp/test_second_opinion_tool.py` as starting point
- Use existing mock fixtures and utilities from `tests/conftest.py`
- Test cost tracking, error handling, and parameter validation

**3. Integration** (minimal server changes):
- Register new tool in `src/second_opinion/mcp/server.py`
- Update `src/second_opinion/mcp/__init__.py` exports
- Add tool-specific configuration to config files

**Benefits of This Approach**:
- **90%+ code reuse** from existing infrastructure
- **Consistent quality** through proven patterns
- **Reliable cost optimization** with established response reuse strategies
- **Fast development cycles** with minimal setup required

---

*This implementation guide focuses on proven patterns and actionable guidance for ongoing development. For detailed development commands and testing procedures, see [CLAUDE.md](CLAUDE.md).*/
