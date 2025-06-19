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
   - ✅ OpenRouter client for multi-model access (COMPLETED)
   - LM Studio client for local development (READY TO IMPLEMENT)
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

### Pricing Configuration Variables
- `PRICING__ENABLED`: Enable/disable dynamic pricing (default: true)
- `PRICING__CACHE_TTL_HOURS`: Pricing cache TTL in hours (default: 1)
- `PRICING__FETCH_TIMEOUT`: HTTP fetch timeout in seconds (default: 30.0)
- `PRICING__AUTO_UPDATE_ON_STARTUP`: Auto-update pricing on startup (default: true)
- `PRICING__BACKUP_FILE_PATH`: Custom backup file path (optional)

## 🔧 Development Patterns & Learnings

### Configuration Management
- **Use pydantic-settings**: Don't reinvent environment variable handling
- **YAML for structure**: Use YAML configs for complex nested settings
- **Security pattern**: `repr=False` on sensitive fields to prevent logging secrets
- **Environment nesting**: Use `__` delimiter for nested configs (e.g., `DATABASE__ENCRYPTION_ENABLED=true`)

### Pydantic V2 Best Practices
```python
# ✅ Modern validation patterns
@field_validator('field_name')
@classmethod
def validate_field(cls, v):
    return v

@model_validator(mode='after')
def validate_model(self):
    return self

# ✅ Computed properties for derived data
@property
def computed_field(self) -> type:
    return self.field_a + self.field_b
```

### Testing Isolation
```python
# ✅ Global state management for tests
@pytest.fixture(autouse=True)
def reset_global_state():
    global_manager.reset()
    yield
    global_manager.reset()
```

### Security-First Development
- Validate all inputs with Pydantic models
- Separate validation levels (development vs production)
- Use `repr=False` for sensitive fields
- Test with malicious inputs using `pytest -m security`

## 🎯 Phase 2 & 3 Implementation Achievements

### Security Utils Foundation
```python
# Context-aware sanitization
sanitize_prompt("user input", SecurityContext.USER_PROMPT)
sanitize_prompt("system instruction", SecurityContext.SYSTEM_PROMPT)

# Multi-layer validation
validate_model_name("anthropic/claude-3-5-sonnet")  # ✅ Valid
validate_model_name("model<script>alert('xss')</script>")  # ❌ Blocked

# Cost limit validation with currency handling
validate_cost_limit("$5.00")  # → Decimal("5.00")
validate_cost_limit("€10.50")  # → Decimal("10.50")
```

### Cost Protection System
```python
# Budget reservation and tracking
budget_check = await check_budget(
    estimated_cost=Decimal("0.05"),
    operation_type="second_opinion", 
    model="gpt-4"
)

# Multi-period budget management
daily_usage = await get_usage_summary(BudgetPeriod.DAILY)
monthly_usage = await get_usage_summary(BudgetPeriod.MONTHLY)

# Detailed analytics
analytics = await get_detailed_analytics(days=30)
# Returns: total_cost, models_used, operations_breakdown, daily_spending
```

### Abstract Client Interface
```python
# Standardized provider interface
class CustomProviderClient(BaseClient):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        # Built-in validation and sanitization
        validated_request = await self.validate_request(request)
        
        # Built-in retry logic with exponential backoff
        return await self.retry_with_backoff(self._api_call, validated_request)
    
    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        # Provider-specific cost calculation
        pass
    
    async def get_available_models(self) -> List[ModelInfo]:
        # Cached model discovery
        pass
```

### Security Testing
```bash
# Run comprehensive security tests (48 security scenarios)
uv run pytest tests/test_utils/test_sanitization.py -m security

# Test cost protection and budget enforcement  
uv run pytest tests/test_utils/test_cost_tracking.py -m security

# Test client security validation
uv run pytest tests/test_clients/test_base.py -m security
```

### Phase 3 Core Logic & Evaluation Engine

```python
# Task complexity classification with intelligent algorithm
evaluator = get_evaluator()
complexity = await evaluator.classify_task_complexity("Design a distributed system")
# Returns: TaskComplexity.COMPLEX with weighted scoring

# Smart model recommendations prioritizing quality
recommendation = await evaluator.recommend_model_tier(
    task="Simple question", 
    current_model="gpt-4o"
)
# Returns: RecommendationType.DOWNGRADE with cost analysis

# Response comparison with multi-criteria evaluation
comparison = await evaluator.compare_responses(
    primary_response, comparison_response, "What is machine learning?"
)
# Returns: ComparisonResult with accuracy, completeness, clarity, usefulness scores
```

### Enhanced Prompt Template System

```python
# Template loading with security-aware parameter injection
prompt_manager = get_prompt_manager()
templates = await prompt_manager.list_templates()
# Returns: ['comparison', 'cost_benefit_analysis', 'model_recommendation', ...]

# Secure template rendering with context validation
rendered = await prompt_manager.render_prompt(
    'second_opinion',
    {
        'original_question': 'How does AI work?',
        'primary_response': 'AI uses algorithms...',
        'primary_model': 'gpt-4o'
    },
    security_context=SecurityContext.USER_PROMPT
)
# Returns: Fully rendered prompt with security validation
```

### Integration Testing Results

```bash
# Comprehensive Phase 3 integration test demonstrates:
✅ Task Complexity Classification: Simple → Expert level detection
✅ Model Recommendation System: Smart upgrade/downgrade logic  
✅ Response Comparison Engine: Multi-criteria evaluation scoring
✅ Enhanced Prompt Templates: 6 templates with parameter injection
✅ Cost Effectiveness Analysis: Optimal model identification

# Test coverage: 84 tests passing with 95%+ coverage on new components
```

### Phase 4b Dynamic Pricing System

```python
# Real-time pricing for 1,117+ models
pricing_manager = get_pricing_manager()
cache_info = pricing_manager.get_cache_info()
# Returns: {"models": 1117, "source": "backup", "is_expired": false}

# Accurate cost estimation
cost, source = pricing_manager.estimate_cost("gpt-4", 1000, 500)
# Returns: (Decimal("0.06000"), "pricing_data_backup")

# OpenRouter client integration
client = OpenRouterClient(api_key="sk-or-...")
estimated_cost = await client.estimate_cost(request)
# Uses dynamic pricing automatically

# Cost tracking integration  
cost_guard = get_cost_guard()
estimated_cost = await cost_guard.estimate_request_cost(request)
# Uses pricing manager by default, legacy mode for compatibility
```

### Integration Test Results

```bash
# Complete dynamic pricing integration test:
✅ Pricing Manager: 1,117 models loaded from LiteLLM data
✅ GPT-4 Cost Estimate: $0.06 for 1000 input + 500 output tokens
✅ OpenRouter Client: $0.06009 (nearly identical, slight token estimation differences)
✅ Cost Tracking: Successfully integrated with pricing manager
✅ 50+ new tests passing with comprehensive coverage
```