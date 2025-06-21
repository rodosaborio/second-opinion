# Second Opinion Implementation Guide

This document serves as the master implementation guide for the Second Opinion AI tool, focusing on current architecture patterns and proven implementation strategies for building reliable MCP tools.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MCP Implementation Blueprint](#mcp-implementation-blueprint)  
3. [Core Infrastructure Components](#core-infrastructure-components)
4. [Development Patterns & Best Practices](#development-patterns--best-practices)
5. [Testing Strategy](#testing-strategy)
6. [Deployment & Production](#deployment--production)

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

**Key Decision Points:**
1. **Input Validation**: Multi-layer sanitization before any external calls
2. **Model Selection**: Configuration-driven provider detection with explicit parameters
3. **Cost Protection**: Pre-flight estimation with hard limits and reservation system
4. **Error Handling**: Graceful degradation with informative error messages

### Proven Technology Stack

**Core Components:**
- **FastMCP**: MCP server framework with lifecycle management
- **OpenRouter**: Unified API for all cloud model providers (Anthropic, OpenAI, Google)
- **Pydantic V2**: Data validation and serialization
- **LiteLLM**: Dynamic pricing data for 1,117+ AI models
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
    # Implementation details below...
```

**Key Architecture Decisions:**
1. **FastMCP Framework**: Proven reliability with excellent developer experience
2. **Explicit Parameter Design**: Client intelligence over complex detection logic
3. **Response Reuse Priority**: Cost optimization through existing response evaluation
4. **Comprehensive Documentation**: Rich parameter descriptions for excellent UX

### Session Management Pattern

**Lightweight Session** (`src/second_opinion/mcp/session.py`):
```python
class MCPSession:
    """Lightweight session management focused on cost tracking and caching."""
    
    def __init__(self):
        self.session_id: str = str(uuid4())
        self.total_cost: Decimal = Decimal("0.0")
        self.tool_costs: Dict[str, Decimal] = {}
        self.operation_count: int = 0
        self.last_used_model: Optional[str] = None
        
    def record_cost(self, tool_name: str, cost: Decimal, model: str):
        """Record operation cost with session tracking."""
        self.total_cost += cost
        self.tool_costs[tool_name] = self.tool_costs.get(tool_name, Decimal("0.0")) + cost
        self.operation_count += 1
        self.last_used_model = model

# Global session management
def get_session() -> MCPSession:
    """Get or create current MCP session."""
    global _current_session
    if _current_session is None:
        _current_session = MCPSession()
    return _current_session
```

**Session Design Principles:**
- **Minimal State**: Only track essential cost and caching data
- **No Complex Detection**: Rely on explicit client parameters
- **Cost Transparency**: Track all operations for user visibility
- **Thread Safety**: Simple design avoids concurrency issues

### Tool Implementation Pattern

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
    else:
        # Fallback to session or configuration defaults
        pass
    
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
    """
    Detect model provider based on model name patterns.
    
    This replaces hardcoded provider selection with configuration-driven logic.
    """
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

**Key Benefits:**
- **No Hardcoding**: Supports both cloud and local providers automatically
- **Extensible**: Easy to add new providers without code changes
- **Reliable**: Simple pattern matching reduces detection errors
- **User-Correctable**: Clear error messages when model format is incorrect

### Response Reuse Cost Optimization

**Core Cost Optimization Strategy**:
```python
async def optimize_api_costs(
    primary_model: str,
    primary_response: str | None,
    prompt: str
) -> tuple[str, Decimal]:
    """
    Implement response reuse for significant cost savings.
    
    Returns: (response_content, api_cost)
    """
    
    if primary_response:
        # Use provided response - zero additional API cost
        clean_response = filter_think_tags(primary_response)
        return clean_response, Decimal("0.0")
    else:
        # Generate new response - incurs API cost
        provider = detect_model_provider(primary_model)
        client = create_client_from_config(provider)
        
        request = ModelRequest(
            model=primary_model,
            messages=[Message(role="user", content=prompt)]
        )
        
        response = await client.complete(request)
        clean_response = filter_think_tags(response.content)
        return clean_response, response.cost_estimate
```

**Cost Savings Impact:**
- **50-80% cost reduction** when primary response is provided
- **Zero marginal cost** for response evaluation and comparison
- **Predictable budgeting** with accurate cost estimation upfront

### Response Processing Pipeline

**Think Tag Filtering**:
```python
def filter_think_tags(text: str) -> str:
    """
    Remove reasoning tags from model responses for clean display.
    
    Handles multiple tag formats and edge cases.
    """
    think_patterns = [
        r'<think>.*?</think>',           # Complete tags
        r'<thinking>.*?</thinking>',     # Alternative format
        r'<thought>.*?</thought>',       # Another common format
        r'<reasoning>.*?</reasoning>',   # Explicit reasoning
        r'<internal>.*?</internal>',     # Internal thoughts
        r'<think>.*?(?=\n\n|$)',         # Unclosed tags
        r'^\\s*</?think[^>]*>.*?(?=\n[A-Za-z]|\\n\\n|$)',  # Edge cases
    ]
    
    filtered_text = text
    for pattern in think_patterns:
        filtered_text = re.sub(pattern, '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up excessive whitespace
    filtered_text = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_text)
    filtered_text = filtered_text.strip()
    
    return filtered_text
```

**Response Formatting for MCP**:
```python
def format_tool_response(
    evaluation_result: dict,
    cost_summary: dict,
    session_cost_summary: bool = True
) -> str:
    """
    Format tool response for optimal MCP client display.
    
    Uses decision-support framing with actionable recommendations.
    """
    
    response_sections = [
        "# 🤔 Second Opinion: Should You Stick or Switch?",
        "",
        "## 🎯 My Recommendation",
        format_recommendation(evaluation_result),
        "",
        "## 💰 Cost Analysis",
        format_cost_analysis(cost_summary),
        "",
        "## 🚀 Next Steps",
        format_action_items(evaluation_result)
    ]
    
    if session_cost_summary:
        response_sections.extend([
            "",
            "## 📊 Session Summary", 
            format_session_summary()
        ])
    
    return "\n".join(response_sections)
```

## Core Infrastructure Components

### Security & Input Validation

**Multi-Layer Security Approach**:
```python
class InputSanitizer:
    """Context-aware input validation and sanitization."""
    
    def sanitize_prompt(self, prompt: str, context: SecurityContext) -> str:
        """
        Sanitize user prompts with context awareness.
        
        USER_PROMPT: Permissive for code snippets and technical content
        API_REQUEST: Strict validation for security
        """
        if context == SecurityContext.USER_PROMPT:
            # Allow code snippets, technical content
            return self._permissive_sanitization(prompt)
        else:
            # Strict validation for API requests
            return self._strict_sanitization(prompt)
    
    def validate_model_name(self, model_name: str) -> str:
        """Validate and normalize model names with helpful error messages."""
        if not re.match(r'^[a-zA-Z0-9/_-]+$', model_name):
            raise ValueError(
                f"Invalid model name format: {model_name}. "
                f"Expected format: 'provider/model' (e.g., 'anthropic/claude-3-5-sonnet')"
            )
        return model_name.lower()
```

### Cost Tracking & Budget Protection

**Budget Guard System**:
```python
class CostGuard:
    """Multi-layer cost protection with reservation system."""
    
    async def check_and_reserve_budget(
        self,
        estimated_cost: Decimal,
        operation_type: str,
        model: str,
        per_request_override: Optional[Decimal] = None
    ) -> BudgetCheck:
        """
        Reserve budget for operation with hierarchy:
        1. per_request_override (CLI flags, tool parameters)
        2. Tool configuration (model_profiles.yaml)  
        3. Settings default (settings.yaml)
        """
        
        # Determine effective cost limit
        if per_request_override is not None:
            cost_limit = per_request_override
        else:
            cost_limit = await self._get_configured_limit(operation_type)
        
        # Check against limit
        if estimated_cost > cost_limit:
            raise CostLimitExceededError(
                f"Estimated cost ${estimated_cost:.4f} exceeds limit ${cost_limit:.4f}"
            )
        
        # Create reservation
        reservation = BudgetReservation(
            id=str(uuid4()),
            amount=estimated_cost,
            operation_type=operation_type,
            model=model,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.active_reservations[reservation.id] = reservation
        return BudgetCheck(
            approved=True,
            reservation_id=reservation.id,
            available_budget=await self._calculate_available_budget()
        )
```

### Dynamic Pricing Integration

**LiteLLM Pricing Manager**:
```python
class PricingManager:
    """Dynamic pricing data with intelligent fallbacks."""
    
    def __init__(self):
        self._cache: Dict[str, ModelPricing] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        self._lock = threading.RLock()
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> tuple[Decimal, str]:
        """
        Estimate cost with multiple fallback strategies.
        
        Returns: (total_cost, source)
        Sources: "litellm", "local_backup", "conservative_fallback"
        """
        
        # Try LiteLLM pricing data first
        pricing_info = self._get_cached_pricing(model)
        if pricing_info:
            input_cost = Decimal(input_tokens) * pricing_info.input_cost_per_1k / 1000
            output_cost = Decimal(output_tokens) * pricing_info.output_cost_per_1k / 1000
            return input_cost + output_cost, "litellm"
        
        # Fall back to conservative estimate
        return self._conservative_estimate(model, input_tokens, output_tokens)
    
    def _conservative_estimate(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[Decimal, str]:
        """Conservative cost estimates based on model tier."""
        
        model_lower = model.lower()
        if any(tier in model_lower for tier in ['gpt-4', 'claude-3-opus', 'gemini-ultra']):
            # Premium tier
            return Decimal("0.10"), "conservative_fallback"
        elif any(tier in model_lower for tier in ['gpt-3.5', 'claude-3-haiku', 'gemini-flash']):
            # Budget tier  
            return Decimal("0.01"), "conservative_fallback"
        else:
            # Default mid-tier
            return Decimal("0.05"), "conservative_fallback"
```

## Development Patterns & Best Practices

### Configuration-Driven Development

**Key Pattern: Never Hardcode What Should Be Configurable**
```python
# ❌ Bad: Hardcoded provider
client = create_client_from_config('openrouter')

# ✅ Good: Configuration-driven
provider = detect_model_provider(model)
client = create_client_from_config(provider)

# ❌ Bad: Hardcoded cost limits  
budget_check = await cost_guard.check_and_reserve_budget(estimated_cost, 'tool', model, Decimal("0.10"))

# ✅ Good: Configuration hierarchy
budget_check = await cost_guard.check_and_reserve_budget(
    estimated_cost, 'tool', model, 
    per_request_override=user_cost_limit  # CLI flags > config > defaults
)
```

### Error Handling Patterns

**Graceful Degradation Strategy**:
```python
async def robust_evaluation_with_fallback(
    primary_response: str,
    comparison_response: str,
    criteria: EvaluationCriteria
) -> ComparisonResult:
    """
    Evaluation with graceful fallback to simulation.
    
    Maintains service reliability when evaluation API fails.
    """
    
    try:
        # Try real model-based evaluation first
        evaluation_result = await self._evaluate_with_model(
            evaluation_prompt, primary_response, comparison_response, criteria
        )
        return evaluation_result
        
    except Exception as e:
        logger.warning(f"Evaluation API call failed: {e}. Falling back to simulation.")
        
        # Fall back to simulation - always works
        evaluation_result = await self._simulate_evaluation(
            primary_response, comparison_response, criteria
        )
        return evaluation_result
```

### Testing Patterns

**Mock Implementation Strategy**:
```python
class MockClient(BaseClient):
    """Complete mock implementation for testing."""
    
    def __init__(self, mock_responses: Dict[str, str] = None):
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.requests_made = []
    
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Mock completion with realistic response structure."""
        self.call_count += 1
        self.requests_made.append(request)
        
        mock_content = self.mock_responses.get(request.model, "Mock response")
        
        return ModelResponse(
            content=mock_content,
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            cost_estimate=Decimal("0.01"),
            provider="mock",
            metadata={'mock': True}
        )
    
    # Implement ALL abstract methods - critical for proper mocking
    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        return Decimal("0.01")
    
    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-model", provider="mock")]
    
    async def get_model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model, provider="mock")
    
    async def get_available_models(self) -> List[ModelInfo]:
        return await self.list_models()
```

**Test Isolation Fixtures**:
```python
@pytest.fixture(autouse=True, scope="function")
def reset_global_state():
    """Reset all global state between tests."""
    from second_opinion.config import config_manager
    from second_opinion.utils.cost_tracking import get_cost_guard
    from second_opinion.utils.pricing import get_pricing_manager
    
    # Reset configuration
    config_manager.reset()
    
    # Reset cost tracking
    cost_guard = get_cost_guard()
    cost_guard.reset()
    
    # Reset pricing cache
    pricing_manager = get_pricing_manager()
    pricing_manager.clear_cache()
    
    yield
    
    # Cleanup after test
    # ... cleanup code
```

## Testing Strategy

### Test Categories & Coverage

**1. Unit Tests** (`tests/test_*/`):
- **Component isolation**: Mock all external dependencies
- **Security validation**: Malicious input handling
- **Configuration scenarios**: Environment variables, YAML configs, defaults
- **Edge cases**: Invalid inputs, network failures, cost limit scenarios

**2. Integration Tests** (`tests/integration/`):
- **MCP tool end-to-end**: Complete tool execution flows
- **Client integration**: Provider-specific behavior with mocked APIs
- **Cost tracking flows**: Budget reservation, recording, analytics
- **Error handling**: Graceful degradation and fallback scenarios

**3. MCP-Specific Tests** (`tests/test_mcp/`):
- **Server lifecycle**: Startup, shutdown, session management
- **Tool registration**: Parameter validation, documentation
- **Cost optimization**: Response reuse, budget integration
- **Error propagation**: MCP-specific error handling

### Mock Strategy for MCP Tools

**Provider Detection Mocking**:
```python
@pytest.fixture
def mock_provider_detection(monkeypatch):
    """Mock provider detection for controlled testing."""
    
    def mock_detect(model: str) -> str:
        if "local" in model.lower():
            return "lmstudio"
        return "openrouter"
    
    monkeypatch.setattr(
        "second_opinion.utils.client_factory.detect_model_provider",
        mock_detect
    )
```

**Cost Tracking Mocks**:
```python
@pytest.fixture
def mock_cost_guard():
    """Mock cost guard with realistic budget behavior."""
    
    class MockCostGuard:
        def __init__(self):
            self.reservations = {}
            self.recorded_costs = []
        
        async def check_and_reserve_budget(self, cost, operation, model, override=None):
            reservation_id = str(uuid4())
            self.reservations[reservation_id] = cost
            return BudgetCheck(approved=True, reservation_id=reservation_id)
        
        async def record_actual_cost(self, reservation_id, actual_cost, model, operation):
            self.recorded_costs.append({
                'reservation_id': reservation_id,
                'cost': actual_cost,
                'model': model,
                'operation': operation
            })
    
    return MockCostGuard()
```

## Deployment & Production

### MCP Server Configuration

**Claude Desktop Integration**:
```json
{
  "mcpServers": {
    "second-opinion": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/second-opinion",
        "run",
        "python",
        "-m",
        "second_opinion.mcp.server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-your-key-here"
      }
    }
  }
}
```

**Server Startup Command**:
```bash
# Production server
uv run python -m second_opinion.mcp.server

# Development server with debugging
uv run python -m second_opinion.mcp.server --dev
```

### Production Hardening

**Error Handling & Resilience**:
- Comprehensive input validation for all MCP tool parameters
- Graceful degradation when evaluation APIs fail
- Rate limiting and resource protection with cost guards
- Circuit breaker patterns for external API reliability

**Security & Performance**:
- Multi-layer input sanitization with context awareness
- Cost protection and budget enforcement at every operation
- Session security with automatic cleanup and timeouts
- Connection pooling and caching optimization for performance

### Monitoring & Observability

**Structured Logging**:
```python
logger.info(
    "MCP tool execution completed",
    extra={
        "tool_name": "second_opinion",
        "model": primary_model,
        "cost": float(actual_cost),
        "duration_ms": duration.total_seconds() * 1000,
        "session_id": session.session_id
    }
)
```

**Key Metrics to Track**:
- Tool usage patterns and popular model combinations
- Cost optimization impact (API cost savings through response reuse)
- Error rates and fallback activation frequency
- Response times and user satisfaction

## Blueprint for Future MCP Tools

### Reusable Implementation Template

Based on the successful `second_opinion` tool, future MCP tools should follow this proven pattern:

**1. Tool Registration**:
```python
@mcp.tool(name="new_tool", description="Clear, actionable description")
async def new_tool(
    # Required parameters first
    core_parameter: str,
    
    # Optional parameters with sensible defaults
    model: str | None = None,
    existing_response: str | None = None,
    context: str | None = None,
    cost_limit: float | None = None
) -> str:
```

**2. Standard Implementation Flow**:
- Parameter validation and defaults
- Configuration-driven model provider detection
- Cost estimation and budget check with hierarchy
- Response reuse optimization for cost savings
- Core tool logic with error handling
- Actual cost recording and session tracking
- Structured response formatting for MCP clients

**3. Testing Infrastructure**:
- Use existing mock utilities and fixtures
- Test all parameter combinations and edge cases
- Validate cost tracking and budget integration
- Ensure proper error handling and fallback behavior

### Next MCP Tools Pipeline

**Ready for Implementation**:
1. **`should_downgrade`** - Test cheaper alternatives using existing responses
2. **`should_upgrade`** - Evaluate premium model benefits with cost-benefit analysis
3. **`compare_responses`** - Detailed side-by-side analysis of provided responses
4. **`usage_analytics`** - Session and historical usage analysis

Each tool can leverage the complete infrastructure built for `second_opinion`, requiring only tool-specific logic while reusing:
- Session management and cost tracking
- Provider detection and client creation
- Response processing and formatting
- Error handling and fallback mechanisms
- Testing infrastructure and mock utilities

This blueprint ensures consistent quality, reliable cost optimization, and excellent developer experience across all MCP tools.