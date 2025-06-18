# Second Opinion Implementation Guide

This document serves as the master implementation guide for the Second Opinion AI tool. It contains detailed specifications, design decisions, implementation patterns, and solutions to common challenges.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Implementation Phases](#implementation-phases)
3. [Component Specifications](#component-specifications)
4. [API Design Patterns](#api-design-patterns)
5. [Security Implementation](#security-implementation)
6. [Testing Strategy](#testing-strategy)
7. [Common Solutions](#common-solutions)
8. [Implementation Challenges](#implementation-challenges)
9. [Development Notes](#development-notes)

## Architecture Overview

### System Design Philosophy

**KISS Principles Applied:**
- Static configuration over dynamic complexity
- Explicit interfaces over implicit behavior  
- Predictable cost behavior over optimization
- Security by default at every layer

### Primary Model Detection Strategy

**CLI Mode:**
```bash
second-opinion --primary-model "anthropic/claude-3-5-sonnet" second_opinion "Question here"
```
- Global context set via command line argument
- All tools use this as the reference point for comparisons

**MCP Mode:**
- Detect primary model from MCP client context when available
- Fallback to configuration defaults when detection fails
- Maintain per-session state for consistency across tool calls

### Data Flow Architecture

```
User Input → Security Validation → Model Selection → API Calls → Response Processing → Cost Tracking → Storage
```

**Key Decision Points:**
1. **Input Validation**: Multi-layer sanitization before any external calls
2. **Model Selection**: Primary model + tool-specific comparison models
3. **Cost Protection**: Pre-flight estimation with hard limits
4. **Error Handling**: Graceful degradation with informative errors

## Implementation Phases

### Phase 1: Foundation (Days 1-7)

#### Days 1-2: Core Data Models
**File**: `src/second_opinion/core/models.py`

**Key Models:**
```python
class ModelRequest(BaseModel):
    """Standardized request format across all providers"""
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    
class ModelResponse(BaseModel):
    """Standardized response format with cost tracking"""
    content: str
    model: str
    usage: TokenUsage
    cost_estimate: Decimal
    provider: str
    metadata: Dict[str, Any] = {}

class EvaluationResult(BaseModel):
    """Results from model comparison or recommendation"""
    primary_response: str
    comparison_response: Optional[str] = None
    recommendation: RecommendationType
    confidence: float
    cost_analysis: CostAnalysis
    reasoning: str
```

**Implementation Challenges:**
- Different providers have different response formats
- Token counting varies between providers
- Cost calculation needs to be consistent

#### Days 3-4: Configuration System
**Files**: 
- `src/second_opinion/config/settings.py`
- `src/second_opinion/config/model_configs.py`

**Configuration Hierarchy:**
1. Command line arguments (highest priority)
2. Environment variables
3. User configuration files
4. Default configuration (lowest priority)

**Key Implementation Pattern:**
```python
@dataclass
class AppConfig:
    """Main application configuration with validation"""
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Load configuration with hierarchy and validation"""
        # Environment → User Config → Defaults
        
    def get_model_config(self, tool_name: str, primary_model: str) -> ModelConfig:
        """Get tool-specific configuration for given primary model"""
        # Dynamic configuration based on primary model
```

**Security Considerations:**
- Validate all configuration values before use
- Never log configuration containing potential secrets
- Fail fast on invalid configuration

#### Days 5-7: Security Infrastructure
**Files:**
- `src/second_opinion/config/security.py`
- `src/second_opinion/utils/sanitization.py`

**Input Sanitization Strategy:**
```python
class InputSanitizer:
    """Multi-layer input validation and sanitization"""
    
    def sanitize_prompt(self, prompt: str, context: SecurityContext) -> str:
        """Sanitize user prompts for safe API consumption"""
        # Remove potential API keys, excessive whitespace, etc.
        
    def validate_model_name(self, model_name: str) -> str:
        """Validate and normalize model names"""
        # Prevent injection attacks via model names
```

**Database Encryption:**
- SQLite with SQLCipher for encrypted storage
- Separate encryption keys for different data types
- Automatic key rotation support

### Phase 2: Client System (Days 8-14)

#### Abstract Client Interface
**File**: `src/second_opinion/clients/base.py`

```python
class BaseClient(ABC):
    """Abstract interface for all model providers"""
    
    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute model completion with standardized interface"""
        
    @abstractmethod
    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        """Estimate cost before making request"""
        
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models with capabilities"""
```

#### OpenRouter Implementation Challenges

**Model Name Normalization:**
- OpenRouter uses "provider/model" format
- Need mapping to canonical model names
- Handle model deprecation gracefully

**Dynamic Pricing Integration:**
```python
class OpenRouterClient(BaseClient):
    async def _get_current_pricing(self, model: str) -> PricingInfo:
        """Fetch real-time pricing from OpenRouter API"""
        # Cache pricing data with TTL
        # Handle pricing API failures gracefully
```

**Rate Limiting Strategy:**
- Per-model rate limits (different models have different limits)
- Exponential backoff with jitter
- Priority queuing for different tool types

#### LM Studio Implementation Considerations

**Local Server Integration:**
```python
class LMStudioClient(BaseClient):
    async def _check_server_health(self) -> bool:
        """Verify LM Studio server is running and responsive"""
        
    async def _get_loaded_model(self) -> Optional[str]:
        """Get currently loaded model from LM Studio"""
        # Handle case where no model is loaded
```

**Key Differences from Cloud Providers:**
- No cost tracking (local inference is free)
- Different error modes (server unreachable vs model not loaded)
- Limited model switching capabilities

### Phase 3: Core Logic (Days 15-21)

#### Prompt System Architecture
**Files:**
- `src/second_opinion/prompts/manager.py`
- `prompts/templates/*.txt`

**Template Management:**
```python
class PromptManager:
    """Manage prompt templates with parameter injection"""
    
    def load_template(self, template_name: str) -> PromptTemplate:
        """Load template with validation and caching"""
        
    def render_prompt(self, template: PromptTemplate, **kwargs) -> str:
        """Render template with type-safe parameter injection"""
        # Escape parameters based on target model requirements
```

**Model-Specific Optimization:**
- Different models perform better with different prompt styles
- Claude prefers detailed system prompts
- GPT models work well with conversational format
- Local models may need simpler prompts

#### Evaluation Engine Design
**File**: `src/second_opinion/core/evaluator.py`

**Response Comparison Algorithm:**
```python
class ResponseEvaluator:
    """Core logic for comparing and evaluating model responses"""
    
    async def compare_responses(
        self, 
        primary: ModelResponse, 
        comparison: ModelResponse,
        criteria: EvaluationCriteria
    ) -> ComparisonResult:
        """Compare two responses across multiple dimensions"""
        
    async def recommend_model_tier(
        self, 
        task: str, 
        current_model: str,
        current_response: Optional[str] = None
    ) -> RecommendationResult:
        """Recommend whether to upgrade/downgrade model"""
```

**Task Complexity Classification:**
- Use lightweight model to classify task complexity
- Categories: Simple, Moderate, Complex, Expert
- Cache classifications to avoid repeated analysis

### Phase 4: CLI Interface (Days 22-24)

#### Primary Model Context Management
**File**: `src/second_opinion/cli/main.py`

```python
@app.command()
def second_opinion(
    prompt: str,
    primary_model: str = typer.Option(None, "--primary-model"),
    cost_limit: float = typer.Option(0.10, "--cost-limit"),
    context: str = typer.Option(None, "--context")
):
    """Get a second opinion on an AI response"""
    # Set global context for primary model
    # Execute tool with cost protection
```

**User Experience Considerations:**
- Clear cost warnings before expensive operations
- Progress indicators for long-running operations
- Rich formatting for response comparisons

### Phase 5: MCP Integration (Days 25-28)

#### MCP Tool Implementation
**Files:**
- `src/second_opinion/mcp/server.py`
- `src/second_opinion/mcp/tools.py`

**Context Management Strategy:**
```python
class MCPSession:
    """Manage session state for MCP interactions"""
    
    def __init__(self):
        self.primary_model: Optional[str] = None
        self.conversation_context: List[Message] = []
        
    def detect_primary_model(self, request_context: Dict) -> Optional[str]:
        """Attempt to detect primary model from MCP context"""
        # Parse MCP client hints
        # Fallback to configuration defaults
```

**Tool Implementation Priority:**
1. `second_opinion` - Core functionality, straightforward implementation
2. `should_downgrade` - Cost optimization focus
3. `should_upgrade` - More complex due to cost implications
4. `compare_responses` - Requires sophisticated analysis
5. `usage_analytics` - Needs historical data processing

## Component Specifications

### Configuration Management

**Model Configuration Structure:**
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

**Dynamic Configuration Loading:**
- Primary model affects which comparison models are selected
- Tool-specific overrides for different use cases
- Runtime validation of model availability

### Cost Protection System

**Multi-Layer Cost Controls:**
1. **Pre-request Estimation**: Calculate cost before API call
2. **Per-request Limits**: Hard cap on individual tool usage
3. **Daily/Monthly Budgets**: Aggregate spending controls
4. **Model Fallbacks**: Cheaper alternatives when limits exceeded

**Implementation Pattern:**
```python
class CostGuard:
    async def check_and_reserve_budget(
        self, 
        estimated_cost: Decimal,
        operation_type: str
    ) -> BudgetCheck:
        """Reserve budget for operation or raise BudgetExceeded"""
        
    async def record_actual_cost(
        self, 
        reservation_id: str,
        actual_cost: Decimal
    ):
        """Record actual cost and adjust budgets"""
```

### Storage System

**Database Schema Design:**
```sql
-- Conversation tracking with encryption
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    primary_model TEXT NOT NULL,
    tool_used TEXT NOT NULL,
    encrypted_content BLOB NOT NULL,
    cost_total DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage analytics
CREATE TABLE usage_stats (
    date DATE,
    model TEXT,
    tool TEXT,
    request_count INTEGER,
    total_cost DECIMAL(10,4),
    PRIMARY KEY (date, model, tool)
);
```

**Privacy Considerations:**
- Encrypt all conversation content
- Configurable data retention periods
- User-controlled data export and deletion

## API Design Patterns

### Error Handling Strategy

**Error Categories:**
1. **User Errors**: Invalid input, configuration issues
2. **Provider Errors**: API failures, rate limits, authentication
3. **System Errors**: Database issues, configuration problems
4. **Cost Errors**: Budget exceeded, estimation failures

**Error Handling Pattern:**
```python
class SecondOpinionError(Exception):
    """Base exception with user-friendly messaging"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        
    @property
    def user_message(self) -> str:
        """User-friendly error message without internal details"""
```

### Request/Response Patterns

**Unified Request Processing:**
```python
async def process_tool_request(
    tool_name: str,
    request_data: Dict[str, Any],
    context: RequestContext
) -> ToolResponse:
    """Standard request processing pipeline"""
    
    # 1. Input validation and sanitization
    validated_input = await validate_input(request_data, tool_name)
    
    # 2. Cost estimation and budget check
    estimated_cost = await estimate_request_cost(validated_input, context)
    await cost_guard.check_budget(estimated_cost)
    
    # 3. Execute tool logic
    result = await execute_tool(tool_name, validated_input, context)
    
    # 4. Record actual cost and update analytics
    await record_usage(result.actual_cost, context)
    
    return result
```

### Configuration Loading Pattern

**Hierarchical Configuration:**
```python
def load_configuration() -> AppConfig:
    """Load configuration with proper hierarchy and validation"""
    
    config = {}
    
    # Load defaults
    config.update(load_default_config())
    
    # Override with user config
    if user_config_exists():
        config.update(load_user_config())
    
    # Override with environment variables
    config.update(load_env_config())
    
    # Override with CLI arguments
    config.update(load_cli_config())
    
    # Validate final configuration
    return AppConfig.model_validate(config)
```

## Security Implementation

### Input Validation Strategy

**Multi-Layer Validation:**
1. **Schema Validation**: Pydantic models ensure type safety
2. **Content Filtering**: Remove potential API keys, malicious content
3. **Size Limits**: Prevent resource exhaustion attacks
4. **Provider-Specific**: Handle provider-specific requirements

```python
class SecurityValidator:
    async def validate_prompt(self, prompt: str) -> str:
        """Validate and sanitize user prompts"""
        
        # Check for potential API keys
        if self._contains_api_key_pattern(prompt):
            raise SecurityError("Potential API key detected in prompt")
            
        # Remove excessive whitespace
        prompt = self._normalize_whitespace(prompt)
        
        # Check size limits
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValidationError(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH}")
            
        return prompt
```

### API Key Management

**Secure Key Handling:**
```python
class KeyManager:
    def __init__(self):
        self._keys: Dict[str, str] = {}
        self._load_keys_from_env()
        
    def get_key(self, provider: str) -> str:
        """Get API key for provider with validation"""
        key = self._keys.get(provider)
        if not key:
            raise ConfigurationError(f"No API key configured for {provider}")
            
        # Validate key format
        if not self._validate_key_format(provider, key):
            raise SecurityError(f"Invalid API key format for {provider}")
            
        return key
        
    def _validate_key_format(self, provider: str, key: str) -> bool:
        """Validate API key format for specific provider"""
        patterns = {
            "openrouter": r"^sk-or-.*",
            "anthropic": r"^sk-ant-.*",
            "openai": r"^sk-.*"
        }
        pattern = patterns.get(provider)
        return bool(pattern and re.match(pattern, key))
```

### Database Encryption

**Encryption Strategy:**
```python
class EncryptedStorage:
    def __init__(self, encryption_key: str):
        self.cipher_suite = Fernet(encryption_key.encode())
        
    async def store_conversation(self, conversation: Conversation) -> str:
        """Store conversation with encryption"""
        
        # Serialize conversation data
        data = conversation.model_dump_json()
        
        # Encrypt sensitive content
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        
        # Store with metadata
        conversation_id = str(uuid4())
        await self.db.execute(
            "INSERT INTO conversations (id, primary_model, tool_used, encrypted_content, cost_total) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, conversation.primary_model, conversation.tool, encrypted_data, conversation.cost)
        )
        
        return conversation_id
```

## Testing Strategy

### Test Categories

**1. Unit Tests** (`tests/test_*/`)
- Individual component functionality
- Mocked external dependencies
- Security validation logic
- Configuration loading and validation

**2. Integration Tests** (`tests/integration/`)
- Client integration with mocked APIs
- MCP tool end-to-end functionality
- Database operations with encryption
- Cost tracking across multiple operations

**3. Security Tests** (`tests/security/`)
- Input sanitization with malicious inputs
- API key validation and protection
- Encryption/decryption correctness
- Cost limit enforcement

**4. Performance Tests** (`tests/performance/`)
- Response time under load
- Memory usage with large conversations
- Database query performance
- Concurrent request handling

### Mock Strategies

**Provider API Mocking:**
```python
class MockOpenRouterClient(BaseClient):
    """Mock OpenRouter client for testing"""
    
    def __init__(self, mock_responses: Dict[str, ModelResponse]):
        self.mock_responses = mock_responses
        self.call_count = 0
        
    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        return self.mock_responses.get(request.model, self._default_response())
```

**Security Test Scenarios:**
- SQL injection attempts in prompts
- API key extraction attempts
- Buffer overflow attempts with large inputs
- Cost limit bypass attempts

## Common Solutions & Patterns

### Configuration Validation Pattern

```python
def validate_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Validate model configuration with helpful error messages"""
    
    try:
        return ModelConfig.model_validate(config)
    except ValidationError as e:
        # Convert Pydantic errors to user-friendly messages
        errors = []
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            message = error["msg"]
            errors.append(f"Configuration error in {field}: {message}")
        
        raise ConfigurationError("Invalid configuration:\n" + "\n".join(errors))
```

### Cost Calculation Utilities

```python
def calculate_request_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_data: Dict[str, PricingInfo]
) -> Decimal:
    """Calculate cost for a model request"""
    
    pricing = pricing_data.get(model)
    if not pricing:
        # Conservative estimate for unknown models
        return Decimal("0.10")
        
    input_cost = Decimal(input_tokens) * pricing.input_cost_per_1k / 1000
    output_cost = Decimal(output_tokens) * pricing.output_cost_per_1k / 1000
    
    return input_cost + output_cost
```

### Retry Logic with Exponential Backoff

```python
async def retry_with_backoff(
    operation: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """Execute operation with exponential backoff retry logic"""
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except RetryableError as e:
            if attempt == max_retries:
                raise e
                
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
```

## Implementation Challenges & Solutions

### Challenge: Model Provider Inconsistencies

**Issue**: Different providers have different API formats, error codes, and capabilities.

**Solution**: 
- Abstract client interface with standardized request/response format
- Provider-specific adapters handling the differences
- Comprehensive error mapping to unified error types

**Implementation Notes**:
- OpenRouter sometimes returns non-standard error codes
- Anthropic requires specific message format (system vs user roles)
- Local models may not support all parameters (temperature, max_tokens)

### Challenge: Cost Prediction Accuracy

**Issue**: Accurate cost prediction is difficult, especially for new models with changing pricing.

**Solution**:
- Conservative cost estimates with safety margins
- Real-time pricing updates from provider APIs
- User-configurable cost limits with clear warnings

**Implementation Notes**:
- Cache pricing data with short TTL (5 minutes)
- Fall back to conservative estimates when pricing unavailable
- Track prediction accuracy and adjust algorithms

### Challenge: MCP Context Detection

**Issue**: MCP doesn't always provide clear model context, making primary model detection difficult.

**Solution**:
- Multi-layered detection strategy:
  1. Parse MCP client metadata when available
  2. Infer from conversation patterns
  3. Fall back to user configuration defaults
  4. Allow explicit model specification

**Implementation Notes**:
- Different MCP clients provide different levels of context
- Some clients don't expose the model being used
- Need graceful fallback strategy for all cases

### Challenge: Local vs Cloud Model Capabilities

**Issue**: Local models (LM Studio) often have different capabilities than cloud models.

**Solution**:
- Capability detection and feature flags
- Graceful degradation for unsupported features
- Clear user communication about limitations

**Implementation Notes**:
- Local models may not support system messages
- Token counting may be inconsistent
- Some models don't support temperature control

## Development Notes

### Code Organization Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Dependency Injection**: Easy testing and component swapping
3. **Interface-First Design**: Define interfaces before implementations
4. **Security by Default**: Security considerations in every component

### Performance Considerations

- **Async/Await**: All I/O operations are asynchronous
- **Connection Pooling**: Reuse HTTP connections for API calls
- **Caching**: Cache configuration, pricing data, and model capabilities
- **Resource Limits**: Prevent memory exhaustion with large responses

### Monitoring and Observability

```python
# Structured logging for operations
logger.info(
    "Model request completed",
    extra={
        "model": request.model,
        "tokens_used": response.usage.total_tokens,
        "cost": float(response.cost_estimate),
        "duration_ms": duration.total_seconds() * 1000
    }
)
```

### Future Extension Points

1. **Plugin System**: Interface for adding new model providers
2. **Custom Evaluators**: User-defined comparison criteria
3. **Export Formats**: Additional data export options
4. **Advanced Analytics**: ML-based usage optimization

This implementation guide will be updated throughout development with new insights, solutions, and patterns discovered during implementation.