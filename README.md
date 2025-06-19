# Second Opinion

An AI tool that provides "second opinion" functionality for AI responses via MCP (Model Context Protocol). Get smarter AI usage by comparing responses across different models, finding cost-effective alternatives, and tracking your usage patterns.

## =� Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone and install**:
   ```bash
   git clone <repository-url>
   cd second-opinion
   uv sync --dev
   ```

2. **Set up your API keys** (see [API Key Setup](#-api-key-setup) below)

3. **Test the installation**:
   ```bash
   uv run second-opinion --help
   ```

4. **Validate your setup** (optional):
   ```bash
   uv run python setup_guide.py
   ```

## = API Key Setup

### Getting Your OpenRouter API Key

1. **Sign up at [OpenRouter](https://openrouter.ai/)**
2. **Get your API key** from [OpenRouter Keys](https://openrouter.ai/keys)
3. **Set up your environment** (see below)

### Environment Configuration

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your OpenRouter API key**:
   ```bash
   # Required: OpenRouter API Key
   OPENROUTER_API_KEY=sk-or-your_actual_api_key_here
   
   # Recommended: Set conservative cost limits for testing
   DAILY_COST_LIMIT=1.00
   MONTHLY_COST_LIMIT=10.00
   DEFAULT_COST_LIMIT=0.05
   
   # Optional: Generate encryption keys for data security
   # Run: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   DATABASE_ENCRYPTION_KEY=your_generated_encryption_key_here
   SESSION_ENCRYPTION_KEY=your_session_encryption_key_here
   ```

3. **Verify your setup**:
   ```bash
   # Run the setup validation script
   uv run python setup_guide.py
   
   # Or test directly with a simple prompt
   uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" "What's 2+2?"
   ```

### Security Best Practices

-  **Never commit `.env` to version control** (it's in `.gitignore`)
-  **Use different encryption keys** for database and sessions
-  **Set conservative cost limits** when starting out
-  **Regularly rotate your API keys**
-  **Monitor your usage** through OpenRouter dashboard

## =� Cost Management

Second Opinion includes comprehensive cost tracking and budget protection:

### Built-in Cost Protection
- **Per-request limits**: Prevent expensive single requests
- **Daily/monthly budgets**: Set spending caps
- **Real-time cost estimates**: Know costs before making requests
- **Model recommendations**: Get suggestions for cost-effective alternatives

### Cost Configuration
```bash
# Set in your .env file:
DEFAULT_COST_LIMIT=0.05        # Max cost per request ($0.05)
DAILY_COST_LIMIT=1.00          # Daily limit ($1.00)
MONTHLY_COST_LIMIT=10.00       # Monthly limit ($10.00)
```

### Cost Monitoring
```bash
# View your current usage and limits
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" \
  --verbose "Tell me about cost tracking"
```

## =� Usage Examples

### Basic Model Comparison
```bash
# Compare responses from two models
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --comparison-model "openai/gpt-4o" \
  "Explain quantum computing in simple terms"
```

### Advanced Usage

#### Save API Calls with Existing Responses
```bash
# Use existing response to avoid re-generating from primary model
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --existing-response "Quantum computing uses quantum mechanics..." \
  --comparison-model "openai/gpt-4o" \
  "Explain quantum computing in simple terms"
```

#### Multiple Comparison Models
```bash
# Compare against multiple models at once
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --comparison-model "openai/gpt-4o" \
  --comparison-model "google/gemini-pro" \
  --comparison-model "meta-llama/llama-3.1-405b" \
  "Analyze the economic impact of climate change"
```

#### Verbose Mode for Detailed Analysis
```bash
# Show full responses and detailed analysis
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --comparison-model "openai/gpt-4o" \
  --verbose \
  "Complex reasoning task here"
```

#### Context-Aware Recommendations
```bash
# Provide context for better model recommendations
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --context "This is for academic research requiring high accuracy" \
  "Analyze statistical significance in clinical trials"
```

### Workflow Examples

#### Cost-Conscious Development
```bash
# Start with a cheaper model, get recommendations for upgrades
uv run second-opinion \
  --primary-model "openai/gpt-3.5-turbo" \
  --comparison-model "anthropic/claude-3-5-sonnet" \
  "Write a Python function to sort a list"
```

#### Quality-First Analysis
```bash
# Use premium model, get cost-effective alternatives
uv run second-opinion \
  --primary-model "anthropic/claude-3-5-sonnet" \
  --comparison-model "openai/gpt-4o-mini" \
  --comparison-model "google/gemini-flash" \
  "Perform complex data analysis"
```

## <� Development

### Setup Development Environment
```bash
# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Fast development testing
uv run pytest

# Full coverage testing
uv run pytest --cov=second_opinion --cov-report=html --cov-fail-under=85

# Run specific test categories
uv run pytest -m security     # Security tests
uv run pytest -m integration  # Integration tests
uv run pytest -m "not slow"   # Skip slow tests
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Run all quality checks
uv run black . && uv run ruff check . --fix && uv run mypy src/
```

## = MCP Integration

Second Opinion can run as an MCP server for integration with Claude Desktop and other MCP clients:

```bash
# Start MCP server
uv run python -m second_opinion.mcp.server

# Development mode with auto-reload
uv run python -m second_opinion.mcp.server --dev
```

## =� Security Features

- **Input sanitization**: Prevents prompt injection attacks
- **Response filtering**: Removes sensitive information from outputs
- **API key validation**: Ensures proper key formats and security
- **Rate limiting**: Prevents abuse and unexpected charges
- **Encryption**: Protects stored data and sessions
- **Cost guards**: Prevents runaway spending

## =� Available Models

Second Opinion supports 1,100+ models through OpenRouter, including:

### Popular Models
- **Anthropic**: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`
- **Google**: `google/gemini-pro`, `google/gemini-flash`
- **Meta**: `meta-llama/llama-3.1-405b`, `meta-llama/llama-3.1-70b`
- **Cohere**: `cohere/command-r-plus`, `cohere/command-r`

### Model Selection Tips
- **For reasoning tasks**: Use `anthropic/claude-3-5-sonnet` or `openai/gpt-4o`
- **For cost-effective tasks**: Try `openai/gpt-4o-mini` or `google/gemini-flash`
- **For creative writing**: Consider `anthropic/claude-3-haiku` or `meta-llama/llama-3.1-70b`
- **For code generation**: Use `anthropic/claude-3-5-sonnet` or `openai/gpt-4o`

## =� Troubleshooting

### Common Issues

#### "API key not found" Error
```bash
# Check your .env file exists and has the correct key
cat .env | grep OPENROUTER_API_KEY

# Verify the key format (should start with sk-or-)
echo $OPENROUTER_API_KEY
```

#### "Cost limit exceeded" Error
```bash
# Check your current limits
grep -E "(COST_LIMIT|DAILY_LIMIT|MONTHLY_LIMIT)" .env

# Increase limits temporarily for testing
# Edit .env and increase DEFAULT_COST_LIMIT
```

#### "Model not found" Error
```bash
# Use a known model name from OpenRouter
uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" "test"

# Check available models at: https://openrouter.ai/models
```

#### Import/Module Errors
```bash
# Reinstall dependencies
uv sync --dev

# Check Python version (requires 3.12+)
python --version
```

### Setup Validation Script

Use the built-in setup validation script to diagnose configuration issues:

```bash
uv run python setup_guide.py
```

This script will:
- ✅ Check if `.env` file exists and is properly configured
- ✅ Validate API key format and presence
- ✅ Test configuration loading
- ✅ Generate encryption keys if needed
- ✅ Optionally test API connectivity

### Getting Help

- **Setup Validation**: `uv run python setup_guide.py`
- **CLI Help**: `uv run second-opinion --help`
- **Configuration Issues**: Check `CLAUDE.md` for detailed configuration guide
- **Bug Reports**: Create an issue with detailed error messages and steps to reproduce

## =� Configuration Options

### Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# API Keys
OPENROUTER_API_KEY=sk-or-your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here  # Optional
OPENAI_API_KEY=sk-your_key_here         # Optional

# Cost Management
DEFAULT_COST_LIMIT=0.05     # Per-request limit in USD
DAILY_COST_LIMIT=1.00       # Daily spending limit
MONTHLY_COST_LIMIT=10.00    # Monthly spending limit

# Performance
REQUEST_TIMEOUT=30          # Request timeout in seconds
MAX_CONCURRENT_REQUESTS=5   # Concurrent request limit
RATE_LIMIT_PER_MINUTE=60   # API rate limit

# Security
ENABLE_INPUT_SANITIZATION=true    # Sanitize inputs
ENABLE_RESPONSE_FILTERING=true    # Filter responses
DATABASE_ENCRYPTION_KEY=...       # Database encryption
SESSION_ENCRYPTION_KEY=...        # Session encryption
```

### YAML Configuration

For advanced configuration, create `config/settings.yaml`:

```yaml
cost_management:
  default_per_request_limit: "0.10"
  daily_limit: "5.00"
  monthly_limit: "50.00"
  warning_threshold: 0.80

security:
  input_sanitization: true
  response_filtering: true
  rate_limit_per_minute: 100
  max_concurrent_requests: 10

analytics:
  enabled: true
  anonymize_data: true
  retention_days: 90
```

## > Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

## =� License

[Add your license information here]

---

**<� Ready to get smarter AI responses?** Start with the [Quick Start](#-quick-start) guide above!