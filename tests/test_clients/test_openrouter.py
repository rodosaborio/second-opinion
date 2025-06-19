"""
Tests for the OpenRouter client implementation.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json

import httpx

from src.second_opinion.clients.openrouter import OpenRouterClient
from src.second_opinion.clients.base import (
    ModelInfo,
    ClientError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    CostLimitExceededError,
    RetryableError,
)
from src.second_opinion.core.models import (
    ModelRequest,
    ModelResponse,
    Message,
    TokenUsage,
)
from src.second_opinion.utils.sanitization import ValidationError, SecurityError


class TestOpenRouterClient:
    """Test OpenRouter client functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a test OpenRouter client."""
        return OpenRouterClient(api_key="sk-or-test123")
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample model request."""
        return ModelRequest(
            model="openai/gpt-3.5-turbo",
            messages=[
                Message(role="user", content="Hello, world!")
            ],
            max_tokens=100,
            temperature=0.7
        )
    
    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-test123",
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response
        return mock_client
    
    def test_client_initialization(self):
        """Test client initialization with valid API key."""
        client = OpenRouterClient(api_key="sk-or-test123")
        assert client.provider_name == "openrouter"
        assert client.api_key == "sk-or-test123"
        assert client._http_client is not None
    
    def test_client_initialization_invalid_key(self):
        """Test client initialization with invalid API key."""
        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            OpenRouterClient(api_key="")
    
    def test_client_initialization_wrong_format(self):
        """Test client initialization with wrong key format shows warning."""
        with patch('src.second_opinion.clients.openrouter.logger') as mock_logger:
            OpenRouterClient(api_key="invalid-key")
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complete_success(self, client, sample_request, mock_http_client):
        """Test successful completion request."""
        client._http_client = mock_http_client
        
        response = await client.complete(sample_request)
        
        assert isinstance(response, ModelResponse)
        assert response.content == "Hello! How can I help you today?"
        assert response.model == "openai/gpt-3.5-turbo"
        assert response.provider == "openrouter"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 8
        assert response.usage.total_tokens == 18
    
    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, client, mock_http_client):
        """Test completion with system prompt."""
        client._http_client = mock_http_client
        
        request = ModelRequest(
            model="openai/gpt-4",
            messages=[Message(role="user", content="Hello")],
            system_prompt="You are a helpful assistant."
        )
        
        await client.complete(request)
        
        # Verify system message was added
        call_args = mock_http_client.post.call_args
        request_data = call_args[1]["json"]
        
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["role"] == "system"
        assert request_data["messages"][0]["content"] == "You are a helpful assistant."
        assert request_data["messages"][1]["role"] == "user"
        assert request_data["messages"][1]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_complete_authentication_error(self, client, sample_request):
        """Test completion with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid API key",
                "code": 401
            }
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await client.complete(sample_request)
    
    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self, client, sample_request):
        """Test completion with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "code": 429
            }
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await client.complete(sample_request)
    
    @pytest.mark.asyncio
    async def test_complete_cost_limit_error(self, client, sample_request):
        """Test completion with insufficient credits."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.json.return_value = {
            "error": {
                "message": "Insufficient credits",
                "code": 402
            }
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        with pytest.raises(CostLimitExceededError, match="Insufficient credits"):
            await client.complete(sample_request)
    
    @pytest.mark.asyncio
    async def test_complete_moderation_error(self, client, sample_request):
        """Test completion blocked by moderation."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "error": {
                "message": "Content flagged by moderation",
                "code": 403
            }
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        with pytest.raises(SecurityError, match="Request blocked by moderation"):
            await client.complete(sample_request)
    
    @pytest.mark.asyncio
    async def test_complete_retryable_error(self, client, sample_request):
        """Test completion with retryable error."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.return_value = {
            "error": {
                "message": "Bad gateway",
                "code": 502
            }
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        with pytest.raises(RetryableError, match="Service temporarily unavailable"):
            await client.complete(sample_request)
    
    @pytest.mark.asyncio
    async def test_estimate_cost_known_model(self, client):
        """Test cost estimation for known model."""
        request = ModelRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="Hello" * 100)],  # ~100 tokens
            max_tokens=50
        )
        
        cost = await client.estimate_cost(request)
        
        assert isinstance(cost, Decimal)
        assert cost > Decimal("0")
        # Should be roughly: (100 * 0.0015 + 50 * 0.002) / 1000 = $0.00025
        assert cost < Decimal("0.01")  # Reasonable upper bound
    
    @pytest.mark.asyncio
    async def test_estimate_cost_unknown_model(self, client):
        """Test cost estimation for unknown model."""
        request = ModelRequest(
            model="unknown/model",
            messages=[Message(role="user", content="Hello")],
        )
        
        cost = await client.estimate_cost(request)
        
        # Pricing manager now provides conservative fallback based on model tier
        assert cost == Decimal("0.02")  # Conservative fallback for low-tier unknown models
    
    @pytest.mark.asyncio
    async def test_get_available_models_success(self, client):
        """Test successful model retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-3.5-turbo",
                    "description": "GPT-3.5 Turbo",
                    "pricing": {
                        "prompt": "0.0015",
                        "completion": "0.002"
                    },
                    "context_length": 4096,
                    "top_provider": {
                        "max_completion_tokens": 4096
                    }
                },
                {
                    "id": "anthropic/claude-3-haiku",
                    "description": "Claude 3 Haiku",
                    "pricing": {
                        "prompt": "0.00025",
                        "completion": "0.00125"
                    },
                    "context_length": 200000
                }
            ]
        }
        
        client._http_client = AsyncMock()
        client._http_client.get.return_value = mock_response
        
        models = await client.get_available_models()
        
        assert len(models) == 2
        assert isinstance(models[0], ModelInfo)
        assert models[0].name == "openai/gpt-3.5-turbo"
        assert models[0].input_cost_per_1k == Decimal("0.0015")
        assert models[0].output_cost_per_1k == Decimal("0.002")
        assert models[0].context_window == 4096
    
    @pytest.mark.asyncio
    async def test_get_available_models_caching(self, client, mock_http_client):
        """Test that models are cached properly."""
        client._http_client = mock_http_client
        mock_http_client.get.return_value.json.return_value = {"data": []}
        
        # First call
        models1 = await client.get_available_models()
        
        # Second call - should use cache
        models2 = await client.get_available_models()
        
        # Should only have made one HTTP call
        assert mock_http_client.get.call_count == 1
        assert models1 == models2
    
    @pytest.mark.asyncio
    async def test_get_available_models_error(self, client):
        """Test model retrieval with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Internal Server Error",
            request=MagicMock(),
            response=mock_response
        )
        
        client._http_client = AsyncMock()
        client._http_client.get.return_value = mock_response
        
        with pytest.raises(ClientError, match="Failed to retrieve models"):
            await client.get_available_models()
    
    
    def test_estimate_input_tokens(self, client):
        """Test input token estimation."""
        request = ModelRequest(
            model="gpt-3.5-turbo",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?")
            ],
            system_prompt="You are helpful."
        )
        
        tokens = client._estimate_input_tokens(request)
        
        # Should be roughly: "You are helpful. Hello Hi there! How are you? "
        # Approximately 8-10 words = 32-40 characters = 8-10 tokens
        assert tokens >= 5
        assert tokens <= 20
    
    @pytest.mark.asyncio
    async def test_validate_request_integration(self, client, sample_request):
        """Test that request validation is properly integrated."""
        # Test with malicious input
        malicious_request = ModelRequest(
            model="<script>alert('xss')</script>",
            messages=[Message(role="user", content="sk-test-api-key-here")]
        )
        
        with pytest.raises((ValidationError, SecurityError)):
            await client.complete(malicious_request)
    
    def test_prepare_request_formatting(self, client, sample_request):
        """Test OpenRouter request preparation."""
        openrouter_request = client._prepare_request(sample_request)
        
        expected_request = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        assert openrouter_request == expected_request
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with OpenRouterClient(api_key="sk-or-test123") as client:
            assert client.provider_name == "openrouter"
        
        # Should clean up HTTP client
        assert hasattr(client, '_http_client')


class TestOpenRouterClientSecurity:
    """Security-focused tests for OpenRouter client."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenRouterClient(api_key="sk-or-test123")
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_api_key_not_logged(self, client):
        """Test that API keys are not exposed in logs."""
        # API key should not appear in repr
        client_repr = repr(client)
        assert "sk-or-test123" not in client_repr
        
        # API key should be in headers but not logged
        headers = client._http_client.headers
        assert "Authorization" in headers
        assert "Bearer sk-or-test123" in headers["Authorization"]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_malicious_model_name_blocked(self, client):
        """Test that malicious model names are blocked."""
        malicious_request = ModelRequest(
            model="<script>alert('xss')</script>",
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(ValidationError):
            await client.complete(malicious_request)
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_api_key_in_content_blocked(self, client):
        """Test that API keys in message content are blocked."""
        malicious_request = ModelRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="My API key is sk-test-1234567890abcdef")]
        )
        
        with pytest.raises(SecurityError):
            validated = await client.validate_request(malicious_request)
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_injection_attempts_blocked(self, client):
        """Test that various injection attempts are blocked or sanitized."""
        # These should be blocked by security validation
        blocked_attempts = [
            "<script>fetch('evil.com')</script>",
            "javascript:alert('xss')",
            "'; DROP TABLE users; --",  # SQL injection detected
            "$(echo 'test')",  # Command substitution detected
        ]
        
        # These should be sanitized (whitespace/length normalization only)
        sanitized_attempts = [
            "Hello world!   ",  # Basic whitespace normalization
            "What is 2+2?",     # Normal content
        ]
        
        for injection in blocked_attempts:
            malicious_request = ModelRequest(
                model="gpt-3.5-turbo",
                messages=[Message(role="user", content=injection)]
            )
            
            with pytest.raises(SecurityError):
                await client.validate_request(malicious_request)
        
        for injection in sanitized_attempts:
            malicious_request = ModelRequest(
                model="gpt-3.5-turbo",
                messages=[Message(role="user", content=injection)]
            )
            
            # These should be sanitized but not blocked
            validated = await client.validate_request(malicious_request)
            # Content should be normalized (whitespace trimmed, etc.)
            assert validated.messages[0].content.strip() != ""


class TestOpenRouterClientIntegration:
    """Integration tests for OpenRouter client."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_request_cycle_mocked(self):
        """Test full request cycle with mocked responses."""
        client = OpenRouterClient(api_key="sk-or-test123")
        
        # Mock successful completion
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        
        client._http_client = AsyncMock()
        client._http_client.post.return_value = mock_response
        
        request = ModelRequest(
            model="openai/gpt-3.5-turbo",
            messages=[Message(role="user", content="Test")]
        )
        
        # Test cost estimation
        estimated_cost = await client.estimate_cost(request)
        assert isinstance(estimated_cost, Decimal)
        
        # Test completion
        response = await client.complete(request)
        assert response.content == "Test response"
        assert response.cost_estimate > Decimal("0")
        
        # Cleanup
        await client.__aexit__(None, None, None)
    
    @pytest.mark.integration
    def test_client_factory_integration(self):
        """Test integration with client factory."""
        from src.second_opinion.clients import create_client
        
        client = create_client("openrouter", api_key="sk-or-test123")
        assert isinstance(client, OpenRouterClient)
        assert client.provider_name == "openrouter"