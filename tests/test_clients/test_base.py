"""
Tests for the abstract base client interface.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.second_opinion.clients.base import (
    AuthenticationError,
    BaseClient,
    ClientError,
    CostLimitExceededError,
    ModelInfo,
    RateLimitError,
    RetryableError,
)
from src.second_opinion.core.models import (
    Message,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from src.second_opinion.utils.sanitization import SecurityError, ValidationError


class MockClient(BaseClient):
    """Mock implementation of BaseClient for testing."""

    def __init__(self, **kwargs):
        super().__init__("mock-provider", **kwargs)
        self.complete_mock = AsyncMock()
        self.estimate_cost_mock = AsyncMock()
        self.get_available_models_mock = AsyncMock()

    async def complete(self, request):
        return await self.complete_mock(request)

    async def estimate_cost(self, request):
        return await self.estimate_cost_mock(request)

    async def get_available_models(self):
        # Check cache first (like a real implementation would)
        cached = await self._get_cached_models()
        if cached is not None:
            return cached

        # Get from mock and cache
        models = await self.get_available_models_mock()
        await self._cache_models(models)
        return models


class TestModelInfo:
    """Test ModelInfo class."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            name="gpt-4",
            provider="openai",
            input_cost_per_1k=Decimal("0.03"),
            output_cost_per_1k=Decimal("0.06"),
            max_tokens=4096,
            context_window=8192,
        )

        assert model.name == "gpt-4"
        assert model.provider == "openai"
        assert model.input_cost_per_1k == Decimal("0.03")
        assert model.output_cost_per_1k == Decimal("0.06")
        assert model.max_tokens == 4096
        assert model.context_window == 8192
        assert model.supports_system_messages is True
        assert model.supports_streaming is False

    def test_model_info_with_invalid_name(self):
        """Test ModelInfo with invalid model name."""
        with pytest.raises(SecurityError):
            ModelInfo(
                name="invalid<script>name",
                provider="test",
                input_cost_per_1k=Decimal("0.01"),
                output_cost_per_1k=Decimal("0.02"),
            )

    def test_model_info_repr(self):
        """Test ModelInfo string representation."""
        model = ModelInfo(
            name="claude-3",
            provider="anthropic",
            input_cost_per_1k=Decimal("0.015"),
            output_cost_per_1k=Decimal("0.075"),
        )

        repr_str = repr(model)
        assert "claude-3" in repr_str
        assert "anthropic" in repr_str
        assert "0.015" in repr_str


class TestClientErrors:
    """Test client error classes."""

    def test_client_error(self):
        """Test basic ClientError."""
        error = ClientError("Something went wrong", "test-provider", "test-model")
        assert error.message == "Something went wrong"
        assert error.provider == "test-provider"
        assert error.model == "test-model"
        assert "test-provider: Something went wrong" in str(error)
        assert "Model: test-model" in str(error)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key", "test-provider")
        assert isinstance(error, ClientError)
        assert error.message == "Invalid API key"

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Rate limit exceeded", "test-provider", retry_after=60)
        assert isinstance(error, ClientError)
        assert error.retry_after == 60

    def test_cost_limit_exceeded_error(self):
        """Test CostLimitExceededError."""
        error = CostLimitExceededError(
            "Cost limit exceeded",
            "test-provider",
            estimated_cost=Decimal("5.00"),
            cost_limit=Decimal("1.00"),
        )
        assert isinstance(error, ClientError)
        assert error.estimated_cost == Decimal("5.00")
        assert error.cost_limit == Decimal("1.00")


class TestBaseClient:
    """Test BaseClient abstract class."""

    def setup_method(self):
        self.client = MockClient(timeout=30, max_retries=2)

    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.provider_name == "mock-provider"
        assert self.client.timeout == 30
        assert self.client.max_retries == 2
        assert self.client._models_cache is None

    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        client = MockClient(api_key="test-key", timeout=60)
        assert client.api_key == "test-key"
        assert client.timeout == 60

    @pytest.mark.asyncio
    async def test_validate_request(self):
        """Test request validation and sanitization."""
        request = ModelRequest(
            model="  gpt-4  ",
            messages=[
                Message(role="user", content="  What is AI?  "),
                Message(role="assistant", content="AI is artificial intelligence."),
            ],
            system_prompt="  You are a helpful assistant.  ",
        )

        validated = await self.client.validate_request(request)

        assert validated.model == "gpt-4"
        assert validated.messages[0].content == "What is AI?"
        assert validated.messages[1].content == "AI is artificial intelligence."
        assert validated.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_validate_request_with_malicious_content(self):
        """Test request validation with malicious content."""
        request = ModelRequest(
            model="gpt-4",
            messages=[Message(role="user", content="<script>alert('xss')</script>")],
        )

        with pytest.raises(SecurityError):
            await self.client.validate_request(request)

    @pytest.mark.asyncio
    async def test_validate_request_with_api_key(self):
        """Test request validation with potential API key."""
        request = ModelRequest(
            model="gpt-4",
            messages=[
                Message(
                    role="user", content="My key is sk-1234567890abcdef1234567890abcdef"
                )
            ],
        )

        with pytest.raises(SecurityError):
            await self.client.validate_request(request)

    @pytest.mark.asyncio
    async def test_check_model_availability(self):
        """Test checking model availability."""
        mock_models = [
            ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06")),
            ModelInfo("claude-3", "anthropic", Decimal("0.015"), Decimal("0.075")),
        ]
        self.client.get_available_models_mock.return_value = mock_models

        assert await self.client.check_model_availability("gpt-4") is True
        assert await self.client.check_model_availability("nonexistent") is False

    @pytest.mark.asyncio
    async def test_check_model_availability_error(self):
        """Test checking model availability with client error."""
        self.client.get_available_models_mock.side_effect = ClientError(
            "API error", "test"
        )

        result = await self.client.check_model_availability("gpt-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test getting model information."""
        mock_models = [
            ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06")),
            ModelInfo("claude-3", "anthropic", Decimal("0.015"), Decimal("0.075")),
        ]
        self.client.get_available_models_mock.return_value = mock_models

        model_info = await self.client.get_model_info("gpt-4")
        assert model_info is not None
        assert model_info.name == "gpt-4"
        assert model_info.provider == "openai"

        missing_info = await self.client.get_model_info("nonexistent")
        assert missing_info is None

    @pytest.mark.asyncio
    async def test_model_caching(self):
        """Test model list caching."""
        mock_models = [ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06"))]
        self.client.get_available_models_mock.return_value = mock_models

        # First call should hit the API
        models1 = await self.client.get_available_models()
        assert self.client.get_available_models_mock.call_count == 1

        # Check that models are cached
        cached_models = await self.client._get_cached_models()
        assert cached_models is not None
        assert len(cached_models) == 1

        # Simulate cache expiry
        self.client._cache_timestamp = datetime.now(UTC).replace(year=2020)
        expired_cache = await self.client._get_cached_models()
        assert expired_cache is None

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test successful retry operation."""

        async def mock_operation():
            return "success"

        result = await self.client.retry_with_backoff(mock_operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_with_backoff_retryable_error(self):
        """Test retry with retryable error."""
        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary error", "test")
            return "success"

        # Speed up test by reducing delays
        self.client.base_delay = 0.01
        self.client.max_delay = 0.02

        result = await self.client.retry_with_backoff(mock_operation)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exhausted(self):
        """Test retry with exhausted attempts."""

        async def mock_operation():
            raise RetryableError("Persistent error", "test")

        # Speed up test
        self.client.base_delay = 0.01
        self.client.max_delay = 0.02

        with pytest.raises(RetryableError):
            await self.client.retry_with_backoff(mock_operation)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_non_retryable(self):
        """Test retry with non-retryable error."""

        async def mock_operation():
            raise AuthenticationError("Auth failed", "test")

        with pytest.raises(AuthenticationError):
            await self.client.retry_with_backoff(mock_operation)

    def test_calculate_token_cost(self):
        """Test token cost calculation."""
        model_info = ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06"))

        cost = self.client._calculate_token_cost(1000, 500, model_info)
        expected = (
            Decimal("1000") * Decimal("0.03") / 1000
            + Decimal("500") * Decimal("0.06") / 1000
        )
        assert cost == expected
        assert cost == Decimal("0.06")  # 0.03 + 0.03

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Test empty string
        assert self.client._estimate_tokens("") == 1

        # Test normal text
        text = "This is a test sentence with multiple words."
        tokens = self.client._estimate_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_create_error_response(self):
        """Test error response creation."""
        request = ModelRequest(
            model="gpt-4", messages=[Message(role="user", content="test")]
        )
        error = Exception("Test error")

        response = self.client._create_error_response(error, request)

        assert response.model == "gpt-4"
        assert response.provider == "mock-provider"
        assert "Error: Test error" in response.content
        assert response.cost_estimate == Decimal("0")
        assert response.metadata["error"] is True
        assert response.metadata["error_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager usage."""
        async with self.client as client:
            assert client is self.client

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.client)
        assert "MockClient" in repr_str
        assert "mock-provider" in repr_str

    def test_calculate_token_cost(self):
        """Test internal token cost calculation method."""
        model_info = ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06"))

        cost = self.client._calculate_token_cost(1000, 500, model_info)
        expected_input = Decimal("1000") * Decimal("0.03") / 1000  # $0.03
        expected_output = Decimal("500") * Decimal("0.06") / 1000  # $0.03
        expected_total = expected_input + expected_output

        assert cost == expected_total
        assert cost == Decimal("0.06")

    def test_estimate_tokens(self):
        """Test internal token estimation method."""
        # Test various text lengths
        assert self.client._estimate_tokens("") == 1  # Minimum 1 token
        assert self.client._estimate_tokens("Hello") == 1  # 5 chars = 1 token
        assert self.client._estimate_tokens("Hello world") == 2  # 11 chars = 2 tokens
        assert self.client._estimate_tokens("A" * 100) == 25  # 100 chars = 25 tokens

    def test_create_error_response(self):
        """Test error response creation method."""
        request = ModelRequest(
            model="gpt-4", messages=[Message(role="user", content="test message")]
        )
        error = Exception("Test error occurred")

        response = self.client._create_error_response(error, request)

        assert response.model == "gpt-4"
        assert response.provider == "mock-provider"
        assert "Error: Test error occurred" in response.content
        assert response.cost_estimate == Decimal("0")
        assert response.usage.total_tokens == 0
        assert response.metadata["error"] is True
        assert response.metadata["error_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_get_cached_models_empty(self):
        """Test cached models when cache is empty."""
        cached = await self.client._get_cached_models()
        assert cached is None

    @pytest.mark.asyncio
    async def test_cache_models(self):
        """Test model caching functionality."""
        models = [ModelInfo("gpt-4", "openai", Decimal("0.03"), Decimal("0.06"))]

        # Cache the models
        await self.client._cache_models(models)

        # Verify cache
        cached = await self.client._get_cached_models()
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].name == "gpt-4"


class TestClientIntegration:
    """Integration tests for client functionality."""

    @pytest.mark.asyncio
    async def test_complete_request_flow(self):
        """Test complete request processing flow."""
        client = MockClient()

        # Mock a successful response
        mock_response = ModelResponse(
            content="AI is artificial intelligence.",
            model="gpt-4",
            usage=TokenUsage(input_tokens=10, output_tokens=15, total_tokens=25),
            cost_estimate=Decimal("0.001"),
            provider="mock-provider",
        )
        client.complete_mock.return_value = mock_response

        request = ModelRequest(
            model="gpt-4", messages=[Message(role="user", content="What is AI?")]
        )

        # Validate request first
        validated_request = await client.validate_request(request)

        # Complete the request
        response = await client.complete(validated_request)

        assert response.content == "AI is artificial intelligence."
        assert response.model == "gpt-4"
        assert response.provider == "mock-provider"
        client.complete_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_estimation_flow(self):
        """Test cost estimation flow."""
        client = MockClient()
        client.estimate_cost_mock.return_value = Decimal("0.05")

        request = ModelRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Long prompt that costs money")],
        )

        cost = await client.estimate_cost(request)
        assert cost == Decimal("0.05")
        client.estimate_cost_mock.assert_called_once()

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_security_validation_in_complete_flow(self):
        """Test security validation in complete request flow."""
        client = MockClient()

        # Request with potential security issue
        malicious_request = ModelRequest(
            model="gpt-4",
            messages=[Message(role="user", content="<script>alert('xss')</script>")],
        )

        with pytest.raises(SecurityError):
            await client.validate_request(malicious_request)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_api_key_detection_in_validation(self):
        """Test API key detection during validation."""
        client = MockClient()

        request_with_key = ModelRequest(
            model="gpt-4",
            messages=[
                Message(
                    role="user",
                    content="My API key is sk-1234567890abcdef1234567890abcdef",
                )
            ],
        )

        with pytest.raises(SecurityError):
            await client.validate_request(request_with_key)
