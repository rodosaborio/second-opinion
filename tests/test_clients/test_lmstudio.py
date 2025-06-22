"""
Tests for LM Studio client implementation.

This module contains comprehensive tests for the LMStudioClient,
including unit tests with mocked responses and integration scenarios.
"""

import json
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from second_opinion.clients.base import (
    ClientError,
    ModelInfo,
    RetryableError,
)
from second_opinion.clients.lmstudio import LMStudioClient
from second_opinion.core.models import Message, ModelRequest, ModelResponse


class TestLMStudioClient:
    """Test cases for LM Studio client."""

    @pytest.fixture
    def client(self):
        """Create LM Studio client for testing."""
        return LMStudioClient(base_url="http://localhost:1234")

    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client."""
        mock_client = AsyncMock()
        return mock_client

    @pytest.fixture
    def sample_request(self):
        """Sample model request for testing."""
        return ModelRequest(
            model="qwen3-8b-mlx",
            messages=[Message(role="user", content="What is 2+2?")],
            max_tokens=100,
            temperature=0.7,
        )

    @pytest.fixture
    def mock_models_response(self):
        """Mock response for /v1/models endpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-8b-mlx",
                    "object": "model",
                    "created": 1640995200,
                    "owned_by": "local",
                    "context_length": 8192,
                },
                {
                    "id": "qwen3-0.6b-mlx",
                    "object": "model",
                    "created": 1640995200,
                    "owned_by": "local",
                    "context_length": 4096,
                },
            ],
        }

    @pytest.fixture
    def mock_completion_response(self):
        """Mock response for chat completion."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1640995200,
            "model": "qwen3-8b-mlx",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "2+2 equals 4."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }

    def test_init_default_base_url(self):
        """Test client initialization with default base URL."""
        client = LMStudioClient()
        assert (
            client.base_url == "http://localhost:1234/v1"
        )  # Updated to include /v1 endpoint
        assert client.provider_name == "lmstudio"
        assert client.api_key is None

    def test_init_custom_base_url(self):
        """Test client initialization with custom base URL."""
        custom_url = "http://192.168.1.100:1234"
        client = LMStudioClient(base_url=custom_url)
        assert client.base_url == custom_url + "/v1"  # /v1 is automatically appended

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        client = LMStudioClient(base_url="http://localhost:1234/")
        assert (
            client.base_url == "http://localhost:1234/v1"
        )  # Trailing slash removed, /v1 added

    @pytest.mark.asyncio
    async def test_estimate_cost_always_zero(self, client, sample_request):
        """Test that cost estimation always returns zero for local inference."""
        cost = await client.estimate_cost(sample_request)
        assert cost == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_get_available_models_success(self, client, mock_models_response):
        """Test successful model retrieval."""
        with patch.object(client, "_http_client") as mock_http:
            mock_response = Mock()
            mock_response.json.return_value = mock_models_response
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_http.get = AsyncMock(return_value=mock_response)

            models = await client.get_available_models()

            assert len(models) == 2
            assert models[0].name == "qwen3-8b-mlx"
            assert models[0].provider == "lmstudio"
            assert models[0].input_cost_per_1k == Decimal("0.00")
            assert models[0].output_cost_per_1k == Decimal("0.00")
            assert models[0].context_window == 8192
            assert models[1].name == "qwen3-0.6b-mlx"
            assert models[1].context_window == 4096

    @pytest.mark.asyncio
    async def test_get_available_models_caches_results(
        self, client, mock_models_response
    ):
        """Test that models are cached."""
        with patch.object(client, "_http_client") as mock_http:
            mock_response = Mock()
            mock_response.json.return_value = mock_models_response
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_http.get = AsyncMock(return_value=mock_response)

            # First call
            models1 = await client.get_available_models()
            # Second call should use cache
            models2 = await client.get_available_models()

            assert models1 == models2
            # HTTP client should only be called once
            mock_http.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_models_connection_error(self, client):
        """Test handling of connection errors when getting models."""
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(ClientError) as exc_info:
                await client.get_available_models()

            assert "Failed to connect to LM Studio server" in str(exc_info.value)
            assert "Please ensure LM Studio is running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_models_response):
        """Test successful health check."""
        with patch.object(client, "_http_client") as mock_http:
            mock_response = Mock()
            mock_response.json.return_value = mock_models_response
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)

            is_healthy = await client._check_server_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_models_loaded(self, client):
        """Test health check when no models are loaded."""
        with patch.object(client, "_http_client") as mock_http:
            mock_response = Mock()
            mock_response.json.return_value = {"object": "list", "data": []}
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)

            is_healthy = await client._check_server_health()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_server_down(self, client):
        """Test health check when server is down."""
        with patch.object(client, "_http_client") as mock_http:
            mock_http.get.side_effect = httpx.ConnectError("Connection refused")

            is_healthy = await client._check_server_health()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_caching(self, client, mock_models_response):
        """Test that health check results are cached."""
        with patch.object(client, "_http_client") as mock_http:
            mock_response = Mock()
            mock_response.json.return_value = mock_models_response
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)

            # First health check
            is_healthy1 = await client._check_server_health()
            # Second health check should use cached result
            is_healthy2 = await client._check_server_health()

            assert is_healthy1 is True
            assert is_healthy2 is True
            # HTTP client should only be called once due to caching
            mock_http.get.assert_called_once()

    def test_prepare_request_basic(self, client, sample_request):
        """Test preparing a basic request."""
        prepared = client._prepare_request(sample_request)

        expected = {
            "model": "qwen3-8b-mlx",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "max_tokens": 100,
            "temperature": 0.7,
        }

        assert prepared == expected

    def test_prepare_request_with_system_prompt(self, client):
        """Test preparing request with system prompt."""
        request = ModelRequest(
            model="qwen3-8b-mlx",
            messages=[Message(role="user", content="Hello")],
            system_prompt="You are a helpful assistant.",
        )

        prepared = client._prepare_request(request)

        expected = {
            "model": "qwen3-8b-mlx",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        }

        assert prepared == expected

    def test_prepare_request_minimal(self, client):
        """Test preparing minimal request."""
        request = ModelRequest(
            model="qwen3-8b-mlx", messages=[Message(role="user", content="Test")]
        )

        prepared = client._prepare_request(request)

        expected = {
            "model": "qwen3-8b-mlx",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": False,
        }

        assert prepared == expected

    @pytest.mark.asyncio
    async def test_parse_response_success(
        self, client, sample_request, mock_completion_response
    ):
        """Test successful response parsing."""
        mock_response = Mock()
        mock_response.json.return_value = mock_completion_response

        result = await client._parse_response(mock_response, sample_request)

        assert isinstance(result, ModelResponse)
        assert result.content == "2+2 equals 4."
        assert result.model == "qwen3-8b-mlx"
        assert result.provider == "lmstudio"
        assert result.cost_estimate == Decimal("0.00")
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 6
        assert result.usage.total_tokens == 16
        assert result.metadata["finish_reason"] == "stop"
        assert result.metadata["local_inference"] is True

    @pytest.mark.asyncio
    async def test_parse_response_no_choices(self, client, sample_request):
        """Test response parsing with no choices."""
        mock_response = Mock()
        mock_response.json.return_value = {"choices": []}

        with pytest.raises(ClientError) as exc_info:
            await client._parse_response(mock_response, sample_request)

        assert "no choices returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_response_invalid_json(self, client, sample_request):
        """Test response parsing with invalid JSON."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with pytest.raises(ClientError) as exc_info:
            await client._parse_response(mock_response, sample_request)

        assert "Failed to parse response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_http_error_400_bad_request(self, client):
        """Test handling 400 bad request error."""
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }

        with pytest.raises(ClientError) as exc_info:
            await client._handle_http_error(mock_response)

        assert "Invalid request: Invalid request format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_http_error_404_model_not_found(self, client):
        """Test handling 404 model not found error."""
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Model not found"}}

        with pytest.raises(ClientError) as exc_info:
            await client._handle_http_error(mock_response)

        assert "Model not found or not loaded in LM Studio" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_http_error_500_server_error(self, client):
        """Test handling 500 server error."""
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }

        with pytest.raises(RetryableError) as exc_info:
            await client._handle_http_error(mock_response)

        assert "LM Studio server error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_http_error_503_unavailable(self, client):
        """Test handling 503 service unavailable error."""
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 503
        mock_response.json.return_value = {"error": {"message": "Service unavailable"}}

        with pytest.raises(RetryableError) as exc_info:
            await client._handle_http_error(mock_response)

        assert "LM Studio server unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_success(
        self, client, sample_request, mock_completion_response
    ):
        """Test successful completion request."""
        with (
            patch.object(client, "_check_server_health", return_value=True),
            patch.object(client, "_http_client") as mock_http,
        ):
            mock_response = Mock()
            mock_response.json.return_value = mock_completion_response
            mock_response.is_success = True
            mock_http.post = AsyncMock(return_value=mock_response)

            result = await client.complete(sample_request)

            assert isinstance(result, ModelResponse)
            assert result.content == "2+2 equals 4."
            assert result.model == "qwen3-8b-mlx"
            assert result.cost_estimate == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_complete_server_unhealthy(self, client, sample_request):
        """Test completion when server is unhealthy."""
        with patch.object(client, "_check_server_health", return_value=False):
            with pytest.raises(ClientError) as exc_info:
                await client.complete(sample_request)

            assert "LM Studio server is not responding" in str(exc_info.value)
            assert "Please ensure LM Studio is running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_timeout_error(self, client, sample_request):
        """Test completion with timeout error."""
        with (
            patch.object(client, "_check_server_health", return_value=True),
            patch.object(client, "retry_with_backoff") as mock_retry,
        ):
            mock_retry.side_effect = RetryableError(
                "Request timed out", provider="lmstudio"
            )

            with pytest.raises(RetryableError):
                await client.complete(sample_request)

    @pytest.mark.asyncio
    async def test_complete_connection_error(self, client, sample_request):
        """Test completion with connection error."""
        with (
            patch.object(client, "_check_server_health", return_value=True),
            patch.object(client, "retry_with_backoff") as mock_retry,
        ):
            mock_retry.side_effect = ClientError(
                "Cannot connect to LM Studio server", provider="lmstudio"
            )

            with pytest.raises(ClientError) as exc_info:
                await client.complete(sample_request)

            assert "Cannot connect to LM Studio server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_loaded_model_success(self, client, mock_models_response):
        """Test getting loaded model when models are available."""
        with patch.object(client, "get_available_models") as mock_get_models:
            models = [
                ModelInfo(
                    name="qwen3-8b-mlx",
                    provider="lmstudio",
                    input_cost_per_1k=Decimal("0.00"),
                    output_cost_per_1k=Decimal("0.00"),
                )
            ]
            mock_get_models.return_value = models

            loaded_model = await client.get_loaded_model()
            assert loaded_model == "qwen3-8b-mlx"

    @pytest.mark.asyncio
    async def test_get_loaded_model_no_models(self, client):
        """Test getting loaded model when no models are available."""
        with patch.object(client, "get_available_models", return_value=[]):
            loaded_model = await client.get_loaded_model()
            assert loaded_model is None

    @pytest.mark.asyncio
    async def test_get_loaded_model_client_error(self, client):
        """Test getting loaded model when client error occurs."""
        with patch.object(
            client,
            "get_available_models",
            side_effect=ClientError("Connection failed", "lmstudio"),
        ):
            loaded_model = await client.get_loaded_model()
            assert loaded_model is None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, client):
        """Test that HTTP client is properly closed when using as context manager."""
        with patch.object(client._http_client, "aclose") as mock_close:
            async with client:
                pass
            mock_close.assert_called_once()

    def test_repr(self, client):
        """Test string representation of client."""
        repr_str = repr(client)
        assert "LMStudioClient" in repr_str
        assert "http://localhost:1234" in repr_str


class TestLMStudioClientIntegration:
    """Integration tests for LM Studio client (require running LM Studio)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_server_connection(self):
        """Test connection to real LM Studio server (if available)."""
        client = LMStudioClient()

        try:
            # Try to get models - this will work if LM Studio is running
            models = await client.get_available_models()

            # If we got here, LM Studio is running
            assert isinstance(models, list)

            # If models are loaded, try a completion
            if models:
                request = ModelRequest(
                    model=models[0].name,
                    messages=[Message(role="user", content="Say 'Hello World'")],
                    max_tokens=10,
                )

                response = await client.complete(request)
                assert isinstance(response, ModelResponse)
                assert response.cost_estimate == Decimal("0.00")
                assert len(response.content) > 0

        except ClientError:
            # LM Studio is not running, skip test
            pytest.skip("LM Studio server not available for integration test")  # type: ignore


@pytest.mark.asyncio
async def test_lmstudio_client_factory_integration():
    """Test creating LM Studio client through factory."""
    from second_opinion.utils.client_factory import create_lmstudio_client

    client = create_lmstudio_client()

    assert isinstance(client, LMStudioClient)
    assert client.base_url == "http://localhost:1234/v1"


@pytest.mark.asyncio
async def test_lmstudio_client_factory_custom_url():
    """Test creating LM Studio client with custom URL."""
    from second_opinion.utils.client_factory import create_lmstudio_client

    custom_url = "http://192.168.1.100:8080"
    client = create_lmstudio_client(base_url=custom_url)

    assert isinstance(client, LMStudioClient)
    assert client.base_url == custom_url + "/v1"
