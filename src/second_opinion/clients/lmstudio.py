"""
LM Studio client implementation.

This module provides a concrete implementation of the BaseClient interface
for LM Studio's local AI model server with OpenAI-compatible API.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from ..core.models import ModelRequest, ModelResponse, TokenUsage
from ..utils.sanitization import SecurityError
from .base import (
    AuthenticationError,
    BaseClient,
    ClientError,
    CostLimitExceededError,
    ModelInfo,
    RateLimitError,
    RetryableError,
)

logger = logging.getLogger(__name__)


class LMStudioClient(BaseClient):
    """
    LM Studio client implementing the BaseClient interface.

    Provides access to locally running AI models through LM Studio's
    OpenAI-compatible API server.
    """

    def __init__(self, base_url: str = "http://localhost:1234", **kwargs: Any) -> None:
        super().__init__("lmstudio", api_key=None, **kwargs)

        self.base_url = base_url.rstrip("/")

        # HTTP client configuration (no authentication needed for local server)
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Second Opinion AI Tool",
            },
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )

        # Track server health and loaded models
        self._server_healthy = False
        self._last_health_check: datetime | None = None
        self._health_check_ttl = 30  # 30 seconds

        logger.info(f"Initialized LM Studio client for {self.base_url}")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """
        Execute model completion via LM Studio API.

        Args:
            request: Standardized model request

        Returns:
            Standardized model response

        Raises:
            ClientError: API or validation errors
        """
        # Validate and sanitize request
        validated_request = await self.validate_request(request)

        # Check server health before making request
        if not await self._check_server_health():
            raise ClientError(
                "LM Studio server is not responding. Please ensure LM Studio is running and has a model loaded.",
                provider=self.provider_name,
            )

        # Prepare OpenAI-compatible request
        lmstudio_request = self._prepare_request(validated_request)

        try:
            # Make API call with retry logic
            response = await self.retry_with_backoff(
                self._make_completion_request, lmstudio_request
            )

            # Parse and return standardized response
            return await self._parse_response(response, validated_request)

        except Exception as e:
            logger.error(f"LM Studio completion failed: {e}")
            if isinstance(e, (ClientError, SecurityError)):
                raise
            else:
                raise ClientError(
                    f"LM Studio request failed: {str(e)}",
                    provider=self.provider_name,
                    model=validated_request.model,
                ) from e

    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        """
        Estimate cost for local inference (always $0.00).

        Args:
            request: Standardized model request

        Returns:
            Cost estimate (always $0.00 for local inference)
        """
        # Local inference is always free
        return Decimal("0.00")

    async def get_available_models(self) -> list[ModelInfo]:
        """
        Get list of available models from LM Studio.

        Returns:
            List of available models with metadata

        Raises:
            ClientError: If LM Studio server is unreachable
        """
        # Check cache first
        cached_models = await self._get_cached_models()
        if cached_models is not None:
            return cached_models

        try:
            response = await self._http_client.get("/v1/models")
            await self._handle_http_error(response)

            data = response.json()
            models = []

            for model_data in data.get("data", []):
                model_info = ModelInfo(
                    name=model_data["id"],
                    provider=self.provider_name,
                    input_cost_per_1k=Decimal("0.00"),  # Free local inference
                    output_cost_per_1k=Decimal("0.00"),  # Free local inference
                    max_tokens=model_data.get("context_length"),
                    supports_system_messages=True,  # Most local models support system messages
                    supports_streaming=True,
                    context_window=model_data.get("context_length"),
                    description=f"Local model: {model_data['id']}",
                )
                models.append(model_info)

            # Cache the results
            await self._cache_models(models)

            logger.info(f"Retrieved {len(models)} models from LM Studio")
            return models

        except httpx.RequestError as e:
            raise ClientError(
                f"Failed to connect to LM Studio server at {self.base_url}. "
                f"Please ensure LM Studio is running: {str(e)}",
                provider=self.provider_name,
            ) from e
        except Exception as e:
            logger.error(f"Failed to get LM Studio models: {e}")
            raise ClientError(
                f"Failed to retrieve models from LM Studio: {str(e)}",
                provider=self.provider_name,
            ) from e

    async def _check_server_health(self) -> bool:
        """
        Check if LM Studio server is healthy and has models loaded.

        Returns:
            True if server is healthy and ready
        """
        # Check if we have a recent health check
        now = datetime.now(UTC)
        if (
            self._last_health_check is not None
            and (now - self._last_health_check).total_seconds() < self._health_check_ttl
            and self._server_healthy
        ):
            return True

        try:
            # Try to get models list as health check
            response = await self._http_client.get("/v1/models", timeout=5.0)

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                # Server is healthy if we can get models and at least one is available
                self._server_healthy = len(models) > 0
                self._last_health_check = now

                if not self._server_healthy:
                    logger.warning(
                        "LM Studio server is running but no models are loaded"
                    )

                return self._server_healthy
            else:
                self._server_healthy = False
                return False

        except Exception as e:
            logger.debug(f"LM Studio health check failed: {e}")
            self._server_healthy = False
            return False

    def _prepare_request(self, request: ModelRequest) -> dict[str, Any]:
        """
        Prepare OpenAI-compatible request for LM Studio.

        Args:
            request: Validated model request

        Returns:
            Dictionary formatted for LM Studio API
        """
        # Convert messages to OpenAI format
        messages = []

        # Add system message if present
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Add conversation messages
        for message in request.messages:
            messages.append({"role": message.role, "content": message.content})

        # Build request payload
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False,  # We don't support streaming yet
        }

        # Add optional parameters if specified
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        return payload

    async def _make_completion_request(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Make the actual completion request to LM Studio.

        Args:
            payload: Request payload

        Returns:
            HTTP response

        Raises:
            RetryableError: For transient errors
            ClientError: For permanent errors
        """
        try:
            response = await self._http_client.post(
                "/v1/chat/completions", json=payload
            )
            await self._handle_http_error(response)
            return response

        except httpx.TimeoutException as e:
            raise RetryableError(
                f"Request timed out after {self.timeout}s - model may be processing",
                provider=self.provider_name,
                model=payload.get("model"),
            ) from e
        except httpx.ConnectError as e:
            raise ClientError(
                f"Cannot connect to LM Studio server at {self.base_url}. "
                "Please ensure LM Studio is running.",
                provider=self.provider_name,
                model=payload.get("model"),
            ) from e

    async def _parse_response(
        self, response: httpx.Response, original_request: ModelRequest
    ) -> ModelResponse:
        """
        Parse LM Studio response into standardized format.

        Args:
            response: HTTP response from LM Studio
            original_request: Original request for context

        Returns:
            Standardized model response
        """
        try:
            data = response.json()

            # Extract response content
            if "choices" not in data or not data["choices"]:
                raise ClientError(
                    "Invalid response format: no choices returned",
                    provider=self.provider_name,
                    model=original_request.model,
                )

            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "")

            # Extract usage information
            usage_data = data.get("usage", {})
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            # Create standardized response
            model_name = data.get("model", original_request.model)

            return ModelResponse(
                content=content,
                model=model_name,
                usage=usage,
                cost_estimate=Decimal("0.00"),  # Local inference is free
                provider=self.provider_name,
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "local_inference": True,
                    "server_url": self.base_url,
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse LM Studio response: {e}")
            raise ClientError(
                f"Failed to parse response: {str(e)}",
                provider=self.provider_name,
                model=original_request.model,
            ) from e

    async def _handle_http_error(self, response: httpx.Response) -> None:
        """
        Handle HTTP errors from LM Studio API.

        Args:
            response: HTTP response to check

        Raises:
            Appropriate ClientError subclass based on status code
        """
        if response.is_success:
            return

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
        except Exception:
            error_message = f"HTTP {response.status_code}: {response.reason_phrase}"

        if response.status_code == 400:
            raise ClientError(
                f"Invalid request: {error_message}", provider=self.provider_name
            )
        elif response.status_code == 404:
            if "model" in error_message.lower():
                raise ClientError(
                    f"Model not found or not loaded in LM Studio: {error_message}",
                    provider=self.provider_name,
                )
            else:
                raise ClientError(
                    f"Endpoint not found: {error_message}. "
                    "Please ensure you're running a compatible LM Studio version.",
                    provider=self.provider_name,
                )
        elif response.status_code == 500:
            raise RetryableError(
                f"LM Studio server error (may be temporary): {error_message}",
                provider=self.provider_name,
            )
        elif response.status_code == 503:
            raise RetryableError(
                f"LM Studio server unavailable: {error_message}",
                provider=self.provider_name,
            )
        else:
            raise ClientError(
                f"LM Studio API error ({response.status_code}): {error_message}",
                provider=self.provider_name,
            )

    async def get_loaded_model(self) -> str | None:
        """
        Get the currently loaded model in LM Studio.

        Returns:
            Model name if loaded, None if no model is loaded
        """
        try:
            models = await self.get_available_models()
            # In LM Studio, typically only one model is loaded at a time
            # Return the first available model
            return models[0].name if models else None
        except ClientError:
            return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with HTTP client cleanup."""
        await self._http_client.aclose()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    def __repr__(self):
        return f"LMStudioClient(base_url='{self.base_url}')"
