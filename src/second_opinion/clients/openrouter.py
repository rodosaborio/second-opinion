"""
OpenRouter client implementation.

This module provides a concrete implementation of the BaseClient interface
for OpenRouter's unified AI model API.
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


class OpenRouterClient(BaseClient):
    """
    OpenRouter client implementing the BaseClient interface.

    Provides access to hundreds of AI models through OpenRouter's unified API.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__("openrouter", api_key, **kwargs)

        if not api_key:
            raise ValueError("OpenRouter API key is required")

        # Validate API key format
        if not api_key.startswith("sk-or-"):
            logger.warning("OpenRouter API key should start with 'sk-or-'")

        # HTTP client configuration
        self._http_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/anthropics/claude-code",
                "X-Title": "Second Opinion AI Tool"
            },
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

        # Default cost estimates (conservative fallbacks)
        self._default_costs = {
            "gpt-3.5-turbo": {"input": Decimal("0.0015"), "output": Decimal("0.002")},
            "gpt-4": {"input": Decimal("0.03"), "output": Decimal("0.06")},
            "gpt-4o": {"input": Decimal("0.005"), "output": Decimal("0.015")},
            "claude-3-sonnet": {"input": Decimal("0.003"), "output": Decimal("0.015")},
            "claude-3-5-sonnet": {"input": Decimal("0.003"), "output": Decimal("0.015")},
            "claude-3-haiku": {"input": Decimal("0.00025"), "output": Decimal("0.00125")},
        }

        logger.info("Initialized OpenRouter client")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """
        Execute model completion via OpenRouter API.

        Args:
            request: Standardized model request

        Returns:
            Standardized model response

        Raises:
            ClientError: API or validation errors
        """
        # Validate and sanitize request
        validated_request = await self.validate_request(request)

        # Prepare OpenRouter API request
        openrouter_request = self._prepare_request(validated_request)

        try:
            # Make API call with retry logic
            response = await self.retry_with_backoff(
                self._make_completion_request,
                openrouter_request
            )

            # Parse and return standardized response
            return await self._parse_response(response, validated_request)

        except Exception as e:
            logger.error(f"OpenRouter completion failed: {e}")
            if isinstance(e, ClientError | SecurityError):
                raise
            else:
                raise ClientError(
                    f"Unexpected error during completion: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    details={"error_type": type(e).__name__}
                ) from e

    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        """
        Estimate cost for a model request.

        Args:
            request: Model request to estimate cost for

        Returns:
            Estimated cost in USD
        """
        # Estimate token counts
        input_tokens = self._estimate_input_tokens(request)
        output_tokens = request.max_tokens or 1000  # Conservative estimate

        # Get cost rates for model
        model_key = self._normalize_model_name(request.model)
        costs = self._default_costs.get(model_key)

        if not costs:
            # Conservative fallback for unknown models
            logger.warning(f"No cost data for model {request.model}, using fallback")
            return Decimal("0.10")

        # Calculate cost
        input_cost = Decimal(input_tokens) * costs["input"] / 1000
        output_cost = Decimal(output_tokens) * costs["output"] / 1000

        total_cost = input_cost + output_cost
        logger.debug(f"Estimated cost for {request.model}: ${total_cost}")

        return total_cost

    async def get_available_models(self) -> list[ModelInfo]:
        """
        Get list of available models from OpenRouter.

        Returns:
            List of available models with metadata
        """
        # Check cache first
        cached_models = await self._get_cached_models()
        if cached_models is not None:
            return cached_models

        try:
            response = await self._http_client.get("/models")
            response.raise_for_status()

            models_data = response.json()
            models = self._parse_models_response(models_data)

            # Cache the results
            await self._cache_models(models)

            logger.info(f"Retrieved {len(models)} models from OpenRouter")
            return models

        except httpx.HTTPStatusError as e:
            raise ClientError(
                f"Failed to retrieve models: {e.response.status_code}",
                provider=self.provider_name,
                details={"status_code": e.response.status_code}
            ) from e
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            raise ClientError(
                f"Failed to retrieve models: {e}",
                provider=self.provider_name,
                details={"error_type": type(e).__name__}
            ) from e

    def _prepare_request(self, request: ModelRequest) -> dict[str, Any]:
        """Prepare OpenRouter API request from standardized request."""
        messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in request.messages
        ]

        openrouter_request: dict[str, Any] = {
            "model": request.model,
            "messages": messages
        }

        # Add optional parameters
        if request.max_tokens is not None:
            openrouter_request["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            openrouter_request["temperature"] = request.temperature

        if request.system_prompt:
            # Add system message at the beginning
            messages.insert(0, {
                "role": "system",
                "content": request.system_prompt
            })

        return openrouter_request

    async def _make_completion_request(self, request_data: dict[str, Any]) -> httpx.Response:
        """Make the actual HTTP request to OpenRouter."""
        response = await self._http_client.post("/chat/completions", json=request_data)

        # Handle HTTP errors
        if response.status_code != 200:
            await self._handle_http_error(response)

        return response

    async def _handle_http_error(self, response: httpx.Response) -> None:
        """Handle HTTP errors from OpenRouter API."""
        try:
            error_data = response.json()
            error_info = error_data.get("error", {})
            error_message = error_info.get("message", f"HTTP {response.status_code}")
            error_metadata = error_info.get("metadata", {})
        except Exception:
            error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            error_metadata = {}

        # Map HTTP status codes to our exception types
        if response.status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {error_message}",
                provider=self.provider_name,
                details=error_metadata
            )
        elif response.status_code == 402:
            raise CostLimitExceededError(
                f"Insufficient credits: {error_message}",
                provider=self.provider_name,
                estimated_cost=Decimal("0"),  # Unknown at this point
                cost_limit=Decimal("0"),  # Unknown at this point
                details=error_metadata
            )
        elif response.status_code == 403:
            raise SecurityError(f"Request blocked by moderation: {error_message}")
        elif response.status_code == 429:
            retry_after = None
            if "retry-after" in response.headers:
                try:
                    retry_after = int(response.headers["retry-after"])
                except ValueError:
                    pass

            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                provider=self.provider_name,
                retry_after=retry_after,
                details=error_metadata
            )
        elif response.status_code in (408, 502, 503):
            raise RetryableError(
                f"Service temporarily unavailable: {error_message}",
                provider=self.provider_name,
                details=error_metadata
            )
        else:
            raise ClientError(
                f"API error: {error_message}",
                provider=self.provider_name,
                details={**error_metadata, "status_code": response.status_code}
            )

    async def _parse_response(self, response: httpx.Response, request: ModelRequest) -> ModelResponse:
        """Parse OpenRouter response into standardized format."""
        try:
            data = response.json()

            # Extract completion content
            choices = data.get("choices", [])
            if not choices:
                raise ClientError(
                    "No choices in response",
                    provider=self.provider_name,
                    model=request.model
                )

            content = choices[0].get("message", {}).get("content", "")

            # Extract usage information
            usage_data = data.get("usage", {})
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

            # Calculate actual cost
            try:
                model_info = await self._get_model_cost_info(request.model)
                actual_cost = self._calculate_token_cost(
                    usage.input_tokens,
                    usage.output_tokens,
                    model_info
                )
            except Exception:
                # Fallback cost calculation
                actual_cost = Decimal("0.001") * usage.total_tokens

            return ModelResponse(
                content=content,
                model=request.model,
                usage=usage,
                cost_estimate=actual_cost,
                provider=self.provider_name,
                metadata={
                    "openrouter_id": data.get("id"),
                    "finish_reason": choices[0].get("finish_reason"),
                    "response_time": datetime.now(UTC)
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            raise ClientError(
                f"Failed to parse response: {e}",
                provider=self.provider_name,
                model=request.model,
                details={"error_type": type(e).__name__}
            ) from e

    def _parse_models_response(self, models_data: dict[str, Any]) -> list[ModelInfo]:
        """Parse OpenRouter models response."""
        models = []

        model_list = models_data.get("data", [])
        for model_data in model_list:
            try:
                model_id = model_data.get("id", "")
                if not model_id:
                    continue

                # Extract pricing (if available)
                pricing = model_data.get("pricing", {})
                input_cost = Decimal(str(pricing.get("prompt", "0.001")))
                output_cost = Decimal(str(pricing.get("completion", "0.002")))

                # Extract capabilities
                context_length = model_data.get("context_length")
                max_tokens = model_data.get("top_provider", {}).get("max_completion_tokens")

                model_info = ModelInfo(
                    name=model_id,
                    provider="openrouter",
                    input_cost_per_1k=input_cost,
                    output_cost_per_1k=output_cost,
                    max_tokens=max_tokens,
                    context_window=context_length,
                    description=model_data.get("description", "")
                )

                models.append(model_info)

            except Exception as e:
                logger.warning(f"Failed to parse model {model_data.get('id', 'unknown')}: {e}")
                continue

        return models

    async def _get_model_cost_info(self, model_name: str) -> ModelInfo:
        """Get cost information for a specific model."""
        # Try to get from available models first
        try:
            models = await self.get_available_models()
            for model in models:
                if model.name == model_name:
                    return model
        except Exception as e:
            logger.debug(f"Failed to get model cost info from available models: {e}")

        # Fallback to default costs
        model_key = self._normalize_model_name(model_name)
        costs = self._default_costs.get(model_key)

        if costs:
            return ModelInfo(
                name=model_name,
                provider=self.provider_name,
                input_cost_per_1k=costs["input"],
                output_cost_per_1k=costs["output"]
            )

        # Ultimate fallback
        return ModelInfo(
            name=model_name,
            provider=self.provider_name,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002")
        )

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize OpenRouter model name to key for cost lookup."""
        # Remove provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        # Common mappings
        mappings = {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4",
            "gpt-4o": "gpt-4o",
            "claude-3-sonnet-20240229": "claude-3-sonnet",
            "claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
            "claude-3-haiku-20240307": "claude-3-haiku",
        }

        return mappings.get(model_name, model_name)

    def _estimate_input_tokens(self, request: ModelRequest) -> int:
        """Estimate input tokens for a request."""
        total_text = ""

        # Add system prompt
        if request.system_prompt:
            total_text += request.system_prompt + " "

        # Add all messages
        for message in request.messages:
            total_text += message.content + " "

        return self._estimate_tokens(total_text)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up HTTP client on exit."""
        if hasattr(self, '_http_client'):
            await self._http_client.aclose()
        await super().__aexit__(exc_type, exc_val, exc_tb)
