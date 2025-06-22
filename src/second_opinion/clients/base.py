"""
Abstract base client interface for model providers.

This module defines the standard interface that all model provider clients
must implement, ensuring consistent behavior across different providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from ..core.models import ModelRequest, ModelResponse, TokenUsage
from ..utils.sanitization import SecurityContext, sanitize_prompt, validate_model_name

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about a model's capabilities and pricing."""

    def __init__(
        self,
        name: str,
        provider: str,
        input_cost_per_1k: Decimal,
        output_cost_per_1k: Decimal,
        max_tokens: int | None = None,
        supports_system_messages: bool = True,
        supports_streaming: bool = False,
        context_window: int | None = None,
        description: str | None = None,
    ):
        self.name = validate_model_name(name)
        self.provider = provider
        self.input_cost_per_1k = Decimal(str(input_cost_per_1k))
        self.output_cost_per_1k = Decimal(str(output_cost_per_1k))
        self.max_tokens = max_tokens
        self.supports_system_messages = supports_system_messages
        self.supports_streaming = supports_streaming
        self.context_window = context_window
        self.description = description or f"{provider} model: {name}"

    def __repr__(self) -> str:
        return f"ModelInfo(name='{self.name}', provider='{self.provider}', input_cost=${self.input_cost_per_1k}/1k)"


class ClientError(Exception):
    """Base exception for client errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        parts = [f"{self.provider}: {self.message}"]
        if self.model:
            parts.append(f"Model: {self.model}")
        return " | ".join(parts)


class AuthenticationError(ClientError):
    """Authentication failed with provider."""

    pass


class RateLimitError(ClientError):
    """Rate limit exceeded."""

    def __init__(
        self, message: str, provider: str, retry_after: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, provider, **kwargs)
        self.retry_after = retry_after


class ModelNotFoundError(ClientError):
    """Requested model not found or unavailable."""

    pass


class CostLimitExceededError(ClientError):
    """Request would exceed cost limits."""

    def __init__(
        self,
        message: str,
        provider: str,
        estimated_cost: Decimal,
        cost_limit: Decimal,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, provider, **kwargs)
        self.estimated_cost = estimated_cost
        self.cost_limit = cost_limit


class RetryableError(ClientError):
    """Error that can be retried."""

    pass


class BaseClient(ABC):
    """
    Abstract base client for all model providers.

    This defines the standard interface that all provider-specific clients
    must implement to ensure consistent behavior across the application.
    """

    def __init__(
        self, provider_name: str, api_key: str | None = None, **kwargs: Any
    ) -> None:
        self.provider_name = provider_name
        self.api_key = api_key
        self._models_cache: list[ModelInfo] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl = 300  # 5 minutes

        # Configuration from kwargs
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 1.0)
        self.max_delay = kwargs.get("max_delay", 60.0)

        logger.info(f"Initialized {self.provider_name} client")

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """
        Execute model completion with standardized interface.

        Args:
            request: Standardized model request

        Returns:
            Standardized model response

        Raises:
            ClientError: Provider-specific errors
            ValidationError: Invalid request parameters
            CostLimitExceededError: Request exceeds cost limits
        """
        pass

    @abstractmethod
    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        """
        Estimate cost before making request.

        Args:
            request: Standardized model request

        Returns:
            Estimated cost in USD

        Raises:
            ModelNotFoundError: Model not found or pricing unavailable
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> list[ModelInfo]:
        """
        Get list of available models with capabilities.

        Returns:
            List of available models with metadata

        Raises:
            ClientError: Provider API errors
        """
        pass

    async def validate_request(self, request: ModelRequest) -> ModelRequest:
        """
        Validate and sanitize request before processing.

        Args:
            request: Model request to validate

        Returns:
            Validated and sanitized request

        Raises:
            ValidationError: Invalid request parameters
        """
        # Validate model name
        model_name = validate_model_name(request.model)

        # Sanitize messages
        sanitized_messages = []
        for message in request.messages:
            sanitized_content = sanitize_prompt(
                message.content,
                SecurityContext.API_REQUEST,  # Use strict validation for all client API requests
            )
            sanitized_messages.append(
                message.model_copy(update={"content": sanitized_content})
            )

        # Sanitize system prompt if present
        system_prompt = None
        if request.system_prompt:
            system_prompt = sanitize_prompt(
                request.system_prompt, SecurityContext.API_REQUEST
            )

        # Create validated request
        validated_request = request.model_copy(
            update={
                "model": model_name,
                "messages": sanitized_messages,
                "system_prompt": system_prompt,
            }
        )

        return validated_request

    async def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available from this provider.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available
        """
        try:
            models = await self.get_available_models()
            return any(model.name == model_name for model in models)
        except ClientError:
            return False

    async def get_model_info(self, model_name: str) -> ModelInfo | None:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information if available, None otherwise
        """
        try:
            models = await self.get_available_models()
            for model in models:
                if model.name == model_name:
                    return model
        except ClientError:
            pass
        return None

    async def _get_cached_models(self) -> list[ModelInfo] | None:
        """Get models from cache if still valid."""
        if (
            self._models_cache is None
            or self._cache_timestamp is None
            or (datetime.now(UTC) - self._cache_timestamp).total_seconds()
            > self._cache_ttl
        ):
            return None
        return self._models_cache

    async def _cache_models(self, models: list[ModelInfo]) -> None:
        """Cache models list with timestamp."""
        self._models_cache = models
        self._cache_timestamp = datetime.now(UTC)

    async def retry_with_backoff(
        self, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute operation with exponential backoff retry logic.

        Args:
            operation: Async function to execute
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Result of successful operation

        Raises:
            ClientError: If all retries are exhausted
        """
        last_exception: RetryableError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except RetryableError as e:
                last_exception = e
                if attempt == self.max_retries:
                    break

                # Calculate delay with exponential backoff and jitter
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                jitter = delay * 0.1  # 10% jitter
                actual_delay = delay + (jitter * (2 * hash(str(e)) / 2**32 - 1))

                logger.warning(
                    f"Attempt {attempt + 1} failed for {self.provider_name}: {e}. "
                    f"Retrying in {actual_delay:.2f}s"
                )
                await asyncio.sleep(actual_delay)
            except ClientError:
                # Non-retryable errors
                raise

        # All retries exhausted
        if last_exception is not None:
            raise last_exception
        else:
            raise ClientError(
                "Operation failed without retryable errors", self.provider_name
            )

    def _calculate_token_cost(
        self, input_tokens: int, output_tokens: int, model_info: ModelInfo
    ) -> Decimal:
        """
        Calculate cost for given token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_info: Model pricing information

        Returns:
            Total cost in USD
        """
        input_cost = Decimal(input_tokens) * model_info.input_cost_per_1k / 1000
        output_cost = Decimal(output_tokens) * model_info.output_cost_per_1k / 1000
        return input_cost + output_cost

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count for text.

        This is a fallback method. Providers should implement more accurate
        token counting if available.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)

    def _create_error_response(
        self, error: Exception, request: ModelRequest
    ) -> ModelResponse:
        """
        Create a standardized error response.

        Args:
            error: The exception that occurred
            request: Original request

        Returns:
            Error response with minimal cost
        """
        return ModelResponse(
            content=f"Error: {str(error)}",
            model=request.model,
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            cost_estimate=Decimal("0"),
            provider=self.provider_name,
            metadata={"error": True, "error_type": type(error).__name__},
        )

    async def __aenter__(self) -> "BaseClient":
        """Async context manager entry."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        # Cleanup resources if needed
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"
