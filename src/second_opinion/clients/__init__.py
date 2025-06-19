"""
Client implementations for different AI model providers.

This package provides a unified interface to multiple AI model providers
through the BaseClient abstraction.
"""

from .base import (
    AuthenticationError,
    BaseClient,
    ClientError,
    CostLimitExceededError,
    ModelInfo,
    ModelNotFoundError,
    RateLimitError,
    RetryableError,
)
from .lmstudio import LMStudioClient
from .openrouter import OpenRouterClient

__all__ = [
    "BaseClient",
    "ModelInfo",
    "ClientError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "CostLimitExceededError",
    "RetryableError",
    "LMStudioClient",
    "OpenRouterClient",
    "create_client",
]


def create_client(provider: str, **kwargs) -> BaseClient:
    """
    Create a client for the specified provider.
    
    Args:
        provider: Provider name (e.g., "openrouter", "lmstudio")
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured client instance
        
    Raises:
        ValueError: If provider is not supported
        
    Example:
        >>> client = create_client("openrouter", api_key="sk-or-...")
        >>> response = await client.complete(request)
    """
    provider = provider.lower().strip()

    if provider == "openrouter":
        return OpenRouterClient(**kwargs)
    elif provider == "lmstudio":
        return LMStudioClient(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_supported_providers() -> list[str]:
    """Get list of supported provider names."""
    return ["openrouter", "lmstudio"]
