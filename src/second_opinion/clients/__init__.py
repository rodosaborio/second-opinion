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
    "get_client_for_model",
    "detect_model_provider",
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


def detect_model_provider(model: str) -> str:
    """
    Detect the appropriate provider for a given model.
    
    Args:
        model: Model identifier
        
    Returns:
        Provider name ("lmstudio" or "openrouter")
        
    Examples:
        >>> detect_model_provider("qwen3-4b-mlx")
        "lmstudio"
        >>> detect_model_provider("anthropic/claude-3-5-sonnet")
        "openrouter"
    """
    # Local model patterns - these should use LM Studio
    local_patterns = [
        "mlx",      # MLX models
        "qwen",     # Qwen models  
        "llama",    # Llama models
        "mistral",  # Mistral models (when local)
        "codestral", # Codestral models
        "devstral", # Devstral models
    ]
    
    model_lower = model.lower()
    
    # If model has provider prefix (provider/model), it's a cloud model
    if "/" in model and len(model.split("/")) == 2:
        return "openrouter"
    
    # Check for local model patterns
    if any(pattern in model_lower for pattern in local_patterns) and "/" not in model:
        return "lmstudio"
    
    # Default to OpenRouter for unknown models
    return "openrouter"


def get_client_for_model(model: str, **kwargs) -> BaseClient:
    """
    Get the appropriate client for a given model.
    
    Args:
        model: Model identifier
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured client instance for the model
        
    Raises:
        ValueError: If provider detection fails
        
    Examples:
        >>> client = get_client_for_model("qwen3-4b-mlx")
        >>> client = get_client_for_model("anthropic/claude-3-5-sonnet")
    """
    provider = detect_model_provider(model)
    return create_client(provider, **kwargs)
