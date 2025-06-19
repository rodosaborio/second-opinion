"""
Client factory utilities for creating configured AI model clients.

This module provides convenience functions for creating clients from
application configuration with proper validation and error handling.
"""

import logging
from typing import Any

from ..clients import BaseClient, create_client, get_supported_providers
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class ClientFactoryError(Exception):
    """Error creating client from configuration."""
    pass


def create_client_from_config(
    provider: str,
    config_overrides: dict[str, Any] | None = None
) -> BaseClient:
    """
    Create a client using application configuration.
    
    Args:
        provider: Provider name (e.g., "openrouter")
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured client instance
        
    Raises:
        ClientFactoryError: If client creation fails
        
    Example:
        >>> client = create_client_from_config("openrouter")
        >>> response = await client.complete(request)
    """
    try:
        settings = get_settings()

        # Validate provider is supported
        if provider not in get_supported_providers():
            raise ClientFactoryError(f"Unsupported provider: {provider}")

        # Get provider-specific configuration
        client_config = _get_provider_config(provider, settings)

        # Apply any overrides
        if config_overrides:
            client_config.update(config_overrides)

        # Create and return client
        client = create_client(provider, **client_config)

        logger.info(f"Created {provider} client successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to create {provider} client: {e}")
        raise ClientFactoryError(f"Failed to create {provider} client: {e}") from e


def _get_provider_config(provider: str, settings) -> dict[str, Any]:
    """Get configuration for a specific provider."""
    if provider == "openrouter":
        api_key = settings.get_api_key("openrouter")
        if not api_key:
            raise ClientFactoryError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY environment variable."
            )

        return {
            "api_key": api_key,
            "timeout": settings.api.timeout,
            "max_retries": settings.api.retries,
            "base_delay": 1.0,
            "max_delay": settings.api.max_backoff,
        }

    elif provider == "lmstudio":
        return {
            "base_url": settings.lmstudio_base_url,
            "timeout": settings.api.timeout,
            "max_retries": settings.api.retries,
            "base_delay": 1.0,
            "max_delay": settings.api.max_backoff,
        }

    else:
        raise ClientFactoryError(f"Unknown provider configuration: {provider}")


def create_openrouter_client(
    api_key: str | None = None,
    **kwargs
) -> BaseClient:
    """
    Convenience function to create an OpenRouter client.
    
    Args:
        api_key: Optional API key override
        **kwargs: Additional client configuration
        
    Returns:
        Configured OpenRouter client
    """
    config_overrides = {}
    if api_key:
        config_overrides["api_key"] = api_key
    config_overrides.update(kwargs)

    return create_client_from_config("openrouter", config_overrides)


def create_lmstudio_client(
    base_url: str | None = None,
    **kwargs
) -> BaseClient:
    """
    Convenience function to create an LM Studio client.
    
    Args:
        base_url: Optional base URL override (default: http://localhost:1234)
        **kwargs: Additional client configuration
        
    Returns:
        Configured LM Studio client
    """
    config_overrides = {}
    if base_url:
        config_overrides["base_url"] = base_url
    config_overrides.update(kwargs)

    return create_client_from_config("lmstudio", config_overrides)


def validate_provider_config(provider: str) -> bool:
    """
    Validate that a provider is properly configured.
    
    Args:
        provider: Provider name to validate
        
    Returns:
        True if properly configured, False otherwise
    """
    try:
        _get_provider_config(provider, get_settings())
        return True
    except ClientFactoryError:
        return False


def get_configured_providers() -> list[str]:
    """Get list of providers that are properly configured."""
    configured = []

    for provider in get_supported_providers():
        if validate_provider_config(provider):
            configured.append(provider)

    return configured
