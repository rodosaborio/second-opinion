"""
Tests for client factory utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.second_opinion.clients.base import BaseClient
from src.second_opinion.clients.lmstudio import LMStudioClient
from src.second_opinion.clients.openrouter import OpenRouterClient
from src.second_opinion.utils.client_factory import (
    ClientFactoryError,
    _get_provider_config,
    create_client_from_config,
    create_lmstudio_client,
    create_openrouter_client,
    get_configured_providers,
    validate_provider_config,
)


class TestClientFactory:
    """Test client factory functionality."""

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_create_client_from_config_openrouter(self, mock_get_settings):
        """Test creating OpenRouter client from config."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = "sk-or-test123"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60
        mock_get_settings.return_value = mock_settings

        client = create_client_from_config("openrouter")

        assert isinstance(client, OpenRouterClient)
        assert client.provider_name == "openrouter"
        assert client.api_key == "sk-or-test123"
        mock_settings.get_api_key.assert_called_once_with("openrouter")

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_create_client_from_config_missing_key(self, mock_get_settings):
        """Test creating client with missing API key."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(
            ClientFactoryError, match="OpenRouter API key not configured"
        ):
            create_client_from_config("openrouter")

    def test_create_client_from_config_unsupported_provider(self):
        """Test creating client for unsupported provider."""
        with pytest.raises(ClientFactoryError, match="Unsupported provider"):
            create_client_from_config("unsupported")

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_create_client_from_config_with_overrides(self, mock_get_settings):
        """Test creating client with configuration overrides."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = "sk-or-original"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60
        mock_get_settings.return_value = mock_settings

        overrides = {"api_key": "sk-or-override", "timeout": 60}

        client = create_client_from_config("openrouter", overrides)

        assert isinstance(client, OpenRouterClient)
        assert client.api_key == "sk-or-override"  # Should use override
        assert client.timeout == 60  # Should use override

    @patch("src.second_opinion.utils.client_factory.create_client_from_config")
    def test_create_openrouter_client_convenience(self, mock_create):
        """Test convenience function for creating OpenRouter client."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        client = create_openrouter_client(api_key="sk-or-test123", timeout=45)

        mock_create.assert_called_once_with(
            "openrouter", {"api_key": "sk-or-test123", "timeout": 45}
        )
        assert client == mock_client

    @patch("src.second_opinion.utils.client_factory.create_client_from_config")
    def test_create_openrouter_client_no_overrides(self, mock_create):
        """Test convenience function without overrides."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        client = create_openrouter_client()

        mock_create.assert_called_once_with("openrouter", {})
        assert client == mock_client

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_create_client_from_config_lmstudio(self, mock_get_settings):
        """Test creating LM Studio client from config."""
        mock_settings = MagicMock()
        mock_settings.lmstudio_base_url = "http://localhost:1234"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60
        mock_get_settings.return_value = mock_settings

        client = create_client_from_config("lmstudio")

        assert isinstance(client, LMStudioClient)
        assert client.provider_name == "lmstudio"
        assert client.base_url == "http://localhost:1234/v1"

    @patch("src.second_opinion.utils.client_factory.create_client_from_config")
    def test_create_lmstudio_client_convenience(self, mock_create):
        """Test convenience function for creating LM Studio client."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        client = create_lmstudio_client(
            base_url="http://192.168.1.100:1234", timeout=45
        )

        mock_create.assert_called_once_with(
            "lmstudio", {"base_url": "http://192.168.1.100:1234", "timeout": 45}
        )
        assert client == mock_client

    @patch("src.second_opinion.utils.client_factory.create_client_from_config")
    def test_create_lmstudio_client_no_overrides(self, mock_create):
        """Test convenience function without overrides."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        client = create_lmstudio_client()

        mock_create.assert_called_once_with("lmstudio", {})
        assert client == mock_client


class TestProviderConfig:
    """Test provider configuration functionality."""

    def test_get_provider_config_openrouter(self):
        """Test getting OpenRouter provider config."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = "sk-or-test123"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60

        config = _get_provider_config("openrouter", mock_settings)

        expected_config = {
            "api_key": "sk-or-test123",
            "timeout": 30,
            "max_retries": 2,
            "base_delay": 1.0,
            "max_delay": 60,
        }

        assert config == expected_config

    def test_get_provider_config_openrouter_missing_key(self):
        """Test getting OpenRouter config with missing API key."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = None

        with pytest.raises(
            ClientFactoryError, match="OpenRouter API key not configured"
        ):
            _get_provider_config("openrouter", mock_settings)

    def test_get_provider_config_lmstudio(self):
        """Test getting LM Studio provider config."""
        mock_settings = MagicMock()
        mock_settings.lmstudio_base_url = "http://localhost:1234"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60

        config = _get_provider_config("lmstudio", mock_settings)

        expected_config = {
            "base_url": "http://localhost:1234",
            "timeout": 30,
            "max_retries": 2,
            "base_delay": 1.0,
            "max_delay": 60,
        }

        assert config == expected_config

    def test_get_provider_config_unknown_provider(self):
        """Test getting config for unknown provider."""
        mock_settings = MagicMock()

        with pytest.raises(ClientFactoryError, match="Unknown provider configuration"):
            _get_provider_config("unknown", mock_settings)


class TestProviderValidation:
    """Test provider validation functionality."""

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_validate_provider_config_valid(self, mock_get_settings):
        """Test validating a properly configured provider."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = "sk-or-test123"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60
        mock_get_settings.return_value = mock_settings

        assert validate_provider_config("openrouter") is True

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_validate_provider_config_invalid(self, mock_get_settings):
        """Test validating an improperly configured provider."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = None  # Missing API key
        mock_get_settings.return_value = mock_settings

        assert validate_provider_config("openrouter") is False

    @patch("src.second_opinion.utils.client_factory.validate_provider_config")
    @patch("src.second_opinion.utils.client_factory.get_supported_providers")
    def test_get_configured_providers(self, mock_get_supported, mock_validate):
        """Test getting list of configured providers."""
        mock_get_supported.return_value = ["openrouter", "lmstudio", "anthropic"]
        mock_validate.side_effect = lambda p: p in ["openrouter", "anthropic"]

        configured = get_configured_providers()

        assert configured == ["openrouter", "anthropic"]
        assert mock_validate.call_count == 3

    @patch("src.second_opinion.utils.client_factory.validate_provider_config")
    @patch("src.second_opinion.utils.client_factory.get_supported_providers")
    def test_get_configured_providers_none(self, mock_get_supported, mock_validate):
        """Test when no providers are configured."""
        mock_get_supported.return_value = ["openrouter", "lmstudio"]
        mock_validate.return_value = False

        configured = get_configured_providers()

        assert configured == []


class TestClientFactoryErrors:
    """Test error handling in client factory."""

    @patch("src.second_opinion.utils.client_factory.get_settings")
    def test_create_client_generic_error(self, mock_get_settings):
        """Test handling of generic errors during client creation."""
        mock_get_settings.side_effect = Exception("Database connection failed")

        with pytest.raises(
            ClientFactoryError, match="Failed to create openrouter client"
        ):
            create_client_from_config("openrouter")

    @patch("src.second_opinion.utils.client_factory.get_settings")
    @patch("src.second_opinion.utils.client_factory.create_client")
    def test_create_client_creation_error(self, mock_create, mock_get_settings):
        """Test handling of client creation errors."""
        mock_settings = MagicMock()
        mock_settings.get_api_key.return_value = "sk-or-test123"
        mock_settings.api.timeout = 30
        mock_settings.api.retries = 2
        mock_settings.api.max_backoff = 60
        mock_get_settings.return_value = mock_settings

        mock_create.side_effect = ValueError("Invalid API key format")

        with pytest.raises(
            ClientFactoryError, match="Failed to create openrouter client"
        ):
            create_client_from_config("openrouter")


class TestClientFactoryIntegration:
    """Integration tests for client factory."""

    @pytest.mark.integration
    def test_factory_creates_working_client(self):
        """Test that factory creates a working client."""
        # This test uses actual client creation (but mocked settings)
        with patch(
            "src.second_opinion.utils.client_factory.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.get_api_key.return_value = "sk-or-test123"
            mock_settings.api.timeout = 30
            mock_settings.api.retries = 2
            mock_settings.api.max_backoff = 60
            mock_get_settings.return_value = mock_settings

            client = create_client_from_config("openrouter")

            assert isinstance(client, BaseClient)
            assert isinstance(client, OpenRouterClient)
            assert client.provider_name == "openrouter"
            assert client.api_key == "sk-or-test123"
            assert client.timeout == 30
            assert client.max_retries == 2

    @pytest.mark.integration
    def test_convenience_functions_work(self):
        """Test that convenience functions integrate properly."""
        with patch(
            "src.second_opinion.utils.client_factory.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.get_api_key.return_value = "sk-or-default"
            mock_settings.api.timeout = 30
            mock_settings.api.retries = 2
            mock_settings.api.max_backoff = 60
            mock_get_settings.return_value = mock_settings

            # Test with override
            client = create_openrouter_client(api_key="sk-or-override")
            assert client.api_key == "sk-or-override"

            # Test without override
            client2 = create_openrouter_client()
            assert client2.api_key == "sk-or-default"

    @pytest.mark.integration
    def test_lmstudio_factory_creates_working_client(self):
        """Test that LM Studio factory creates a working client."""
        with patch(
            "src.second_opinion.utils.client_factory.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.lmstudio_base_url = "http://localhost:1234"
            mock_settings.api.timeout = 30
            mock_settings.api.retries = 2
            mock_settings.api.max_backoff = 60
            mock_get_settings.return_value = mock_settings

            client = create_client_from_config("lmstudio")

            assert isinstance(client, BaseClient)
            assert isinstance(client, LMStudioClient)
            assert client.provider_name == "lmstudio"
            assert client.base_url == "http://localhost:1234/v1"
            assert client.timeout == 30
            assert client.max_retries == 2

    @pytest.mark.integration
    def test_lmstudio_convenience_functions_work(self):
        """Test that LM Studio convenience functions integrate properly."""
        with patch(
            "src.second_opinion.utils.client_factory.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.lmstudio_base_url = "http://localhost:1234"
            mock_settings.api.timeout = 30
            mock_settings.api.retries = 2
            mock_settings.api.max_backoff = 60
            mock_get_settings.return_value = mock_settings

            # Test with override
            client = create_lmstudio_client(base_url="http://192.168.1.100:8080")
            assert client.base_url == "http://192.168.1.100:8080/v1"

            # Test without override
            client2 = create_lmstudio_client()
            assert client2.base_url == "http://localhost:1234/v1"
