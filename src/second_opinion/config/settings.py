"""
Configuration management for Second Opinion.

This module implements hierarchical configuration loading with validation,
following the pattern: CLI args > env vars > user config > defaults.
"""

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = Field(
        default="data/second_opinion.db", description="Database file path"
    )
    encryption_enabled: bool = Field(
        default=True, description="Enable database encryption"
    )
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    backup_interval: int = Field(
        default=86400, ge=1, description="Backup interval in seconds"
    )
    retention_days: int = Field(
        default=30, ge=1, description="Data retention period in days"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""

    input_sanitization: bool = Field(
        default=True, description="Enable input sanitization"
    )
    response_filtering: bool = Field(
        default=True, description="Enable response filtering"
    )
    min_key_length: int = Field(default=32, ge=8, description="Minimum API key length")
    session_timeout: int = Field(
        default=3600, ge=60, description="Session timeout in seconds"
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=1, description="Rate limit per minute"
    )
    max_concurrent_requests: int = Field(
        default=5, ge=1, le=50, description="Max concurrent requests"
    )


class CostManagementConfig(BaseModel):
    """Cost management configuration."""

    default_per_request_limit: Decimal = Field(
        default=Decimal("0.05"), ge=0, description="Default per-request cost limit"
    )
    daily_limit: Decimal = Field(
        default=Decimal("2.00"), ge=0, description="Daily spending limit"
    )
    monthly_limit: Decimal = Field(
        default=Decimal("20.00"), ge=0, description="Monthly spending limit"
    )
    warning_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Warning threshold as fraction of limit",
    )
    currency: str = Field(default="USD", description="Currency for cost calculations")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v):
        valid_currencies = {"USD", "EUR", "GBP", "CAD", "AUD"}
        if v not in valid_currencies:
            raise ValueError(f"Currency must be one of {valid_currencies}")
        return v


class APIConfig(BaseModel):
    """API configuration."""

    timeout: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )
    retries: int = Field(default=2, ge=0, le=10, description="Number of retries")
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    max_backoff: int = Field(
        default=60, ge=1, description="Maximum backoff time in seconds"
    )


class MCPConfig(BaseModel):
    """MCP server configuration."""

    host: str = Field(default="localhost", description="MCP server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="MCP server port")
    dev_mode: bool = Field(default=True, description="Development mode (auto-reload)")
    cors_enabled: bool = Field(default=True, description="Enable CORS")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v):
        # Basic host validation - allow localhost, IP addresses, domain names
        if not v or len(v.strip()) == 0:
            raise ValueError("Host cannot be empty")
        return v.strip()


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    cache_enabled: bool = Field(default=False, description="Enable response caching")
    cache_duration: int = Field(
        default=300, ge=0, description="Cache duration in seconds"
    )
    max_response_size: int = Field(
        default=1048576, ge=1024, description="Maximum response size in bytes"
    )
    connection_pool_size: int = Field(
        default=10, ge=1, le=100, description="HTTP connection pool size"
    )


class AnalyticsConfig(BaseModel):
    """Analytics configuration."""

    enabled: bool = Field(default=True, description="Enable usage analytics")
    anonymize_data: bool = Field(default=True, description="Anonymize collected data")
    retention_days: int = Field(
        default=90, ge=1, description="Analytics data retention in days"
    )
    export_enabled: bool = Field(default=True, description="Enable data export")


class PricingConfig(BaseModel):
    """Pricing configuration."""

    enabled: bool = Field(default=True, description="Enable dynamic pricing updates")
    cache_ttl_hours: int = Field(
        default=1, ge=1, le=168, description="Pricing cache TTL in hours"
    )
    fetch_timeout: float = Field(
        default=30.0, ge=5.0, le=300.0, description="HTTP fetch timeout in seconds"
    )
    auto_update_on_startup: bool = Field(
        default=True, description="Automatically update pricing on startup"
    )
    fallback_conservative: bool = Field(
        default=True, description="Use conservative estimates for unknown models"
    )
    backup_file_path: str | None = Field(
        default=None, description="Custom backup pricing file path"
    )


class DevelopmentConfig(BaseModel):
    """Development-specific configuration."""

    debug_mode: bool = Field(default=False, description="Enable debug mode")
    mock_apis: bool = Field(default=False, description="Use mock API clients")
    test_data_enabled: bool = Field(
        default=False, description="Enable test data generation"
    )
    profiling_enabled: bool = Field(
        default=False, description="Enable performance profiling"
    )


class AppSettings(BaseSettings):
    """Main application settings using environment variables."""

    # Application info
    app_name: str = Field(default="Second Opinion", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development",
        description="Environment (development/staging/production)",
    )

    # API Keys
    openrouter_api_key: str | None = Field(
        default=None, description="OpenRouter API key", repr=False
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234", description="LM Studio base URL"
    )

    # Security
    database_encryption_key: str | None = Field(
        default=None, description="Database encryption key", repr=False
    )
    session_encryption_key: str | None = Field(
        default=None, description="Session encryption key", repr=False
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Directories
    data_dir: str = Field(default="./data", description="Data directory")
    config_dir: str = Field(default="./config", description="Configuration directory")
    prompts_dir: str = Field(default="./prompts", description="Prompts directory")

    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cost_management: CostManagementConfig = Field(default_factory=CostManagementConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid_environments = {"development", "staging", "production"}
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def validate_api_keys(self):
        """Validate that OpenRouter API key is configured."""
        if not self.openrouter_api_key and self.environment != "development":
            raise ValueError(
                "OpenRouter API key must be configured in non-development environments"
            )

        return self

    @model_validator(mode="after")
    def validate_encryption_keys(self):
        """Validate encryption keys are present when encryption is enabled."""
        if (
            self.database.encryption_enabled
            and not self.database_encryption_key
            and self.environment == "production"
        ):
            raise ValueError(
                "Database encryption key is required when encryption is enabled in production"
            )

        return self

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider."""
        # All providers use OpenRouter now
        if provider.lower() in ["openrouter", "anthropic", "openai", "google"]:
            return self.openrouter_api_key
        return None

    def has_api_key(self, provider: str) -> bool:
        """Check if API key is configured for provider."""
        return self.get_api_key(provider) is not None

    def get_data_path(self, filename: str) -> Path:
        """Get full path for a data file."""
        return Path(self.data_dir) / filename

    def get_config_path(self, filename: str) -> Path:
        """Get full path for a config file."""
        return Path(self.config_dir) / filename

    def get_prompts_path(self, filename: str) -> Path:
        """Get full path for a prompts file."""
        return Path(self.prompts_dir) / filename


class ConfigurationManager:
    """Manages hierarchical configuration loading and validation."""

    def __init__(self):
        self._settings: AppSettings | None = None
        self._user_config: dict[str, Any] = {}

    def load_configuration(
        self,
        config_path: Path | None = None,
        override_env: dict[str, str] | None = None,
    ) -> AppSettings:
        """
        Load configuration with hierarchy: CLI/override > env vars > user config > defaults.

        Args:
            config_path: Path to user configuration file
            override_env: Environment variable overrides (simulating CLI args)

        Returns:
            Validated AppSettings instance
        """
        # Load user configuration from YAML file
        if config_path and config_path.exists():
            self._user_config = self._load_yaml_config(config_path)

        # Override environment if provided (for CLI args)
        if override_env:
            for key, value in override_env.items():
                os.environ[key] = value

        # Create settings with YAML config as base, env vars will override via pydantic-settings
        init_kwargs = {}
        if self._user_config:
            init_kwargs.update(self._user_config)

        self._settings = AppSettings(**init_kwargs)

        # Validate configuration
        self._validate_configuration()

        return self._settings

    def _load_yaml_config(self, config_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except FileNotFoundError:
            return {}
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}") from e

    def _validate_configuration(self):
        """Perform additional configuration validation."""
        if not self._settings:
            raise ValueError("Configuration not loaded")

        # Validate directories exist or can be created
        self._ensure_directory(self._settings.data_dir)
        self._ensure_directory(self._settings.config_dir)
        self._ensure_directory(self._settings.prompts_dir)

        # Validate API key formats
        self._validate_api_key_formats()

    def _ensure_directory(self, dir_path: str):
        """Ensure directory exists, create if necessary."""
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create directory {dir_path}: {e}") from e

    def _validate_api_key_formats(self):
        """Validate API key formats for security."""
        if not self._settings:
            return

        # Only validate OpenRouter key format
        if self._settings.openrouter_api_key:
            import re

            if not re.match(r"^sk-or-.*", self._settings.openrouter_api_key):
                print("Warning: OpenRouter API key should start with 'sk-or-'")

    @property
    def settings(self) -> AppSettings:
        """Get current settings (load default if not loaded)."""
        if self._settings is None:
            self._settings = self.load_configuration()
        return self._settings

    def reset(self):
        """Reset the configuration manager (useful for testing)."""
        self._settings = None
        self._user_config = {}

    def update_setting(self, path: str, value: Any):
        """Update a specific setting using dot notation (e.g., 'cost_management.daily_limit')."""
        if not self._settings:
            raise ValueError("Configuration not loaded")

        parts = path.split(".")
        obj = self._settings

        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Set the final value
        setattr(obj, parts[-1], value)

    def export_config_template(self, output_path: Path):
        """Export a configuration template file."""
        template = {
            "app_name": "Second Opinion",
            "environment": "development",
            "database": {
                "path": "data/second_opinion.db",
                "encryption_enabled": True,
                "backup_enabled": True,
                "retention_days": 30,
            },
            "security": {
                "input_sanitization": True,
                "response_filtering": True,
                "rate_limit_per_minute": 60,
                "max_concurrent_requests": 5,
            },
            "cost_management": {
                "default_per_request_limit": "0.05",
                "daily_limit": "2.00",
                "monthly_limit": "20.00",
                "warning_threshold": 0.80,
            },
            "api": {"timeout": 30, "retries": 2, "backoff_factor": 2.0},
            "mcp": {"host": "localhost", "port": 8000, "dev_mode": True},
            "analytics": {
                "enabled": True,
                "anonymize_data": True,
                "retention_days": 90,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_settings() -> AppSettings:
    """Get the current application settings."""
    return config_manager.settings


def load_config(config_path: Path | None = None) -> AppSettings:
    """Load configuration from file and environment."""
    return config_manager.load_configuration(config_path)
