"""
Model-specific configuration management.

This module handles model configurations, capabilities, and tool-specific settings
with dynamic primary model handling.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from ..core.models import ModelTier


class ToolConfig(BaseModel):
    """Configuration for a specific tool."""

    description: str = Field(..., description="Tool description")
    comparison_models: list[str] | None = Field(
        None, description="Models to use for comparison"
    )
    upgrade_targets: dict[str, str] | None = Field(
        None, description="Upgrade target models"
    )
    downgrade_targets: list[str] | None = Field(
        None, description="Downgrade target models"
    )
    evaluator_model: str | None = Field(None, description="Model for evaluation tasks")

    max_tokens: int = Field(
        default=500, ge=1, le=32000, description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )
    cost_limit_per_request: Decimal = Field(
        default=Decimal("0.05"), ge=0, description="Cost limit per request"
    )
    system_prompt_template: str | None = Field(
        None, description="System prompt template name"
    )

    @field_validator("comparison_models", "downgrade_targets")
    @classmethod
    def validate_model_lists(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Model lists cannot be empty if specified")
        return v


class ModelCapabilities(BaseModel):
    """Model capability information."""

    model: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    tier: ModelTier = Field(..., description="Model capability tier")
    max_context: int | None = Field(None, ge=1, description="Maximum context window")
    max_output: int | None = Field(None, ge=1, description="Maximum output tokens")
    supports_system_messages: bool = Field(
        default=True, description="Supports system messages"
    )
    supports_temperature: bool = Field(
        default=True, description="Supports temperature control"
    )
    supports_streaming: bool = Field(
        default=True, description="Supports streaming responses"
    )
    cost_per_input_token: Decimal | None = Field(
        None, ge=0, description="Cost per input token"
    )
    cost_per_output_token: Decimal | None = Field(
        None, ge=0, description="Cost per output token"
    )


class ModelTierConfig(BaseModel):
    """Configuration for model tiers."""

    budget: list[str] = Field(default_factory=list, description="Budget tier models")
    mid_range: list[str] = Field(
        default_factory=list, description="Mid-range tier models"
    )
    premium: list[str] = Field(default_factory=list, description="Premium tier models")
    reasoning: list[str] = Field(
        default_factory=list, description="Reasoning-specialized models"
    )

    def get_models_for_tier(self, tier: ModelTier) -> list[str]:
        """Get list of models for a specific tier."""
        tier_map = {
            ModelTier.BUDGET: self.budget,
            ModelTier.MID_RANGE: self.mid_range,
            ModelTier.PREMIUM: self.premium,
            ModelTier.REASONING: self.reasoning,
        }
        return tier_map.get(tier, [])

    def get_tier_for_model(self, model: str) -> ModelTier | None:
        """Get tier for a specific model."""
        for tier in ModelTier:
            if model in self.get_models_for_tier(tier):
                return tier
        return None


class CostEstimates(BaseModel):
    """Cost estimation data for models."""

    input_token_costs: dict[str, Decimal] = Field(
        default_factory=dict, description="Input token costs per 1K tokens"
    )
    output_token_costs: dict[str, Decimal] = Field(
        default_factory=dict, description="Output token costs per 1K tokens"
    )

    def get_input_cost(self, model: str) -> Decimal:
        """Get input cost per 1K tokens for model."""
        return self.input_token_costs.get(
            model, Decimal("0.001")
        )  # Conservative default

    def get_output_cost(self, model: str) -> Decimal:
        """Get output cost per 1K tokens for model."""
        return self.output_token_costs.get(
            model, Decimal("0.004")
        )  # Conservative default

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Estimate total cost for a request."""
        input_cost = (Decimal(input_tokens) / 1000) * self.get_input_cost(model)
        output_cost = (Decimal(output_tokens) / 1000) * self.get_output_cost(model)
        return input_cost + output_cost


class DefaultSettings(BaseModel):
    """Default settings for tools and models."""

    max_tokens: int = Field(default=500, ge=1, description="Default max tokens")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Default temperature"
    )
    cost_limit_per_request: Decimal = Field(
        default=Decimal("0.05"), ge=0, description="Default cost limit"
    )
    timeout: int = Field(default=30, ge=1, description="Default timeout in seconds")
    retries: int = Field(default=2, ge=0, description="Default retry count")


class ModelProfilesConfig(BaseModel):
    """Complete model profiles configuration."""

    version: str = Field(default="1.0", description="Configuration version")
    tools: dict[str, ToolConfig] = Field(
        default_factory=dict, description="Tool configurations"
    )
    model_tiers: ModelTierConfig = Field(
        default_factory=ModelTierConfig, description="Model tier definitions"
    )
    cost_estimates: CostEstimates = Field(
        default_factory=CostEstimates, description="Cost estimation data"
    )
    defaults: DefaultSettings = Field(
        default_factory=DefaultSettings, description="Default settings"
    )

    def get_tool_config(self, tool_name: str) -> ToolConfig | None:
        """Get configuration for a specific tool."""
        return self.tools.get(tool_name)

    def get_comparison_models(self, tool_name: str, primary_model: str) -> list[str]:
        """Get comparison models for a tool, excluding the primary model."""
        tool_config = self.get_tool_config(tool_name)
        if not tool_config or not tool_config.comparison_models:
            return []

        # Filter out the primary model to avoid self-comparison
        return [
            model for model in tool_config.comparison_models if model != primary_model
        ]

    def get_upgrade_target(
        self, current_model: str, upgrade_type: str = "any_to_reasoning"
    ) -> str | None:
        """Get upgrade target for a model."""
        for tool_config in self.tools.values():
            if (
                tool_config.upgrade_targets
                and upgrade_type in tool_config.upgrade_targets
            ):
                return tool_config.upgrade_targets[upgrade_type]
        return None

    def get_downgrade_candidates(self, tool_name: str) -> list[str]:
        """Get downgrade candidates for a tool."""
        tool_config = self.get_tool_config(tool_name)
        if not tool_config or not tool_config.downgrade_targets:
            return []
        return tool_config.downgrade_targets


class ModelConfigManager:
    """Manages model configurations and capabilities."""

    def __init__(self):
        self._config: ModelProfilesConfig | None = None
        self._capabilities: dict[str, ModelCapabilities] = {}

    def load_config(self, config_path: Path) -> ModelProfilesConfig:
        """Load model configuration from YAML file."""
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            self._config = ModelProfilesConfig.model_validate(config_data)
            return self._config

        except FileNotFoundError:
            # Return default configuration if file doesn't exist
            self._config = ModelProfilesConfig()
            return self._config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in model config: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load model config: {e}")

    def get_model_config(
        self, tool_name: str, primary_model: str | None = None
    ) -> dict[str, Any]:
        """
        Get configuration for a tool with primary model context.

        Args:
            tool_name: Name of the tool
            primary_model: Primary model being used (for comparison selection)

        Returns:
            Dictionary with tool configuration
        """
        if not self._config:
            raise ValueError("Configuration not loaded")

        tool_config = self._config.get_tool_config(tool_name)
        if not tool_config:
            # Return default configuration
            return {
                "max_tokens": self._config.defaults.max_tokens,
                "temperature": self._config.defaults.temperature,
                "cost_limit_per_request": self._config.defaults.cost_limit_per_request,
                "comparison_models": [],
                "downgrade_targets": [],
            }

        # Build configuration dictionary
        config = {
            "description": tool_config.description,
            "max_tokens": tool_config.max_tokens,
            "temperature": tool_config.temperature,
            "cost_limit_per_request": tool_config.cost_limit_per_request,
            "system_prompt_template": tool_config.system_prompt_template,
        }

        # Add tool-specific model selections
        if tool_config.comparison_models:
            config["comparison_models"] = self._config.get_comparison_models(
                tool_name, primary_model or ""
            )

        if tool_config.downgrade_targets:
            config["downgrade_targets"] = tool_config.downgrade_targets

        if tool_config.upgrade_targets:
            config["upgrade_targets"] = tool_config.upgrade_targets

        if tool_config.evaluator_model:
            config["evaluator_model"] = tool_config.evaluator_model

        return config

    def get_model_tier(self, model: str) -> ModelTier | None:
        """Get the tier for a specific model."""
        if not self._config:
            return None
        return self._config.model_tiers.get_tier_for_model(model)

    def get_tier_models(self, tier: ModelTier) -> list[str]:
        """Get all models in a specific tier."""
        if not self._config:
            return []
        return self._config.model_tiers.get_models_for_tier(tier)

    def get_comparison_models(self, tool_name: str, primary_model: str) -> list[str]:
        """Get comparison models for a tool, excluding the primary model."""
        if not self._config:
            self._load_default_config()
        return (
            self._config.get_comparison_models(tool_name, primary_model)
            if self._config
            else []
        )

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Estimate cost for a model request."""
        if not self._config:
            return Decimal("0.10")  # Conservative fallback

        # Use config's cost estimates or conservative fallback for unknown models
        cost = self._config.cost_estimates.estimate_cost(
            model, input_tokens, output_tokens
        )

        # If no specific pricing data exists, return conservative fallback
        if (
            model not in self._config.cost_estimates.input_token_costs
            and model not in self._config.cost_estimates.output_token_costs
        ):
            return Decimal("0.10")

        return cost

    def register_model_capability(self, capability: ModelCapabilities):
        """Register capability information for a model."""
        self._capabilities[capability.model] = capability

    def get_model_capability(self, model: str) -> ModelCapabilities | None:
        """Get capability information for a model."""
        return self._capabilities.get(model)

    def is_model_supported(self, model: str, feature: str) -> bool:
        """Check if a model supports a specific feature."""
        capability = self.get_model_capability(model)
        if not capability:
            return True  # Assume supported if no capability info

        feature_map = {
            "system_messages": capability.supports_system_messages,
            "temperature": capability.supports_temperature,
            "streaming": capability.supports_streaming,
        }

        return feature_map.get(feature, True)

    def validate_model_request(
        self, model: str, max_tokens: int, has_system_prompt: bool
    ) -> list[str]:
        """Validate a model request against capabilities."""
        warnings = []
        capability = self.get_model_capability(model)

        if not capability:
            return warnings  # No capability info, assume valid

        if capability.max_output and max_tokens > capability.max_output:
            warnings.append(
                f"Requested {max_tokens} tokens exceeds model limit of {capability.max_output}"
            )

        if has_system_prompt and not capability.supports_system_messages:
            warnings.append(f"Model {model} does not support system messages")

        return warnings

    @property
    def config(self) -> ModelProfilesConfig:
        """Get current configuration (load default if not loaded)."""
        if self._config is None:
            self._config = ModelProfilesConfig()
        return self._config

    def _load_default_config(self):
        """Load the default configuration file."""
        try:
            from pathlib import Path

            # Try to find config in standard locations
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "model_profiles.yaml"
            if config_path.exists():
                self.load_config(config_path)
            else:
                # Fallback to default configuration
                self._config = ModelProfilesConfig()
        except Exception:
            # Last resort: empty config
            self._config = ModelProfilesConfig()

    def reset(self):
        """Reset the configuration manager (useful for testing)."""
        self._config = None
        self._capabilities = {}


# Global model config manager instance
model_config_manager = ModelConfigManager()


def get_model_config(
    tool_name: str, primary_model: str | None = None
) -> dict[str, Any]:
    """Get model configuration for a tool."""
    # Ensure configuration is loaded (will load defaults if not already loaded)
    _ = model_config_manager.config
    return model_config_manager.get_model_config(tool_name, primary_model)


def load_model_config(config_path: Path) -> ModelProfilesConfig:
    """Load model configuration from file."""
    return model_config_manager.load_config(config_path)


def estimate_request_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Estimate cost for a model request."""
    return model_config_manager.estimate_cost(model, input_tokens, output_tokens)
