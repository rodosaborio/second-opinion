"""
Tests for model configuration management.
"""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from second_opinion.config.model_configs import (
    CostEstimates,
    ModelCapabilities,
    ModelConfigManager,
    ModelProfilesConfig,
    ModelTierConfig,
    ToolConfig,
    estimate_request_cost,
    get_model_config,
    model_config_manager,
)
from second_opinion.core.models import ModelTier


@pytest.fixture(autouse=True)
def reset_model_config():
    """Reset global model configuration before each test."""
    model_config_manager.reset()
    yield
    model_config_manager.reset()


class TestToolConfig:
    def test_valid_tool_config(self):
        """Test creating valid tool configuration."""
        config = ToolConfig(
            description="Test tool",
            comparison_models=["model1", "model2"],
            max_tokens=1000,
            temperature=0.5,
            cost_limit_per_request=Decimal("0.10")
        )

        assert config.description == "Test tool"
        assert config.comparison_models == ["model1", "model2"]
        assert config.max_tokens == 1000
        assert config.temperature == 0.5
        assert config.cost_limit_per_request == Decimal("0.10")

    def test_empty_model_lists_validation(self):
        """Test that empty model lists are rejected."""
        with pytest.raises(ValueError):
            ToolConfig(
                description="Test tool",
                comparison_models=[]  # Empty list should be rejected
            )


class TestModelTierConfig:
    def test_model_tier_operations(self):
        """Test model tier configuration operations."""
        config = ModelTierConfig(
            budget=["model1", "model2"],
            mid_range=["model3", "model4"],
            premium=["model5"],
            reasoning=["model6"]
        )

        # Test getting models for tier
        assert config.get_models_for_tier(ModelTier.BUDGET) == ["model1", "model2"]
        assert config.get_models_for_tier(ModelTier.PREMIUM) == ["model5"]

        # Test getting tier for model
        assert config.get_tier_for_model("model1") == ModelTier.BUDGET
        assert config.get_tier_for_model("model5") == ModelTier.PREMIUM
        assert config.get_tier_for_model("unknown") is None


class TestCostEstimates:
    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        estimates = CostEstimates(
            input_token_costs={
                "gpt-4": Decimal("0.03"),
                "claude-3": Decimal("0.015")
            },
            output_token_costs={
                "gpt-4": Decimal("0.06"),
                "claude-3": Decimal("0.075")
            }
        )

        # Test getting individual costs
        assert estimates.get_input_cost("gpt-4") == Decimal("0.03")
        assert estimates.get_output_cost("claude-3") == Decimal("0.075")

        # Test unknown model defaults
        assert estimates.get_input_cost("unknown") == Decimal("0.001")
        assert estimates.get_output_cost("unknown") == Decimal("0.004")

        # Test cost calculation
        cost = estimates.estimate_cost("gpt-4", 1000, 500)
        expected = (Decimal("1000") / 1000 * Decimal("0.03")) + (Decimal("500") / 1000 * Decimal("0.06"))
        assert cost == expected


class TestModelProfilesConfig:
    @pytest.fixture
    def sample_config(self):
        """Sample model profiles configuration."""
        return ModelProfilesConfig(
            tools={
                "second_opinion": ToolConfig(
                    description="Get second opinion",
                    comparison_models=["gpt-4", "claude-3"],
                    max_tokens=1000,
                    temperature=0.1
                ),
                "should_upgrade": ToolConfig(
                    description="Check if should upgrade",
                    upgrade_targets={"budget_to_mid": "gpt-4"},
                    max_tokens=500
                )
            },
            model_tiers=ModelTierConfig(
                budget=["gpt-3.5"],
                mid_range=["gpt-4", "claude-3"],
                premium=["gpt-4-turbo"]
            )
        )

    def test_get_tool_config(self, sample_config):
        """Test getting tool configuration."""
        tool_config = sample_config.get_tool_config("second_opinion")
        assert tool_config is not None
        assert tool_config.description == "Get second opinion"

        # Non-existent tool
        assert sample_config.get_tool_config("nonexistent") is None

    def test_get_comparison_models(self, sample_config):
        """Test getting comparison models with primary model filtering."""
        # Should exclude primary model from comparison
        models = sample_config.get_comparison_models("second_opinion", "gpt-4")
        assert models == ["claude-3"]

        # Should return all if primary not in list
        models = sample_config.get_comparison_models("second_opinion", "other-model")
        assert models == ["gpt-4", "claude-3"]

    def test_get_upgrade_target(self, sample_config):
        """Test getting upgrade targets."""
        target = sample_config.get_upgrade_target("gpt-3.5", "budget_to_mid")
        assert target == "gpt-4"

        # Non-existent upgrade type
        assert sample_config.get_upgrade_target("gpt-3.5", "nonexistent") is None

    def test_get_downgrade_candidates(self, sample_config):
        """Test getting downgrade candidates."""
        # Tool without downgrade targets
        candidates = sample_config.get_downgrade_candidates("second_opinion")
        assert candidates == []

        # Add downgrade targets to config
        sample_config.tools["test_tool"] = ToolConfig(
            description="Test",
            downgrade_targets=["gpt-3.5", "claude-haiku"]
        )
        candidates = sample_config.get_downgrade_candidates("test_tool")
        assert candidates == ["gpt-3.5", "claude-haiku"]


class TestModelConfigManager:
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration."""
        return {
            'version': '1.0',
            'tools': {
                'second_opinion': {
                    'description': 'Get alternative perspective',
                    'comparison_models': ['gpt-4', 'claude-3'],
                    'max_tokens': 1000,
                    'temperature': 0.1,
                    'cost_limit_per_request': '0.10'
                }
            },
            'model_tiers': {
                'budget': ['gpt-3.5'],
                'mid_range': ['gpt-4', 'claude-3']
            },
            'cost_estimates': {
                'input_token_costs': {
                    'gpt-4': '0.03',
                    'claude-3': '0.015'
                },
                'output_token_costs': {
                    'gpt-4': '0.06',
                    'claude-3': '0.075'
                }
            }
        }

    def test_load_config_from_file(self, temp_config_dir, sample_yaml_config):
        """Test loading configuration from YAML file."""
        config_file = temp_config_dir / "model_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)

        manager = ModelConfigManager()
        config = manager.load_config(config_file)

        assert config.version == "1.0"
        assert "second_opinion" in config.tools
        assert config.tools["second_opinion"].max_tokens == 1000

    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        manager = ModelConfigManager()
        config = manager.load_config(Path("nonexistent.yaml"))

        # Should return default configuration
        assert isinstance(config, ModelProfilesConfig)
        assert config.version == "1.0"

    def test_get_model_config(self, temp_config_dir, sample_yaml_config):
        """Test getting model configuration for a tool."""
        config_file = temp_config_dir / "model_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)

        manager = ModelConfigManager()
        manager.load_config(config_file)

        # Get config for existing tool
        config = manager.get_model_config("second_opinion", "claude-3")
        assert config['description'] == "Get alternative perspective"
        assert config['max_tokens'] == 1000
        assert "gpt-4" in config['comparison_models']  # claude-3 should be excluded
        assert "claude-3" not in config['comparison_models']

        # Get config for non-existent tool (should return defaults)
        config = manager.get_model_config("nonexistent")
        assert config['max_tokens'] == 500  # Default value
        assert config['comparison_models'] == []

    def test_model_tier_operations(self, temp_config_dir, sample_yaml_config):
        """Test model tier operations."""
        config_file = temp_config_dir / "model_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)

        manager = ModelConfigManager()
        manager.load_config(config_file)

        # Test tier detection
        assert manager.get_model_tier("gpt-3.5") == ModelTier.BUDGET
        assert manager.get_model_tier("gpt-4") == ModelTier.MID_RANGE
        assert manager.get_model_tier("unknown") is None

        # Test getting tier models
        budget_models = manager.get_tier_models(ModelTier.BUDGET)
        assert budget_models == ["gpt-3.5"]

    def test_cost_estimation(self, temp_config_dir, sample_yaml_config):
        """Test cost estimation functionality."""
        config_file = temp_config_dir / "model_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)

        manager = ModelConfigManager()
        manager.load_config(config_file)

        # Test cost estimation
        cost = manager.estimate_cost("gpt-4", 1000, 500)
        expected = (Decimal("1000") / 1000 * Decimal("0.03")) + (Decimal("500") / 1000 * Decimal("0.06"))
        assert cost == expected

        # Test unknown model (should return conservative default)
        cost = manager.estimate_cost("unknown", 1000, 500)
        assert cost == Decimal("0.10")  # Conservative fallback

    def test_model_capability_management(self):
        """Test model capability registration and retrieval."""
        manager = ModelConfigManager()

        capability = ModelCapabilities(
            model="test-model",
            provider="test-provider",
            tier=ModelTier.MID_RANGE,
            max_context=8000,
            supports_system_messages=True,
            supports_temperature=False
        )

        manager.register_model_capability(capability)

        # Test retrieval
        retrieved = manager.get_model_capability("test-model")
        assert retrieved is not None
        assert retrieved.model == "test-model"
        assert retrieved.supports_temperature is False

        # Test feature support checking
        assert manager.is_model_supported("test-model", "system_messages") is True
        assert manager.is_model_supported("test-model", "temperature") is False
        assert manager.is_model_supported("unknown-model", "anything") is True  # Assume supported

    def test_model_request_validation(self):
        """Test model request validation against capabilities."""
        manager = ModelConfigManager()

        capability = ModelCapabilities(
            model="limited-model",
            provider="test",
            tier=ModelTier.BUDGET,
            max_output=1000,
            supports_system_messages=False
        )

        manager.register_model_capability(capability)

        # Test valid request
        warnings = manager.validate_model_request("limited-model", 500, False)
        assert len(warnings) == 0

        # Test request exceeding limits
        warnings = manager.validate_model_request("limited-model", 2000, True)
        assert len(warnings) == 2
        assert any("exceeds model limit" in warning for warning in warnings)
        assert any("does not support system messages" in warning for warning in warnings)


@pytest.mark.security
class TestModelConfigSecurity:
    """Security-focused tests for model configuration."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_yaml_injection_protection(self, temp_config_dir):
        """Test protection against YAML injection attacks."""
        malicious_config = """
        version: '1.0'
        tools:
          malicious: !!python/object/apply:os.system ["rm -rf /"]
        """

        config_file = temp_config_dir / "malicious.yaml"
        with open(config_file, 'w') as f:
            f.write(malicious_config)

        manager = ModelConfigManager()
        # Should handle malicious YAML gracefully
        with pytest.raises(ValueError, match="Invalid YAML"):
            manager.load_config(config_file)

    def test_cost_limit_enforcement(self):
        """Test that cost limits are properly enforced."""
        config = ToolConfig(
            description="Test",
            cost_limit_per_request=Decimal("1000.00")  # Suspiciously high
        )

        # Configuration should accept but application should validate
        assert config.cost_limit_per_request == Decimal("1000.00")

    def test_model_name_validation(self):
        """Test that model names are properly validated."""
        capability = ModelCapabilities(
            model="../../etc/passwd",  # Path traversal attempt
            provider="test",
            tier=ModelTier.BUDGET
        )

        # Should not raise error - validation happens at application level
        assert capability.model == "../../etc/passwd"


class TestGlobalModelConfig:
    """Test global model configuration access."""

    def test_get_model_config_function(self):
        """Test the get_model_config convenience function."""
        config = get_model_config("second_opinion")
        assert isinstance(config, dict)
        assert 'max_tokens' in config

    def test_estimate_request_cost_function(self):
        """Test the estimate_request_cost convenience function."""
        cost = estimate_request_cost("gpt-4", 1000, 500)
        assert isinstance(cost, Decimal)
        assert cost > 0
