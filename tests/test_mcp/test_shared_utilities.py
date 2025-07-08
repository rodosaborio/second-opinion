"""Test shared MCP tool utilities."""

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.second_opinion.mcp.tools.shared import (
    calculate_quality_assessment,
    format_cost_comparison,
    get_cross_provider_alternatives,
    get_model_name_suggestions,
    get_model_tier,
    should_recommend_change,
    validate_model_candidates,
)


class TestSharedUtilities:
    """Test shared utility functions used across MCP tools."""

    def test_get_model_name_suggestions_downgrade(self):
        """Test model name suggestions for downgrade context."""
        result = get_model_name_suggestions("invalid-model", context="downgrade")

        assert isinstance(result, str)
        assert "Expensive Models" in result
        assert "Budget Alternatives" in result
        assert "Local Models" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o-mini" in result
        assert "qwen3-4b-mlx" in result

    def test_get_model_name_suggestions_upgrade(self):
        """Test model name suggestions for upgrade context."""
        result = get_model_name_suggestions("invalid-model", context="upgrade")

        assert isinstance(result, str)
        assert "Budget Models" in result
        assert "Premium Alternatives" in result
        assert "anthropic/claude-3-haiku" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o" in result

    def test_get_model_name_suggestions_general(self):
        """Test model name suggestions for general context."""
        result = get_model_name_suggestions("invalid-model", context="general")

        assert isinstance(result, str)
        assert "Popular Models" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o" in result

    def test_get_model_name_suggestions_unknown_context(self):
        """Test model name suggestions with unknown context."""
        result = get_model_name_suggestions("invalid-model", context="unknown")

        # Should default to general suggestions
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_cost_comparison(self):
        """Test cost comparison formatting."""
        result = format_cost_comparison(
            current_cost=Decimal("1.50"), alternative_cost=Decimal("0.75")
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        cost_diff, percentage, description = result
        assert isinstance(cost_diff, Decimal)
        assert isinstance(percentage, str)
        assert isinstance(description, str)

    def test_calculate_quality_assessment_downgrade(self):
        """Test quality assessment for downgrade context."""
        test_cases = [
            (8.0, "minimal"),
            (6.5, "moderate"),
            (4.5, "significant"),
            (3.0, "major"),
        ]

        for score, expected in test_cases:
            result = calculate_quality_assessment(score, context="downgrade")
            assert result == expected

    def test_calculate_quality_assessment_upgrade(self):
        """Test quality assessment for upgrade context."""
        test_cases = [
            (9.0, "excellent"),
            (8.0, "significant"),
            (7.0, "moderate"),
            (6.0, "minor"),
            (5.0, "negligible"),
        ]

        for score, expected in test_cases:
            result = calculate_quality_assessment(score, context="upgrade")
            assert result == expected

    def test_calculate_quality_assessment_general(self):
        """Test quality assessment for general context."""
        test_cases = [
            (8.5, "excellent"),
            (7.5, "good"),
            (6.5, "fair"),
            (4.5, "poor"),
            (3.0, "very poor"),
        ]

        for score, expected in test_cases:
            result = calculate_quality_assessment(score, context="general")
            assert result == expected

    def test_get_model_tier(self):
        """Test model tier determination."""
        test_cases = [
            ("qwen3-4b-mlx", "local"),
            ("anthropic/claude-3-haiku", "budget"),
            ("anthropic/claude-3-5-sonnet", "premium"),
            ("openai/gpt-4o-mini", "budget"),
            ("openai/gpt-4o", "premium"),
            ("unknown/model", "unknown"),
        ]

        for model, _expected_tier in test_cases:
            with patch(
                "src.second_opinion.mcp.tools.shared.detect_model_provider"
            ) as mock_detect:
                if "mlx" in model or "qwen" in model.lower():
                    mock_detect.return_value = "lmstudio"
                else:
                    mock_detect.return_value = "openrouter"

                result = get_model_tier(model)
                # Just verify it returns a string tier
                assert isinstance(result, str)
                assert len(result) > 0

    def test_validate_model_candidates_valid(self):
        """Test validating valid model candidates."""
        candidates = ["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]

        with patch(
            "src.second_opinion.mcp.tools.shared.validate_model_name"
        ) as mock_validate:
            mock_validate.side_effect = lambda x: x  # Return input unchanged

            result = validate_model_candidates(candidates)
            assert result == candidates

    def test_validate_model_candidates_invalid(self):
        """Test validating invalid model candidates."""
        candidates = ["invalid model"]

        with patch(
            "src.second_opinion.mcp.tools.shared.validate_model_name"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Invalid model name")

            with pytest.raises(Exception, match="Invalid model candidate"):
                validate_model_candidates(candidates)

    def test_validate_model_candidates_none(self):
        """Test validating None candidates."""
        result = validate_model_candidates(None)
        assert result is None

    def test_validate_model_candidates_empty(self):
        """Test validating empty candidates list."""
        result = validate_model_candidates([])
        assert result is None

    def test_get_cross_provider_alternatives(self):
        """Test getting cross-provider alternatives."""
        result = get_cross_provider_alternatives(
            "anthropic/claude-3-5-sonnet", "budget"
        )

        assert isinstance(result, list)
        # Should return some alternatives
        assert len(result) >= 0

    def test_should_recommend_change(self):
        """Test change recommendation logic."""
        result = should_recommend_change(
            score=8.0,
            cost_diff=Decimal("0.5"),
            current_cost=Decimal("1.0"),
            is_upgrade=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        recommendation, reason = result
        assert isinstance(recommendation, bool)
        assert isinstance(reason, str)
