"""
Tests for the should_upgrade MCP tool.

This module tests the `should_upgrade_tool` function that analyzes whether
premium model alternatives could provide quality improvements that justify
additional cost.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from second_opinion.core.models import ModelResponse, TaskComplexity, TokenUsage
from second_opinion.mcp.tools.should_upgrade import should_upgrade_tool


class TestShouldUpgradeTool:
    """Test cases for the should_upgrade MCP tool."""

    @pytest.fixture
    def sample_response(self):
        """Sample response for upgrade analysis."""
        return """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

    @pytest.fixture
    def sample_task(self):
        """Sample task for upgrade analysis."""
        return "Write a Python function to calculate fibonacci"

    @pytest.fixture
    def mock_evaluator(self):
        """Mock evaluator for testing."""
        evaluator = MagicMock()
        evaluator.classify_task_complexity = AsyncMock(
            return_value=TaskComplexity.MODERATE
        )
        evaluator.compare_responses = AsyncMock()
        return evaluator

    @pytest.fixture
    def mock_cost_guard(self):
        """Mock cost guard for testing."""
        cost_guard = MagicMock()
        cost_guard.check_and_reserve_budget = AsyncMock()
        cost_guard.record_actual_cost = AsyncMock()
        return cost_guard

    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        client = MagicMock()
        client.estimate_cost = AsyncMock(return_value=Decimal("0.02"))
        client.complete = AsyncMock()
        return client

    async def test_basic_upgrade_functionality(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test basic upgrade functionality with auto-selection."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        upgrade_response = ModelResponse(
            content="def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.05"),
            provider="openrouter",
        )
        mock_client.complete.return_value = upgrade_response

        # Mock evaluation result showing upgrade is better
        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"  # upgrade candidate wins
        mock_evaluation_result.overall_score = 8.5
        mock_evaluation_result.reasoning = (
            "Upgrade provides better algorithm efficiency"
        )
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test the tool
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            include_premium=True,
        )

        # Verify the result
        assert "Should You Upgrade? Quality Enhancement Analysis" in result
        assert "**Current Model**: anthropic/claude-3-haiku" in result
        assert "Premium Alternatives Tested" in result
        assert "My Recommendation" in result
        assert "Next Steps" in result

        # Verify cost tracking was called
        mock_cost_guard.check_and_reserve_budget.assert_called_once()
        mock_cost_guard.record_actual_cost.assert_called_once()

        # Verify evaluation was performed
        mock_evaluator.compare_responses.assert_called()

    async def test_premium_model_inclusion(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test that premium models are included when include_premium=True."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        premium_response = ModelResponse(
            content="Enhanced fibonacci implementation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.08"),
            provider="openrouter",
        )
        mock_client.complete.return_value = premium_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 9.0
        mock_evaluation_result.reasoning = (
            "Premium model provides superior implementation"
        )
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test with premium models enabled
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="openai/gpt-4o-mini",
            include_premium=True,
        )

        # Verify premium models are mentioned
        assert "**Testing Premium Models**: Yes" in result
        assert "premium" in result.lower()

    async def test_premium_model_disabled(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test that premium models are excluded when include_premium=False."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        mid_tier_response = ModelResponse(
            content="Improved fibonacci implementation",
            model="anthropic/claude-3-5-sonnet",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.03"),
            provider="openrouter",
        )
        mock_client.complete.return_value = mid_tier_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 7.5
        mock_evaluation_result.reasoning = "Mid-tier model provides good improvement"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test with premium models disabled
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            include_premium=False,
        )

        # Verify premium models are not prioritized
        assert "**Testing Premium Models**: No" in result

    async def test_local_model_upgrade(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test upgrading from a local model to cloud alternatives."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )

        def mock_provider_detection(model):
            if "mlx" in model:
                return "lmstudio"
            return "openrouter"

        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            mock_provider_detection,
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        cloud_response = ModelResponse(
            content="Professional fibonacci implementation with error handling",
            model="anthropic/claude-3-5-sonnet",
            usage=TokenUsage(input_tokens=50, output_tokens=120, total_tokens=170),
            cost_estimate=Decimal("0.04"),
            provider="openrouter",
        )
        mock_client.complete.return_value = cloud_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.0
        mock_evaluation_result.reasoning = (
            "Cloud model provides better error handling and documentation"
        )
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test upgrading from local model
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="qwen3-4b-mlx",
            include_premium=True,
        )

        # Verify upgrade analysis from local model
        assert "**Current Model**: qwen3-4b-mlx" in result
        assert "Cost vs Quality Analysis" in result
        # Local model should show some cost increase to cloud alternatives
        assert "increase" in result.lower() or "upgrade" in result.lower()

    async def test_budget_model_upgrade(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test upgrading from a budget model to premium alternatives."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        premium_response = ModelResponse(
            content="Highly optimized fibonacci with memoization and comprehensive documentation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=150, total_tokens=200),
            cost_estimate=Decimal("0.10"),
            provider="openrouter",
        )
        mock_client.complete.return_value = premium_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 9.2
        mock_evaluation_result.reasoning = (
            "Premium model provides exceptional optimization and documentation"
        )
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test upgrading from budget model
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="openai/gpt-4o-mini",
            include_premium=True,
        )

        # Verify upgrade analysis shows quality improvement
        assert "**Current Model**: openai/gpt-4o-mini" in result
        assert "9.2/10" in result
        assert "UPGRADE" in result or "exceptional" in result.lower()

    async def test_code_snippet_acceptance(
        self, mock_evaluator, mock_cost_guard, mock_client, monkeypatch
    ):
        """Test that code snippets are accepted without security issues."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        upgrade_response = ModelResponse(
            content="Enhanced code implementation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=100, output_tokens=150, total_tokens=250),
            cost_estimate=Decimal("0.08"),
            provider="openrouter",
        )
        mock_client.complete.return_value = upgrade_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.0
        mock_evaluation_result.reasoning = "Code quality significantly improved"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test with code snippet containing various programming elements
        code_response = """
import asyncio
import requests

async def fetch_data(url):
    response = requests.get(url)
    return response.json()

# This function demonstrates various Python features
def complex_function(data, callback=None):
    result = []
    for item in data:
        if callback:
            processed = callback(item)
        else:
            processed = item * 2
        result.append(processed)
    return result
"""

        result = await should_upgrade_tool(
            current_response=code_response,
            task="Write a Python function for async data fetching",
            current_model="anthropic/claude-3-haiku",
        )

        # Should not trigger security warnings
        assert "❌" not in result
        assert "Security" not in result
        assert "Should You Upgrade?" in result

    async def test_cost_limit_enforcement(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test that cost limit enforcement works correctly."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup cost guard to reject the budget request
        mock_cost_guard.check_and_reserve_budget.side_effect = Exception(
            "Cost limit exceeded: $0.20 > $0.10"
        )

        # Test with low cost limit
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            cost_limit=0.10,
        )

        # Should return budget error
        assert "❌ **Budget Error**" in result
        assert "Cost limit exceeded" in result

    async def test_no_current_model_specified(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test behavior when no current model is specified."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        upgrade_response = ModelResponse(
            content="Enhanced implementation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.06"),
            provider="openrouter",
        )
        mock_client.complete.return_value = upgrade_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.0
        mock_evaluation_result.reasoning = "Upgrade provides better quality"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test without specifying current model
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model=None,
            include_premium=True,
        )

        # Should use default baseline model
        assert "**Current Model**: anthropic/claude-3-haiku" in result
        assert "Should You Upgrade?" in result

    async def test_complex_task_handling(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test handling of complex tasks that benefit from premium models."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks for complex task
        mock_evaluator.classify_task_complexity = AsyncMock(
            return_value=TaskComplexity.COMPLEX
        )
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        complex_upgrade_response = ModelResponse(
            content="Sophisticated implementation with advanced optimization",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300),
            cost_estimate=Decimal("0.12"),
            provider="openrouter",
        )
        mock_client.complete.return_value = complex_upgrade_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 9.5
        mock_evaluation_result.reasoning = (
            "Premium model excels at complex optimization tasks"
        )
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test with complex task
        complex_task = "Implement a distributed system with fault tolerance, load balancing, and real-time monitoring"

        result = await should_upgrade_tool(
            current_response="Basic implementation",
            task=complex_task,
            current_model="openai/gpt-4o-mini",
            include_premium=True,
        )

        # Should recognize complexity and recommend upgrade
        assert "**Task Complexity**: complex" in result
        assert "9.5/10" in result
        assert "UPGRADE" in result or "premium" in result.lower()

    async def test_quality_improvement_calculations(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test that quality improvement calculations are accurate."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        # Create upgrade response
        upgrade_response = ModelResponse(
            content="Premium quality implementation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.08"),
            provider="openrouter",
        )
        mock_client.complete.return_value = upgrade_response

        # Mock different quality scores
        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.7
        mock_evaluation_result.reasoning = "Significant quality improvement"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            include_premium=True,
        )

        # Should show quality improvement metrics
        assert "8.7/10" in result
        assert (
            "significant improvement" in result.lower() or "excellent" in result.lower()
        )
        assert "Cost vs Quality Analysis" in result

    async def test_custom_upgrade_candidates(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test using custom upgrade candidates instead of auto-selection."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        custom_response = ModelResponse(
            content="Custom model implementation",
            model="openai/gpt-4o",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.06"),
            provider="openrouter",
        )
        mock_client.complete.return_value = custom_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.2
        mock_evaluation_result.reasoning = "Custom model provides good upgrade"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        # Test with custom upgrade candidates
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            upgrade_candidates=["openai/gpt-4o", "google/gemini-pro-1.5"],
            include_premium=True,
        )

        # Should use custom candidates
        assert "openai/gpt-4o" in result
        assert "8.2/10" in result

    async def test_error_handling(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test error handling when upgrade candidate fails."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        # Make client fail
        mock_client.complete.side_effect = Exception("API error")

        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            include_premium=True,
        )

        # Should handle error gracefully
        assert "Should You Upgrade?" in result
        assert "Error: Failed to get response" in result

    async def test_monthly_cost_projection(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test monthly cost projections in upgrade analysis."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        expensive_response = ModelResponse(
            content="Premium implementation",
            model="anthropic/claude-3-opus",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.10"),  # Significantly more expensive
            provider="openrouter",
        )
        mock_client.complete.return_value = expensive_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.5
        mock_evaluation_result.reasoning = "Premium quality justifies cost"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",  # Budget model
            include_premium=True,
        )

        # Should show monthly cost projections
        assert "Monthly Costs (100 requests)" in result
        assert "Current:" in result
        assert "With" in result and "claude-3-opus" in result

    async def test_actionable_recommendations(
        self,
        sample_response,
        sample_task,
        mock_evaluator,
        mock_cost_guard,
        mock_client,
        monkeypatch,
    ):
        """Test that actionable recommendations are provided."""
        # Mock dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_evaluator",
            lambda: mock_evaluator,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.get_cost_guard",
            lambda: mock_cost_guard,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.create_client_from_config",
            lambda x: mock_client,
        )
        monkeypatch.setattr(
            "second_opinion.mcp.tools.should_upgrade.detect_model_provider",
            lambda x: "openrouter",
        )

        # Setup mocks
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test_reservation"
        )

        upgrade_response = ModelResponse(
            content="Quality upgrade implementation",
            model="anthropic/claude-3-5-sonnet",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.04"),
            provider="openrouter",
        )
        mock_client.complete.return_value = upgrade_response

        mock_evaluation_result = MagicMock()
        mock_evaluation_result.winner = "primary"
        mock_evaluation_result.overall_score = 8.0
        mock_evaluation_result.reasoning = "Solid quality improvement"
        mock_evaluator.compare_responses.return_value = mock_evaluation_result

        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            include_premium=True,
        )

        # Should provide actionable next steps
        assert "Next Steps" in result
        assert "1." in result and "2." in result  # Numbered steps
        assert "Trial" in result or "Test" in result or "Consider" in result

    async def test_invalid_current_model(self, sample_response, sample_task):
        """Test handling of invalid current model."""
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="@invalid-model-with-special-chars!",  # Use clearly invalid format with special chars
            include_premium=True,
        )

        # Should return validation error with suggestions
        assert "❌ **Invalid Current Model**" in result
        assert "Suggested formats" in result

    async def test_invalid_upgrade_candidates(self, sample_response, sample_task):
        """Test handling of invalid upgrade candidates."""
        result = await should_upgrade_tool(
            current_response=sample_response,
            task=sample_task,
            current_model="anthropic/claude-3-haiku",
            upgrade_candidates=[
                "@invalid-model!",
                "*another-invalid*",
            ],  # Use clearly invalid formats with special chars
            include_premium=True,
        )

        # Should return validation error
        assert "❌ **Invalid Upgrade Candidates**" in result
        assert "Invalid model candidate" in result
