"""
Unit tests for the MCP should_downgrade tool.

These tests focus on the tool's logic, validation, and error handling
using mocked dependencies.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from second_opinion.core.models import TaskComplexity
from second_opinion.mcp.tools.should_downgrade import (
    _select_downgrade_candidates,
    should_downgrade_tool,
)


class TestShouldDowngradeToolUnit:
    """Unit tests for the should_downgrade tool with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_input_validation_invalid_model(self):
        """Test that invalid model names are rejected."""
        result = await should_downgrade_tool(
            current_response="Test response",
            task="Test task",
            current_model="invalid model name!",
            cost_limit=0.10,
        )

        assert "❌ **Invalid Current Model**" in result
        assert "invalid model name!" in result

    @pytest.mark.asyncio
    async def test_input_validation_invalid_candidates(self):
        """Test that invalid downgrade candidates are rejected."""
        result = await should_downgrade_tool(
            current_response="Test response",
            task="Test task",
            current_model="anthropic/claude-3-5-sonnet",
            downgrade_candidates=["invalid model!", "also invalid!"],
            cost_limit=0.10,
        )

        assert "❌ **Invalid Downgrade Candidate #1**" in result
        assert "invalid model!" in result

    @pytest.mark.asyncio
    async def test_cost_limit_validation(self):
        """Test that invalid cost limits are handled."""
        with patch(
            "second_opinion.mcp.tools.should_downgrade.get_cost_guard"
        ) as mock_cost_guard:
            mock_guard = AsyncMock()
            mock_cost_guard.return_value = mock_guard
            mock_guard.check_and_reserve_budget.side_effect = Exception(
                "Cost limit exceeded"
            )

            result = await should_downgrade_tool(
                current_response="Test response",
                task="Test task",
                current_model="anthropic/claude-3-5-sonnet",
                cost_limit=0.001,  # Very small limit to trigger budget error
            )

            assert "❌ **Budget Error**" in result
            assert "Cost limit exceeded" in result

    def test_select_downgrade_candidates_premium_models(self):
        """Test downgrade candidate selection for premium models."""
        # Test Claude premium -> budget downgrade
        candidates = _select_downgrade_candidates(
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            task_complexity=TaskComplexity.MODERATE,
        )

        assert "qwen3-0.6b-mlx" in candidates  # Local model
        assert "anthropic/claude-3-haiku" in candidates  # Budget cloud alternative

        # Test GPT premium -> budget downgrade
        candidates = _select_downgrade_candidates(
            current_model="openai/gpt-4o",
            test_local=True,
            task_complexity=TaskComplexity.MODERATE,
        )

        assert "qwen3-0.6b-mlx" in candidates  # Local model
        assert "openai/gpt-4o-mini" in candidates  # Budget cloud alternative

    def test_select_downgrade_candidates_local_disabled(self):
        """Test downgrade candidate selection with local models disabled."""
        candidates = _select_downgrade_candidates(
            current_model="anthropic/claude-3-5-sonnet",
            test_local=False,
            task_complexity=TaskComplexity.MODERATE,
        )

        # Should not include local models
        assert not any("mlx" in model for model in candidates)
        assert not any("codestral" in model for model in candidates)

        # Should include budget cloud alternatives
        assert "anthropic/claude-3-haiku" in candidates

    def test_select_downgrade_candidates_complex_tasks(self):
        """Test downgrade candidate selection for complex tasks."""
        candidates = _select_downgrade_candidates(
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            task_complexity=TaskComplexity.COMPLEX,
        )

        # For complex tasks, should prefer mid-tier local models
        assert "qwen3-4b-mlx" in candidates
        # Should still include the smallest model as an option
        assert len(candidates) > 0

    def test_select_downgrade_candidates_max_limit(self):
        """Test that downgrade candidate selection respects max limit."""
        candidates = _select_downgrade_candidates(
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            task_complexity=TaskComplexity.MODERATE,
            max_candidates=2,
        )

        # Should not exceed max_candidates limit
        assert len(candidates) <= 2
        assert len(candidates) > 0

    @pytest.mark.asyncio
    async def test_sanitization_security_context(self):
        """Test that inputs are properly sanitized."""
        # This test verifies that the sanitization functions are called
        # The actual sanitization logic is tested in test_sanitization.py

        with patch(
            "second_opinion.mcp.tools.should_downgrade.sanitize_prompt"
        ) as mock_sanitize:
            mock_sanitize.return_value = "clean input"

            with patch(
                "second_opinion.mcp.tools.should_downgrade.get_cost_guard"
            ) as mock_cost_guard:
                mock_guard = AsyncMock()
                mock_cost_guard.return_value = mock_guard
                mock_guard.check_and_reserve_budget.side_effect = Exception(
                    "Skip execution"
                )

                try:
                    await should_downgrade_tool(
                        current_response="<script>alert('test')</script>",
                        task="Test task",
                        current_model="anthropic/claude-3-5-sonnet",
                        cost_limit=0.10,
                    )
                except Exception:  # noqa: S110
                    # Expected: Either mocked budget check or OpenRouter API error
                    pass

                # Verify sanitization was called for both inputs
                assert mock_sanitize.call_count >= 2

    @pytest.mark.asyncio
    async def test_default_cost_limit(self):
        """Test that default cost limit is applied when none provided."""
        with patch(
            "second_opinion.mcp.tools.should_downgrade.get_cost_guard"
        ) as mock_cost_guard:
            mock_guard = AsyncMock()
            mock_cost_guard.return_value = mock_guard
            mock_guard.check_and_reserve_budget.side_effect = Exception(
                "Budget check called"
            )

            try:
                await should_downgrade_tool(
                    current_response="Test response",
                    task="Test task",
                    current_model="anthropic/claude-3-5-sonnet",
                    # No cost_limit provided - should use default
                )
            except Exception as e:
                assert "Budget check called" in str(e)

            # Verify budget check was called with some cost limit
            mock_guard.check_and_reserve_budget.assert_called_once()
            args = mock_guard.check_and_reserve_budget.call_args
            assert len(args[0]) >= 3  # estimated_cost, tool_name, model
            # The per_request_override should be the default Decimal("0.15")
            assert args[1]["per_request_override"] == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_current_model_inference(self):
        """Test that current model is inferred when not provided."""
        with patch(
            "second_opinion.mcp.tools.should_downgrade.get_cost_guard"
        ) as mock_cost_guard:
            mock_guard = AsyncMock()
            mock_cost_guard.return_value = mock_guard
            mock_guard.check_and_reserve_budget.side_effect = Exception(
                "Skip execution"
            )

            try:
                await should_downgrade_tool(
                    current_response="Test response",
                    task="Test task",
                    # No current_model provided - should use default
                    cost_limit=0.10,
                )
            except Exception:  # noqa: S110
                # Expected: Either mocked budget check or OpenRouter API error
                pass

            # The tool should have inferred a default model
            # Check the call was made (indicating the model was set)
            mock_guard.check_and_reserve_budget.assert_called_once()
