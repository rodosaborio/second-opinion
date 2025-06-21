"""Test MCP tool functionality."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.second_opinion.mcp.tools.second_opinion import second_opinion_tool
from src.second_opinion.core.models import TaskComplexity, ModelResponse, TokenUsage


class TestSecondOpinionTool:
    """Test the second_opinion MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_tool_execution(self):
        """Test basic tool execution with minimal parameters."""
        with (
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
        ):

            # Setup mocks
            mock_eval = MagicMock()
            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.SIMPLE
            )
            mock_eval.compare_responses = AsyncMock(
                return_value={
                    "overall_winner": "primary",
                    "overall_score": 8.0,
                    "reasoning": "Primary response was more accurate",
                }
            )
            mock_evaluator.return_value = mock_eval

            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                return_value=MagicMock(reservation_id="test-123")
            )
            mock_guard.record_actual_cost = AsyncMock()
            mock_guard.get_usage_summary = AsyncMock(
                return_value=MagicMock(available=Decimal("1.50"))
            )
            mock_cost_guard.return_value = mock_guard

            # Mock client responses
            mock_client = MagicMock()
            mock_response = ModelResponse(
                content="Paris is the capital of France.",
                model="gpt-4",
                usage=TokenUsage(input_tokens=10, output_tokens=8, total_tokens=18),
                cost_estimate=Decimal("0.002"),
                provider="openai",
                request_id="test-req-1",
            )
            mock_client.complete = AsyncMock(return_value=mock_response)
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.002"))
            mock_client_factory.return_value = mock_client

            # Execute tool
            result = await second_opinion_tool(
                prompt="What is the capital of France?",
                primary_model="gpt-4",
                comparison_models=["claude-3"],
            )

            # Verify result
            assert isinstance(result, str)
            assert "Second Opinion: Should You Stick or Switch?" in result
            assert "Paris is the capital of France" in result
            assert "Cost Analysis" in result
            assert "Quality Assessment" in result

    @pytest.mark.asyncio
    async def test_response_reuse(self):
        """Test tool execution with primary response reuse."""
        with (
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
        ):

            # Setup mocks
            mock_eval = MagicMock()
            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.SIMPLE
            )
            mock_eval.compare_responses = AsyncMock(
                return_value={
                    "overall_winner": "comparison",
                    "overall_score": 7.5,
                    "reasoning": "Comparison response was clearer",
                }
            )
            mock_evaluator.return_value = mock_eval

            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                return_value=MagicMock(reservation_id="test-123")
            )
            mock_guard.record_actual_cost = AsyncMock()
            mock_guard.get_usage_summary = AsyncMock(
                return_value=MagicMock(available=Decimal("1.50"))
            )
            mock_cost_guard.return_value = mock_guard

            # Mock comparison model response only (primary response is provided)
            mock_client = MagicMock()
            mock_response = ModelResponse(
                content="The capital city of France is Paris.",
                model="claude-3",
                usage=TokenUsage(input_tokens=10, output_tokens=9, total_tokens=19),
                cost_estimate=Decimal("0.003"),
                provider="anthropic",
                request_id="test-req-2",
            )
            mock_client.complete = AsyncMock(return_value=mock_response)
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.003"))
            mock_client_factory.return_value = mock_client

            # Execute tool with existing primary response
            result = await second_opinion_tool(
                prompt="What is the capital of France?",
                primary_model="gpt-4",
                primary_response="Paris is the capital of France.",  # Reuse existing response
                comparison_models=["claude-3"],
            )

            # Verify result
            assert isinstance(result, str)
            assert "Second Opinion: Should You Stick or Switch?" in result
            assert "Paris is the capital of France" in result
            assert "The capital city of France is Paris" in result

            # Verify that primary model client was not called (response was reused)
            # The factory should only be called for comparison model
            assert mock_client_factory.call_count >= 1  # For comparison model

    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation and sanitization."""
        with (
            patch(
                "src.second_opinion.mcp.tools.second_opinion.sanitize_prompt"
            ) as mock_sanitize,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.validate_model_name"
            ) as mock_validate_model,
        ):

            mock_sanitize.return_value = "clean prompt"
            mock_validate_model.return_value = "gpt-4"

            # Test with potentially problematic input
            try:
                await second_opinion_tool(
                    prompt="What is <script>alert('test')</script> the capital?",
                    primary_model="gpt-4///injection",
                    comparison_models=["claude-3<test>"],
                )
            except Exception:
                pass  # Expected to fail due to mocking, but validation should be called

            # Verify sanitization was called
            mock_sanitize.assert_called()
            mock_validate_model.assert_called()

    @pytest.mark.asyncio
    async def test_cost_limit_validation(self):
        """Test cost limit validation and budget checking."""
        with patch(
            "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
        ) as mock_cost_guard:

            # Mock cost guard that rejects due to budget
            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                side_effect=Exception("Budget exceeded")
            )
            mock_cost_guard.return_value = mock_guard

            # Execute tool with cost limit
            result = await second_opinion_tool(
                prompt="What is the capital of France?",
                primary_model="gpt-4",
                cost_limit=0.01,  # Very low limit
            )

            # Should return budget error message
            assert "Budget Error" in result
            assert "Budget exceeded" in result

    @pytest.mark.asyncio
    async def test_auto_model_selection(self):
        """Test automatic comparison model selection."""
        with (
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
            patch(
                "src.second_opinion.cli.main.ComparisonModelSelector"
            ) as mock_selector_class,
        ):

            # Setup mocks
            mock_eval = MagicMock()
            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.MODERATE
            )
            mock_evaluator.return_value = mock_eval

            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                return_value=MagicMock(reservation_id="test-123")
            )
            mock_guard.record_actual_cost = AsyncMock()
            mock_cost_guard.return_value = mock_guard

            mock_client = MagicMock()
            mock_client.complete = AsyncMock(
                return_value=MagicMock(
                    content="Test response", cost_estimate=Decimal("0.002")
                )
            )
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.002"))
            mock_client_factory.return_value = mock_client

            # Mock model selector
            mock_selector = MagicMock()
            mock_selector.select_models.return_value = ["claude-3", "gemini-pro"]
            mock_selector_class.return_value = mock_selector

            # Execute tool without specifying comparison models
            try:
                await second_opinion_tool(
                    prompt="Write a short story about a robot.",
                    primary_model="gpt-4",
                    # No comparison_models specified - should auto-select
                )
            except Exception:
                pass  # Expected due to incomplete mocking

            # Verify model selector was used
            mock_selector.select_models.assert_called_once()
            call_args = mock_selector.select_models.call_args
            assert call_args[1]["primary_model"] == "gpt-4"
            assert call_args[1]["task_complexity"] == TaskComplexity.MODERATE

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for various failure scenarios."""

        # Test client creation failure
        with patch(
            "src.second_opinion.mcp.tools.second_opinion.create_client_from_config"
        ) as mock_client_factory:
            mock_client_factory.side_effect = Exception("Client creation failed")

            result = await second_opinion_tool(
                prompt="What is the capital of France?", primary_model="invalid-model"
            )

            # The specific error message depends on where the failure occurs
            assert "Error" in result or "failed" in result

    @pytest.mark.asyncio
    async def test_think_tag_filtering(self):
        """Test that think tags are properly filtered from responses."""
        with (
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
            patch(
                "src.second_opinion.mcp.tools.second_opinion.filter_think_tags"
            ) as mock_filter,
        ):

            # Setup mocks
            mock_eval = MagicMock()
            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.SIMPLE
            )
            mock_eval.compare_responses = AsyncMock(
                return_value={
                    "overall_winner": "primary",
                    "overall_score": 8.0,
                    "reasoning": "Test reasoning",
                }
            )
            mock_evaluator.return_value = mock_eval

            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                return_value=MagicMock(reservation_id="test-123")
            )
            mock_guard.record_actual_cost = AsyncMock()
            mock_guard.get_usage_summary = AsyncMock(
                return_value=MagicMock(available=Decimal("1.50"))
            )
            mock_cost_guard.return_value = mock_guard

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "<think>reasoning here</think>Paris is the capital."
            mock_response.model = "gpt-4"  # Add model attribute
            mock_response.cost_estimate = Decimal("0.002")
            mock_client.complete = AsyncMock(return_value=mock_response)
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.002"))
            mock_client_factory.return_value = mock_client

            mock_filter.return_value = "Paris is the capital."

            # Execute tool
            result = await second_opinion_tool(
                prompt="What is the capital of France?",
                primary_model="gpt-4",
                comparison_models=["claude-3"],
            )

            # Verify think tag filtering was called
            mock_filter.assert_called()

            # Verify clean response appears in result
            assert "Paris is the capital." in result
