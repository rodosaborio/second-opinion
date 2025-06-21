"""Integration tests for MCP server with Claude Code."""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.second_opinion.mcp.server import mcp


class TestMCPIntegration:
    """Test MCP server integration scenarios."""

    @pytest.mark.asyncio
    async def test_server_tool_discovery(self):
        """Test that Claude Code can discover tools."""
        # This simulates what Claude Code does when connecting
        tools = await mcp.get_tools()

        assert isinstance(tools, dict)
        assert "second_opinion" in tools

        # Test tool retrieval
        tool = await mcp.get_tool("second_opinion")
        assert tool is not None

    @pytest.mark.asyncio
    async def test_second_opinion_tool_execution_mock(self):
        """Test second_opinion tool execution with mocked dependencies."""
        # Mock all external dependencies
        with (
            patch(
                "second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
        ):

            # Setup evaluator mock
            mock_eval = MagicMock()

            # Mock task complexity with proper enum value
            from second_opinion.core.evaluator import TaskComplexity

            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.SIMPLE
            )

            # Mock comparison result as object with required attributes
            mock_comparison_result = MagicMock()
            mock_comparison_result.winner = "primary"
            mock_comparison_result.overall_score = 8.0
            mock_comparison_result.reasoning = "Primary response was more accurate"
            mock_eval.compare_responses = AsyncMock(return_value=mock_comparison_result)
            mock_evaluator.return_value = mock_eval

            # Setup cost guard mock
            mock_guard = MagicMock()
            budget_check = MagicMock()
            budget_check.reservation_id = "test-reservation-123"
            mock_guard.check_and_reserve_budget = AsyncMock(return_value=budget_check)
            mock_guard.record_actual_cost = AsyncMock()

            # Mock budget usage
            budget_usage = MagicMock()
            budget_usage.available = Decimal("1.50")
            mock_guard.get_usage_summary = AsyncMock(return_value=budget_usage)
            mock_cost_guard.return_value = mock_guard

            # Setup client mock
            mock_client = MagicMock()

            # Mock model response with proper attributes
            mock_response = MagicMock()
            mock_response.content = "Paris is the capital of France."
            mock_response.model = "gpt-4"  # Add model attribute
            mock_response.cost_estimate = Decimal("0.002")
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 18

            # Mock client with all required methods
            mock_client.complete = AsyncMock(return_value=mock_response)
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.002"))
            mock_client_factory.return_value = mock_client

            # Get the tool and execute it (this is what Claude Code would do)
            tool = await mcp.get_tool("second_opinion")

            # Execute the tool
            result = await tool.run(
                {
                    "prompt": "What is the capital of France?",
                    "primary_model": "gpt-4",
                    "comparison_models": ["claude-3"],
                }
            )

            # Verify result structure (FastMCP returns list of content)
            assert isinstance(result, list)
            assert len(result) >= 1

            # Get the text content
            content = result[0].text if hasattr(result[0], "text") else str(result[0])

            assert "Second Opinion: Should You Stick or Switch?" in content
            assert "Paris is the capital of France" in content
            assert "Cost Analysis" in content
            assert "Quality Assessment" in content
            assert "My Recommendation" in content

            # Verify that external systems were called correctly
            mock_eval.classify_task_complexity.assert_called_once()
            mock_guard.check_and_reserve_budget.assert_called_once()
            mock_guard.record_actual_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_reuse_feature(self):
        """Test the response reuse feature for cost optimization."""
        with (
            patch(
                "second_opinion.mcp.tools.second_opinion.get_evaluator"
            ) as mock_evaluator,
            patch(
                "second_opinion.mcp.tools.second_opinion.get_cost_guard"
            ) as mock_cost_guard,
            patch(
                "second_opinion.mcp.tools.second_opinion.create_client_from_config"
            ) as mock_client_factory,
        ):

            # Setup mocks (similar to above)
            mock_eval = MagicMock()

            # Mock task complexity with proper enum value
            from second_opinion.core.evaluator import TaskComplexity

            mock_eval.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.SIMPLE
            )

            # Mock comparison result as object with required attributes
            mock_comparison_result = MagicMock()
            mock_comparison_result.winner = "comparison"
            mock_comparison_result.overall_score = 7.5
            mock_comparison_result.reasoning = "Comparison response was clearer"
            mock_eval.compare_responses = AsyncMock(return_value=mock_comparison_result)
            mock_evaluator.return_value = mock_eval

            mock_guard = MagicMock()
            budget_check = MagicMock()
            budget_check.reservation_id = "test-reservation-456"
            mock_guard.check_and_reserve_budget = AsyncMock(return_value=budget_check)
            mock_guard.record_actual_cost = AsyncMock()
            budget_usage = MagicMock()
            budget_usage.available = Decimal("1.50")
            mock_guard.get_usage_summary = AsyncMock(return_value=budget_usage)
            mock_cost_guard.return_value = mock_guard

            # Only comparison model should be called (primary response is reused)
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "The capital city of France is Paris."
            mock_response.model = "claude-3"  # Add model attribute
            mock_response.cost_estimate = Decimal("0.003")
            mock_client.complete = AsyncMock(return_value=mock_response)
            mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.003"))
            mock_client_factory.return_value = mock_client

            # Get the tool and execute it
            tool = await mcp.get_tool("second_opinion")

            # Test response reuse - provide primary_response to save API calls
            result = await tool.run(
                {
                    "prompt": "What is the capital of France?",
                    "primary_model": "gpt-4",
                    "primary_response": "Paris is the capital of France.",  # Reuse existing response
                    "comparison_models": ["claude-3"],
                    "context": "Testing response reuse feature",
                }
            )

            # Get the text content
            content = result[0].text if hasattr(result[0], "text") else str(result[0])

            # Verify result contains both responses
            assert "Paris is the capital of France" in content
            assert "The capital city of France is Paris" in content
            # Context appears in the summary, which is expected behavior

            # Verify cost optimization worked
            # Primary model client shouldn't be called for completion since response was reused
            # But estimation might still be called for the comparison model
            assert mock_client.complete.call_count == 1  # Only comparison model

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self):
        """Test that cost limits are properly enforced."""
        with patch(
            "src.second_opinion.mcp.tools.second_opinion.get_cost_guard"
        ) as mock_cost_guard:

            # Mock cost guard that rejects due to budget
            mock_guard = MagicMock()
            mock_guard.check_and_reserve_budget = AsyncMock(
                side_effect=Exception(
                    "Cost limit of $0.01 would be exceeded. Estimated cost: $0.05"
                )
            )
            mock_cost_guard.return_value = mock_guard

            # Get the tool and execute it
            tool = await mcp.get_tool("second_opinion")

            # Execute tool with very low cost limit
            result = await tool.run(
                {
                    "prompt": "What is the capital of France?",
                    "primary_model": "gpt-4",
                    "cost_limit": 0.01,  # Very low limit
                }
            )

            # Get the text content
            content = result[0].text if hasattr(result[0], "text") else str(result[0])

            # Should return budget error message
            assert "Budget Error" in content
            assert "exceeds per-request limit" in content
            assert "$0.01" in content

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        tool = await mcp.get_tool("second_opinion")

        # Test with empty prompt - should return error message, not raise exception
        result = await tool.run({"prompt": ""})
        content = result[0].text if hasattr(result[0], "text") else str(result[0])
        # Should handle gracefully, not crash

        # Test with invalid model name
        result = await tool.run(
            {
                "prompt": "Test prompt",
                "primary_model": "definitely-not-a-real-model-name-12345",
            }
        )

        # Should handle gracefully and return error message
        assert isinstance(result, list)
        # The specific error depends on implementation, but should not crash


@pytest.fixture(autouse=True)
def reset_mcp_sessions():
    """Reset MCP sessions between tests."""
    from src.second_opinion.mcp.server import _sessions

    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    with patch.dict(
        "os.environ",
        {
            "OPENROUTER_API_KEY": "sk-or-test-key-for-mcp-testing",
            "ENVIRONMENT": "development",
        },
    ):
        yield
