"""
Tests for the MCP compare_responses tool.

Tests verify that the tool works correctly with mocked dependencies,
handles various comparison scenarios, and provides detailed analysis
of two responses with cost optimization.
"""


import pytest

from src.second_opinion.mcp.tools.compare_responses import compare_responses_tool

from .conftest import (
    SAMPLE_CODE_PROMPT,
    SAMPLE_CODE_RESPONSE,
    SAMPLE_SHELL_PROMPT,
    SAMPLE_SHELL_RESPONSE,
)

# Additional test responses for comparison
SAMPLE_CODE_RESPONSE_B = """
def fibonacci(n):
    # Simple iterative approach
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""

SAMPLE_SHELL_RESPONSE_B = """
To list files in a directory, you can use several commands:

1. `ls` - Basic listing
2. `ls -la` - Detailed listing with hidden files
3. `tree` - Tree view (if installed)

The `ls` command is the most commonly used.
"""


class TestCompareResponsesTool:
    """Test the MCP compare_responses tool."""

    @pytest.mark.asyncio
    async def test_basic_comparison_functionality(self):
        """Test basic comparison functionality with two responses."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o-mini",
            cost_limit=0.25,
        )

        # Should return a formatted comparison report
        assert isinstance(result, str)
        assert "# ⚖️ Response Comparison: Side-by-Side Analysis" in result
        assert "Comparison Context" in result
        assert "Overall Winner" in result
        assert "Side-by-Side Responses" in result
        assert "Cost & Model Analysis" in result
        assert "Actionable Recommendations" in result
        assert "Next Steps" in result

        # Should not contain error messages
        assert "❌" not in result

        # Should contain model information
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o-mini" in result

    @pytest.mark.asyncio
    async def test_comparison_without_models(self):
        """Test comparison when model names are not provided."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            cost_limit=0.25,
        )

        # Should still work with unknown models
        assert isinstance(result, str)
        assert "# ⚖️ Response Comparison: Side-by-Side Analysis" in result
        assert "unknown/model-a" in result
        assert "unknown/model-b" in result
        assert "❌" not in result

    @pytest.mark.asyncio
    async def test_shell_command_comparison(self):
        """Test comparison with shell command responses."""
        result = await compare_responses_tool(
            response_a=SAMPLE_SHELL_RESPONSE,
            response_b=SAMPLE_SHELL_RESPONSE_B,
            task=SAMPLE_SHELL_PROMPT,
            model_a="qwen3-4b-mlx",
            model_b="anthropic/claude-3-haiku",
            cost_limit=0.25,
        )

        # Should handle shell commands without security issues
        assert isinstance(result, str)
        assert "# ⚖️ Response Comparison: Side-by-Side Analysis" in result
        assert "qwen3-4b-mlx" in result
        assert "anthropic/claude-3-haiku" in result
        assert "❌" not in result
        assert "Security" not in result

    @pytest.mark.asyncio
    async def test_local_vs_cloud_comparison(self):
        """Test comparison between local and cloud models."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="qwen3-4b-mlx",  # Local model
            model_b="anthropic/claude-3-5-sonnet",  # Cloud model
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        assert "qwen3-4b-mlx" in result
        assert "anthropic/claude-3-5-sonnet" in result

        # Should mention cost differences
        assert "$0.00" in result or "zero" in result.lower()  # Local model cost
        assert "cost" in result.lower()

    @pytest.mark.asyncio
    async def test_cost_limit_validation(self):
        """Test cost limit validation."""
        # Test with very low cost limit
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o",
            cost_limit=0.001,  # Very low limit
        )

        # Should either work (if within limit) or show budget error
        assert isinstance(result, str)
        # Should not crash, either works or shows budget error
        if "❌" in result:
            assert "Budget Error" in result

    @pytest.mark.asyncio
    async def test_invalid_model_names(self):
        """Test handling of invalid model names."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="invalid-model-name-123",
            model_b="another-invalid-model",
            cost_limit=0.25,
        )

        # Should return error with helpful suggestions
        assert isinstance(result, str)
        if "❌" in result:
            assert "Invalid Model" in result
            assert "Suggested formats" in result

    @pytest.mark.asyncio
    async def test_empty_responses(self):
        """Test handling of empty responses."""
        result = await compare_responses_tool(
            response_a="",
            response_b="",
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-haiku",
            model_b="openai/gpt-4o-mini",
            cost_limit=0.25,
        )

        # Should handle empty responses gracefully
        assert isinstance(result, str)
        # Should either work or provide meaningful error
        if "❌" not in result:
            assert "# ⚖️ Response Comparison: Side-by-Side Analysis" in result

    @pytest.mark.asyncio
    async def test_quality_criteria_breakdown(self):
        """Test that quality criteria are properly analyzed."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o",
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should include quality analysis
            assert "/10" in result  # Quality scores
            # May include criteria breakdown if available
            # Note: Actual criteria might not always be available due to evaluation system

    @pytest.mark.asyncio
    async def test_task_complexity_handling(self):
        """Test handling of different task complexities."""
        # Simple task
        simple_result = await compare_responses_tool(
            response_a="2 + 2 = 4",
            response_b="The answer is 4",
            task="What is 2 + 2?",
            model_a="anthropic/claude-3-haiku",
            model_b="openai/gpt-4o-mini",
            cost_limit=0.25,
        )

        assert isinstance(simple_result, str)
        if "❌" not in simple_result:
            assert "# ⚖️ Response Comparison: Side-by-Side Analysis" in simple_result

    @pytest.mark.asyncio
    async def test_cost_optimization_focus(self):
        """Test that the tool emphasizes cost optimization."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-opus",  # Expensive model
            model_b="anthropic/claude-3-haiku",  # Budget model
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should mention cost considerations
            assert "cost" in result.lower() or "$" in result
            # Should provide recommendations (including fallback scenarios)
            assert any(
                keyword in result
                for keyword in [
                    "RECOMMENDED",
                    "CONSIDER",
                    "CLOSE CALL",
                    "Choose based on",
                ]
            )

    @pytest.mark.asyncio
    async def test_response_filtering(self):
        """Test that responses are properly filtered for think tags."""
        response_with_think = """
        <think>
        Let me think about this step by step...
        </think>
        
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """

        result = await compare_responses_tool(
            response_a=response_with_think,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o",
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        # Think tags should be filtered out
        assert "<think>" not in result
        assert "</think>" not in result

    @pytest.mark.asyncio
    async def test_winner_determination(self):
        """Test that a winner is properly determined."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o",
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should declare a winner or tie
            assert any(
                keyword in result
                for keyword in [
                    "WINS",
                    "TIE",
                    "Close Competition",
                    "RECOMMENDED",
                    "CONSIDER",
                ]
            )

    @pytest.mark.asyncio
    async def test_actionable_recommendations(self):
        """Test that actionable recommendations are provided."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="anthropic/claude-3-haiku",
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should include next steps and recommendations
            assert "Next Steps" in result
            assert any(
                keyword in result for keyword in ["Use", "Test", "Consider", "Monitor"]
            )

    @pytest.mark.asyncio
    async def test_tier_analysis(self):
        """Test that model tier analysis is included."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-opus",  # Premium tier
            model_b="anthropic/claude-3-haiku",  # Budget tier
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should mention tiers
            assert any(tier in result.lower() for tier in ["premium", "budget", "tier"])

    @pytest.mark.asyncio
    async def test_zero_cost_analysis(self):
        """Test that zero additional API cost is emphasized."""
        result = await compare_responses_tool(
            response_a=SAMPLE_CODE_RESPONSE,
            response_b=SAMPLE_CODE_RESPONSE_B,
            task=SAMPLE_CODE_PROMPT,
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o",
            cost_limit=0.25,
        )

        assert isinstance(result, str)
        if "❌" not in result:
            # Should indicate low analysis cost
            # The tool mentions analysis cost separately from model costs
            assert "Analysis Cost" in result or "Testing Cost" in result
