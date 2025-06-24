"""
Tests for the MCP batch_comparison tool.

Tests verify that the tool works correctly with mocked dependencies,
handles multiple model comparisons, and provides comprehensive rankings.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.second_opinion.mcp.tools.batch_comparison import batch_comparison_tool

from .conftest import (
    SAMPLE_CODE_PROMPT,
)


class TestBatchComparisonTool:
    """Test the MCP batch_comparison tool."""

    @pytest.mark.asyncio
    async def test_basic_batch_comparison(self):
        """Test basic batch comparison with existing responses."""
        responses = [
            "def add(a, b): return a + b",
            "def add(a, b):\n    return a + b",
            "add = lambda a, b: a + b",
        ]
        models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "openai/gpt-4o"]

        result = await batch_comparison_tool(
            task="Write a simple addition function",
            responses=responses,
            models=models,
            rank_by="quality",
            cost_limit=0.25,
        )

        # Should return a formatted report
        assert isinstance(result, str)
        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Task Summary" in result
        assert "Cost Analysis" in result
        assert "Model Rankings" in result
        assert "Complete Responses" in result
        assert "Performance Insights" in result
        assert "Recommendations" in result

        # Should not contain error messages
        assert "❌" not in result

    @pytest.mark.asyncio
    async def test_generate_fresh_responses(self):
        """Test generating fresh responses from multiple models."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]

            result = await batch_comparison_tool(
                task=SAMPLE_CODE_PROMPT,
                models=models,
                rank_by="comprehensive",
                cost_limit=0.50,
            )

            # Should return a formatted report
            assert isinstance(result, str)
            assert "# 🏆 Batch Model Comparison Results" in result
            assert "Generate Fresh" in result or "Auto Generate" in result

    @pytest.mark.asyncio
    async def test_auto_model_selection(self):
        """Test automatic model selection when no models provided."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await batch_comparison_tool(
                task="Explain machine learning basics",
                rank_by="quality",
                max_models=3,
                cost_limit=0.30,
            )

            # Should return a formatted report
            assert isinstance(result, str)
            assert "# 🏆 Batch Model Comparison Results" in result

    @pytest.mark.asyncio
    async def test_invalid_max_models(self):
        """Test validation of max_models parameter."""
        # Test too many models
        result = await batch_comparison_tool(
            task="Test task",
            max_models=15,
        )
        assert "❌ **Invalid Max Models**: Maximum allowed is 10" in result

        # Test too few models
        result = await batch_comparison_tool(
            task="Test task",
            max_models=1,
        )
        assert "❌ **Invalid Max Models**: Minimum required is 2" in result

    @pytest.mark.asyncio
    async def test_invalid_ranking_criteria(self):
        """Test validation of ranking criteria."""
        result = await batch_comparison_tool(
            task="Test task",
            rank_by="invalid_criteria",
        )
        assert "❌ **Invalid Ranking Criteria**: 'invalid_criteria'" in result

    @pytest.mark.asyncio
    async def test_mismatched_responses_and_models(self):
        """Test validation when responses and models don't match."""
        result = await batch_comparison_tool(
            task="Test task",
            responses=["response1", "response2"],
            models=["model1"],
        )
        assert "❌ **Mismatched Input**" in result

    @pytest.mark.asyncio
    async def test_too_many_responses(self):
        """Test validation when too many responses provided."""
        responses = ["resp1", "resp2", "resp3", "resp4", "resp5", "resp6"]
        result = await batch_comparison_tool(
            task="Test task",
            responses=responses,
            max_models=3,
        )
        assert "❌ **Too Many Responses**" in result

    @pytest.mark.asyncio
    async def test_cost_ranking(self):
        """Test ranking by cost criteria."""
        responses = [
            "Simple response",
            "More detailed response with explanations",
        ]
        models = ["openai/gpt-4o-mini", "openai/gpt-4o"]

        result = await batch_comparison_tool(
            task="Simple question",
            responses=responses,
            models=models,
            rank_by="cost",
            cost_limit=0.20,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Cost" in result

    @pytest.mark.asyncio
    async def test_speed_ranking(self):
        """Test ranking by speed criteria."""
        responses = [
            "Response from local model",
            "Response from cloud model",
        ]
        models = ["qwen3-4b-mlx", "anthropic/claude-3-5-sonnet"]

        result = await batch_comparison_tool(
            task="Test question",
            responses=responses,
            models=models,
            rank_by="speed",
            cost_limit=0.15,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Speed" in result

    @pytest.mark.asyncio
    async def test_comprehensive_ranking(self):
        """Test comprehensive ranking combining quality, cost, and speed."""
        responses = [
            "Good quality response",
            "Excellent detailed response",
            "Fast simple response",
        ]
        models = ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet", "qwen3-4b-mlx"]

        result = await batch_comparison_tool(
            task="Complex analysis question",
            responses=responses,
            models=models,
            rank_by="comprehensive",
            cost_limit=0.40,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Comprehensive" in result

    @pytest.mark.asyncio
    async def test_anonymous_responses(self):
        """Test evaluating responses without model information."""
        responses = [
            "First anonymous response",
            "Second anonymous response",
        ]

        result = await batch_comparison_tool(
            task="Compare these responses",
            responses=responses,
            rank_by="quality",
            cost_limit=0.20,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Model_1" in result or "Model_2" in result

    @pytest.mark.asyncio
    async def test_context_handling(self):
        """Test that context is properly handled and sanitized."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await batch_comparison_tool(
                task="Technical question",
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                context="This is for technical documentation",
                rank_by="quality",
                cost_limit=0.25,
            )

            assert "# 🏆 Batch Model Comparison Results" in result
            assert "Context" in result

    @pytest.mark.asyncio
    async def test_session_id_handling(self):
        """Test that session_id is properly handled for conversation tracking."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await batch_comparison_tool(
                task="Test with session",
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                session_id="test-session-123",
                cost_limit=0.20,
            )

            assert "# 🏆 Batch Model Comparison Results" in result

    @pytest.mark.asyncio
    async def test_error_response_handling(self):
        """Test handling of error responses during generation."""
        responses = [
            "Good response",
            "Error: Failed to generate response from model: Connection failed",
        ]
        models = ["openai/gpt-4o-mini", "failing-model"]

        result = await batch_comparison_tool(
            task="Test error handling",
            responses=responses,
            models=models,
            rank_by="quality",
            cost_limit=0.20,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "❌" in result  # Error responses should be marked

    @pytest.mark.asyncio
    async def test_local_vs_cloud_analysis(self):
        """Test analysis comparing local and cloud models."""
        responses = [
            "Local model response",
            "Cloud model response",
        ]
        models = ["qwen3-4b-mlx", "anthropic/claude-3-5-sonnet"]

        result = await batch_comparison_tool(
            task="Compare local vs cloud",
            responses=responses,
            models=models,
            rank_by="comprehensive",
            cost_limit=0.25,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Local vs Cloud" in result or "local" in result.lower()

    @pytest.mark.asyncio
    async def test_cost_efficiency_recommendations(self):
        """Test that cost efficiency recommendations are provided."""
        responses = [
            "Expensive but high quality",
            "Cheap but good enough",
        ]
        models = ["openai/gpt-4o", "openai/gpt-4o-mini"]

        result = await batch_comparison_tool(
            task="Cost efficiency test",
            responses=responses,
            models=models,
            rank_by="cost",
            cost_limit=0.30,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "Cost-Efficient" in result or "cost" in result.lower()

    @pytest.mark.asyncio
    async def test_medal_ranking_display(self):
        """Test that top models get medal emojis in rankings."""
        responses = [
            "First place response",
            "Second place response",
            "Third place response",
        ]
        models = ["model1", "model2", "model3"]

        result = await batch_comparison_tool(
            task="Medal test",
            responses=responses,
            models=models,
            rank_by="quality",
            cost_limit=0.25,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "🥇" in result  # Gold medal for first place
        assert "🥈" in result  # Silver medal for second place
        assert "🥉" in result  # Bronze medal for third place

    @pytest.mark.asyncio
    async def test_response_truncation(self):
        """Test that very long responses are properly truncated."""
        long_response = "Very long response. " * 200  # Create very long response
        responses = [
            "Short response",
            long_response,
        ]
        models = ["model1", "model2"]

        result = await batch_comparison_tool(
            task="Truncation test",
            responses=responses,
            models=models,
            rank_by="quality",
            cost_limit=0.20,
        )

        assert "# 🏆 Batch Model Comparison Results" in result
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_model_name_suggestions(self):
        """Test that helpful suggestions are provided for invalid model names."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.validate_model_name"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Invalid model format")

            result = await batch_comparison_tool(
                task="Invalid model test",
                models=["invalid-model-name"],
                cost_limit=0.15,
            )

            assert "❌ **Invalid Model Name**" in result
            assert "**Suggested formats:**" in result
            assert "anthropic/claude" in result
            assert "openai/gpt" in result

    @pytest.mark.asyncio
    async def test_budget_exceeded_error(self):
        """Test handling when budget is exceeded."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.get_cost_guard"
        ) as mock_get_cost_guard:
            mock_cost_guard = AsyncMock()
            mock_cost_guard.check_and_reserve_budget = AsyncMock(
                side_effect=Exception("Budget exceeded")
            )
            mock_get_cost_guard.return_value = mock_cost_guard

            result = await batch_comparison_tool(
                task="Budget test",
                models=["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
                cost_limit=0.01,  # Very low limit
            )

            assert "❌ **Budget Error**" in result

    @pytest.mark.asyncio
    async def test_evaluation_system_failure(self):
        """Test handling when evaluation system fails."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.get_evaluator"
        ) as mock_get_evaluator:
            from src.second_opinion.core.models import TaskComplexity

            mock_evaluator = AsyncMock()
            mock_evaluator.compare_responses = AsyncMock(
                side_effect=Exception("Evaluation failed")
            )
            # Properly mock classify_task_complexity to return a TaskComplexity enum
            mock_evaluator.classify_task_complexity = AsyncMock(
                return_value=TaskComplexity.MODERATE
            )
            mock_get_evaluator.return_value = mock_evaluator

            result = await batch_comparison_tool(
                task="Evaluation failure test",
                responses=["response1", "response2"],
                models=["model1", "model2"],
                cost_limit=0.20,
            )

            # Should still complete with a report, even if evaluation fails
            assert "# 🏆 Batch Model Comparison Results" in result

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self):
        """Test handling of unexpected errors during execution."""
        with patch(
            "src.second_opinion.mcp.tools.batch_comparison.sanitize_prompt"
        ) as mock_sanitize:
            mock_sanitize.side_effect = Exception("Unexpected error")

            result = await batch_comparison_tool(
                task="Error test",
                models=["model1", "model2"],
                cost_limit=0.15,
            )

            assert "❌ **Unexpected Error**" in result
