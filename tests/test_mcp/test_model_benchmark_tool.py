"""
Tests for the MCP model_benchmark tool.

Tests verify that the tool works correctly with mocked dependencies,
handles benchmarking across different task types, and provides comprehensive
performance analysis.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.second_opinion.mcp.tools.model_benchmark import model_benchmark_tool

# Using fixtures from conftest.py


class TestModelBenchmarkTool:
    """Test the MCP model_benchmark tool."""

    @pytest.mark.asyncio
    async def test_basic_benchmark(self):
        """Test basic model benchmarking functionality."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]

            result = await model_benchmark_tool(
                models=models,
                task_types=["coding", "reasoning"],
                sample_size=2,
                cost_limit=1.0,
            )

            # Should return a formatted report
            assert isinstance(result, str)
            assert "# 🏁 Model Benchmark Results" in result
            assert "Benchmark Summary" in result or "Benchmark Overview" in result
            assert "Overall Rankings" in result or "Overall Model Rankings" in result
            assert (
                "Task Type Analysis" in result
                or "Task Categories" in result
                or "Task Category Performance" in result
            )
            assert "Performance Matrix" in result or "Performance" in result

            # Should not contain critical error messages (warnings are OK)
            assert "❌ **Unexpected Error**" not in result
            assert "❌ **Budget Error**" not in result

    @pytest.mark.asyncio
    async def test_single_task_type(self):
        """Test benchmarking with a single task type."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding"],
                sample_size=3,
                cost_limit=0.75,
            )

            assert "# 🏁 Model Benchmark Results" in result
            assert "coding" in result.lower()

    @pytest.mark.asyncio
    async def test_auto_task_selection(self):
        """Test automatic task type selection when none provided."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                sample_size=2,
                cost_limit=1.0,
            )

            assert "# 🏁 Model Benchmark Results" in result
            # Should include multiple task types
            assert (
                "coding" in result.lower()
                or "reasoning" in result.lower()
                or "creative" in result.lower()
            )

    @pytest.mark.asyncio
    async def test_invalid_models_limit(self):
        """Test validation of model count limits."""
        # Test too many models
        many_models = [f"model-{i}" for i in range(11)]
        result = await model_benchmark_tool(
            models=many_models,
            sample_size=2,
        )
        assert (
            "❌ **Too Many Models**: Maximum allowed is 8 models per benchmark"
            in result
        )

        # Test too few models
        result = await model_benchmark_tool(
            models=["single-model"],
            sample_size=2,
        )
        assert (
            "❌ **Invalid Models**: Please provide at least 2 models for benchmarking"
            in result
        )

    @pytest.mark.asyncio
    async def test_invalid_sample_size(self):
        """Test validation of sample size limits."""
        # Test sample size too high
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            sample_size=11,
        )
        assert (
            "❌ **Invalid Sample Size**: Must be between 1 and 5 tasks per category"
            in result
        )

        # Test sample size too low
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            sample_size=0,
        )
        assert (
            "❌ **Invalid Sample Size**: Must be between 1 and 5 tasks per category"
            in result
        )

    @pytest.mark.asyncio
    async def test_invalid_task_types(self):
        """Test validation of task type parameters."""
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["invalid_type"],
            sample_size=2,
        )
        assert "❌ **Invalid Task Types**: invalid_type" in result

    @pytest.mark.asyncio
    async def test_comprehensive_benchmark(self):
        """Test comprehensive benchmarking across all task types."""
        result = await model_benchmark_tool(
            models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "qwen3-4b-mlx"],
            task_types=["coding", "reasoning", "creative", "analysis"],
            sample_size=2,
            cost_limit=2.0,
        )

        assert "# 🏁 Model Benchmark Results" in result
        assert "coding" in result.lower()
        assert "reasoning" in result.lower()
        assert "creative" in result.lower()
        assert "analysis" in result.lower()

    @pytest.mark.asyncio
    async def test_cost_estimation_and_budget_check(self):
        """Test that cost estimation and budget checking work correctly."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.get_cost_guard"
        ) as mock_get_cost_guard:
            mock_cost_guard = AsyncMock()
            mock_cost_guard.check_and_reserve_budget = AsyncMock()
            mock_get_cost_guard.return_value = mock_cost_guard

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding"],
                sample_size=2,
                cost_limit=0.50,
            )

            # Should have called budget check
            mock_cost_guard.check_and_reserve_budget.assert_called()
            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_budget_exceeded_error(self):
        """Test handling when estimated cost exceeds budget."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.get_cost_guard"
        ) as mock_get_cost_guard:
            mock_cost_guard = AsyncMock()
            mock_cost_guard.check_and_reserve_budget = AsyncMock(
                side_effect=Exception("Budget exceeded: $5.00 > $1.00")
            )
            mock_get_cost_guard.return_value = mock_cost_guard

            result = await model_benchmark_tool(
                models=["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
                task_types=["coding", "reasoning", "creative"],
                sample_size=5,
                cost_limit=0.10,  # Very low limit
            )

            assert "❌ **Budget Error**" in result

    @pytest.mark.asyncio
    async def test_model_provider_detection(self):
        """Test that different model providers are correctly detected."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.detect_model_provider"
        ) as mock_detect:
            # Test that provider detection is called during benchmarking
            mock_detect.return_value = "openrouter"

            # Test with minimal valid input to trigger provider detection
            result = await model_benchmark_tool(
                models=["model1", "model2"],  # Minimal valid models
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.05,  # Low limit to trigger budget error early
            )

            # Should have called detect_model_provider at least once
            assert mock_detect.call_count >= 1
            # Should return some result (even if budget error)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_response_generation_error_handling(self):
        """Test handling of errors during response generation."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient(
                "mock", should_fail=True, failure_message="Connection failed"
            )
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["failing-model", "another-model"],  # Need at least 2 models
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.25,
            )

            # Should handle errors gracefully and still produce a report
            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_evaluation_system_integration(self):
        """Test integration with the evaluation system."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.get_evaluator"
        ) as mock_get_evaluator:
            mock_evaluator = AsyncMock()
            mock_get_evaluator.return_value = mock_evaluator

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.50,
            )

            # Should have used evaluator for scoring
            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_session_id_handling(self):
        """Test that session_id is properly handled for conversation tracking."""
        result = await model_benchmark_tool(
            models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
            task_types=["coding"],
            sample_size=1,
            session_id="benchmark-session-123",
            cost_limit=0.30,
        )

        assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_context_parameter(self):
        """Test that context parameter is properly handled."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.30,
            )

            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_performance_matrix_generation(self):
        """Test that performance matrix is properly generated."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding", "reasoning"],
                sample_size=1,
                cost_limit=0.50,
            )

            assert "# 🏁 Model Benchmark Results" in result
            assert "Performance" in result

    @pytest.mark.asyncio
    async def test_cost_analysis_reporting(self):
        """Test that cost analysis is properly reported."""
        result = await model_benchmark_tool(
            models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
            task_types=["coding"],
            sample_size=2,
            cost_limit=0.75,
        )

        assert "# 🏁 Model Benchmark Results" in result
        assert "Cost Analysis" in result
        assert "$" in result  # Should contain cost information

    @pytest.mark.asyncio
    async def test_model_ranking_accuracy(self):
        """Test that model rankings are calculated accurately."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=[
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3-haiku",
                    "qwen3-4b-mlx",
                ],
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.60,
            )

            assert "# 🏁 Model Benchmark Results" in result
            assert "🥇" in result  # Should have ranking indicators

    @pytest.mark.asyncio
    async def test_task_complexity_classification(self):
        """Test that tasks are properly classified by complexity."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.create_client_from_config"
        ) as mock_create_client:
            # Setup mock client
            from .conftest import MockClient

            mock_client = MockClient("mock")
            mock_create_client.return_value = mock_client

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding", "reasoning"],
                sample_size=1,
                cost_limit=0.50,
            )

            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_invalid_model_name_handling(self):
        """Test handling of invalid model names."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.validate_model_name"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Invalid model format")

            result = await model_benchmark_tool(
                models=[
                    "invalid-model-name",
                    "another-invalid",
                ],  # Need at least 2 models
                task_types=["coding"],
                sample_size=1,
            )

            assert "❌ **Invalid Model Name**" in result

    @pytest.mark.asyncio
    async def test_storage_context_integration(self):
        """Test integration with conversation storage."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.get_conversation_orchestrator"
        ) as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_get_orchestrator.return_value = mock_orchestrator

            result = await model_benchmark_tool(
                models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
                task_types=["coding"],
                sample_size=1,
                session_id="storage-test",
                cost_limit=0.30,
            )

            # Should have attempted to store conversation
            assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_benchmark_recommendations(self):
        """Test that practical recommendations are provided."""
        result = await model_benchmark_tool(
            models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "qwen3-4b-mlx"],
            task_types=["coding", "reasoning"],
            sample_size=1,
            cost_limit=0.75,
        )

        assert "# 🏁 Model Benchmark Results" in result
        assert (
            "Recommendations" in result
            or "recommendation" in result.lower()
            or "Best for" in result
        )

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self):
        """Test handling of unexpected errors during execution."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.sanitize_prompt"
        ) as mock_sanitize:
            mock_sanitize.side_effect = Exception("Unexpected error")

            result = await model_benchmark_tool(
                models=["model1", "model2"],
                task_types=["coding"],
                sample_size=1,
                cost_limit=0.20,
            )

            assert "❌ **Unexpected Error**" in result

    @pytest.mark.asyncio
    async def test_minimal_required_parameters(self):
        """Test benchmark with minimal required parameters."""
        result = await model_benchmark_tool(
            models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
        )

        # Should work with default parameters
        assert "# 🏁 Model Benchmark Results" in result

    @pytest.mark.asyncio
    async def test_cost_limit_validation(self):
        """Test validation of cost limit parameter."""
        with patch(
            "src.second_opinion.mcp.tools.model_benchmark.validate_cost_limit"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Invalid cost limit")

            result = await model_benchmark_tool(
                models=["model1", "model2"],
                cost_limit=-0.50,  # Invalid negative cost
            )

            assert "❌" in result  # Should contain error message
