"""
Tests for the model_benchmark MCP tool.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from second_opinion.mcp.tools.model_benchmark import model_benchmark_tool


class TestModelBenchmarkTool:
    """Test the model_benchmark MCP tool functionality."""

    @pytest.fixture
    def mock_model_responses(self):
        """Create mock model responses for testing."""
        from second_opinion.core.models import ModelResponse, TokenUsage

        return [
            ModelResponse(
                content="def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
                model="anthropic/claude-3-5-sonnet",
                usage=TokenUsage(input_tokens=30, output_tokens=50, total_tokens=80),
                cost_estimate=Decimal("0.08"),
                provider="openrouter",
            ),
            ModelResponse(
                content="def factorial(n):\\n    return 1 if n <= 1 else n * factorial(n-1)",
                model="openai/gpt-4o-mini",
                usage=TokenUsage(input_tokens=30, output_tokens=35, total_tokens=65),
                cost_estimate=Decimal("0.03"),
                provider="openrouter",
            ),
            ModelResponse(
                content="def factorial(n):\\n    result = 1\\n    for i in range(1, n+1):\\n        result *= i\\n    return result",
                model="qwen3-4b-mlx",
                usage=TokenUsage(input_tokens=30, output_tokens=45, total_tokens=75),
                cost_estimate=Decimal("0.00"),
                provider="lmstudio",
            ),
        ]

    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results for testing."""
        return [
            {
                "model_a": "anthropic/claude-3-5-sonnet",
                "model_b": "openai/gpt-4o-mini",
                "winner": "primary",  # claude wins
                "score_a": 8.5,
                "score_b": 7.0,
                "overall_score": 8.5,
                "reasoning": "Claude's recursive solution is more elegant",
            },
            {
                "model_a": "anthropic/claude-3-5-sonnet",
                "model_b": "qwen3-4b-mlx",
                "winner": "comparison",  # qwen wins
                "score_a": 8.0,
                "score_b": 8.5,
                "overall_score": 8.5,
                "reasoning": "Qwen's iterative approach is more efficient",
            },
            {
                "model_a": "openai/gpt-4o-mini",
                "model_b": "qwen3-4b-mlx",
                "winner": "comparison",  # qwen wins
                "score_a": 7.0,
                "score_b": 8.5,
                "overall_score": 8.5,
                "reasoning": "Qwen provides better performance and clarity",
            },
        ]

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_basic_benchmark(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        mock_model_responses,
        mock_evaluation_results,
    ):
        """Test basic benchmarking functionality."""
        # Setup mocks
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_evaluator_instance = AsyncMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.classify_task_complexity.return_value = MagicMock(
            value="moderate"
        )

        # Mock client responses
        mock_clients = []
        for response in mock_model_responses:
            client = AsyncMock()
            client.estimate_cost.return_value = response.cost_estimate
            client.complete.return_value = response
            mock_clients.append(client)

        mock_create_client.side_effect = mock_clients * 3  # 3 tasks per client

        # Mock evaluation results
        comparison_results = []
        for result in mock_evaluation_results:
            mock_result = MagicMock()
            mock_result.winner = result["winner"]
            mock_result.score_a = result["score_a"]
            mock_result.score_b = result["score_b"]
            mock_result.overall_score = result["overall_score"]
            mock_result.reasoning = result["reasoning"]
            comparison_results.append(mock_result)

        mock_evaluator_instance.compare_responses.side_effect = (
            comparison_results * 3
        )  # Multiple tasks

        # Execute benchmark
        result = await model_benchmark_tool(
            models=[
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-4o-mini",
                "qwen3-4b-mlx",
            ],
            task_types=["coding"],
            sample_size=1,  # Reduce for test speed
            evaluation_criteria="comprehensive",
            cost_limit=2.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏁 Model Benchmark Results" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o-mini" in result
        assert "qwen3-4b-mlx" in result
        assert "Overall Model Rankings" in result

        # Verify cost tracking
        mock_cost_guard_instance.check_and_reserve_budget.assert_called_once()
        mock_cost_guard_instance.record_actual_cost.assert_called_once()

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_multiple_task_types(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        mock_model_responses,
        mock_evaluation_results,
    ):
        """Test benchmarking across multiple task types."""
        # Setup mocks
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_evaluator_instance = AsyncMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.classify_task_complexity.return_value = MagicMock(
            value="moderate"
        )

        # Mock client responses - need more responses for multiple task types
        mock_clients = []
        for _ in range(6):  # 3 models * 2 task types
            for response in mock_model_responses:
                client = AsyncMock()
                client.estimate_cost.return_value = response.cost_estimate
                client.complete.return_value = response
                mock_clients.append(client)

        mock_create_client.side_effect = mock_clients

        # Mock evaluation results
        comparison_results = []
        for result in mock_evaluation_results:
            mock_result = MagicMock()
            mock_result.winner = result["winner"]
            mock_result.score_a = result["score_a"]
            mock_result.score_b = result["score_b"]
            mock_result.overall_score = result["overall_score"]
            mock_result.reasoning = result["reasoning"]
            comparison_results.append(mock_result)

        mock_evaluator_instance.compare_responses.side_effect = (
            comparison_results * 10
        )  # Multiple comparisons

        # Execute with multiple task types
        result = await model_benchmark_tool(
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o-mini"],
            task_types=["coding", "reasoning"],
            sample_size=1,
            evaluation_criteria="comprehensive",
            cost_limit=2.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏁 Model Benchmark Results" in result
        assert "Task Categories**: coding, reasoning" in result
        assert "Task Category Performance" in result
        assert "Coding Tasks" in result
        assert "Reasoning Tasks" in result

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    async def test_accuracy_evaluation_criteria(
        self,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
    ):
        """Test accuracy-focused evaluation criteria."""
        # Setup mocks
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_evaluator_instance = AsyncMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.classify_task_complexity.return_value = MagicMock(
            value="moderate"
        )

        # Mock client with minimal responses
        with patch(
            "second_opinion.utils.client_factory.create_client_from_config"
        ) as mock_create_client:
            mock_client = AsyncMock()
            mock_client.estimate_cost.return_value = Decimal("0.01")
            mock_client.complete.return_value = MagicMock(
                content="Simple response",
                model="test-model",
                cost_estimate=Decimal("0.01"),
            )
            mock_create_client.return_value = mock_client

            # Execute with accuracy criteria
            result = await model_benchmark_tool(
                models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o-mini"],
                task_types=["coding"],
                sample_size=1,
                evaluation_criteria="accuracy",
                cost_limit=1.0,
            )

            # Verify
            assert isinstance(result, str)
            assert "🏁 Model Benchmark Results" in result
            assert "Evaluation Criteria**: Accuracy" in result

    async def test_invalid_models(self):
        """Test validation of invalid model inputs."""
        # Test too few models
        result = await model_benchmark_tool(
            models=["single-model"],
            task_types=["coding"],
        )
        assert "❌ **Invalid Models**" in result
        assert "at least 2 models" in result

        # Test too many models
        many_models = [f"model-{i}" for i in range(10)]
        result = await model_benchmark_tool(
            models=many_models,
            task_types=["coding"],
        )
        assert "❌ **Too Many Models**" in result
        assert "Maximum allowed is 8" in result

    async def test_invalid_sample_size(self):
        """Test validation of invalid sample size."""
        # Test sample size too small
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["coding"],
            sample_size=0,
        )
        assert "❌ **Invalid Sample Size**" in result

        # Test sample size too large
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["coding"],
            sample_size=6,
        )
        assert "❌ **Invalid Sample Size**" in result

    async def test_invalid_task_types(self):
        """Test validation of invalid task types."""
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["invalid_task_type"],
        )
        assert "❌ **Invalid Task Types**" in result
        assert "invalid_task_type" in result

    async def test_invalid_evaluation_criteria(self):
        """Test validation of invalid evaluation criteria."""
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["coding"],
            evaluation_criteria="invalid_criteria",
        )
        assert "❌ **Invalid Evaluation Criteria**" in result
        assert "invalid_criteria" in result

    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    async def test_budget_error(self, mock_cost_guard):
        """Test budget error handling."""
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance
        mock_cost_guard_instance.check_and_reserve_budget.side_effect = Exception(
            "Budget exceeded"
        )

        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=["coding"],
            cost_limit=0.01,  # Very low limit
        )

        assert "❌ **Budget Error**" in result
        assert "Budget exceeded" in result

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_model_generation_error(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
    ):
        """Test handling of model generation errors."""
        # Setup mocks
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_evaluator_instance = AsyncMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.classify_task_complexity.return_value = MagicMock(
            value="moderate"
        )

        # Mock one successful client and one failing client
        mock_client_success = AsyncMock()
        mock_client_success.estimate_cost.return_value = Decimal("0.01")
        mock_client_success.complete.return_value = MagicMock(
            content="Good response",
            model="good-model",
            cost_estimate=Decimal("0.01"),
        )

        mock_client_fail = AsyncMock()
        mock_client_fail.estimate_cost.return_value = Decimal("0.01")
        mock_client_fail.complete.side_effect = Exception("Model unavailable")

        mock_create_client.side_effect = [mock_client_success, mock_client_fail]

        # Execute
        result = await model_benchmark_tool(
            models=["good-model", "bad-model"],
            task_types=["coding"],
            sample_size=1,
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏁 Model Benchmark Results" in result
        # Should continue with partial results
        assert "good-model" in result

    @pytest.mark.parametrize(
        "task_types,criteria",
        [
            (["coding"], "accuracy"),
            (["reasoning"], "comprehensive"),
            (["creative"], "creativity"),
            (["analysis"], "speed"),
            (["explanation"], "comprehensive"),
        ],
    )
    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_task_type_criteria_combinations(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        task_types,
        criteria,
    ):
        """Test different combinations of task types and evaluation criteria."""
        # Setup mocks
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_evaluator_instance = AsyncMock()
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.classify_task_complexity.return_value = MagicMock(
            value="moderate"
        )

        # Mock client
        mock_client = AsyncMock()
        mock_client.estimate_cost.return_value = Decimal("0.01")
        mock_client.complete.return_value = MagicMock(
            content="Test response",
            model="test-model",
            cost_estimate=Decimal("0.01"),
        )
        mock_create_client.return_value = mock_client

        # Execute
        result = await model_benchmark_tool(
            models=["model1", "model2"],
            task_types=task_types,
            sample_size=1,
            evaluation_criteria=criteria,
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏁 Model Benchmark Results" in result
        assert f"**Evaluation Criteria**: {criteria.title()}" in result
        assert task_types[0].title() in result

    def test_calculate_benchmark_scores(self):
        """Test the benchmark scoring calculation logic."""
        from second_opinion.core.models import ModelResponse, TokenUsage
        from second_opinion.mcp.tools.model_benchmark import _calculate_benchmark_scores

        # Mock benchmark results
        models = ["model_a", "model_b"]
        benchmark_results = [
            {
                "task_type": "coding",
                "model_responses": [
                    ModelResponse(
                        content="Response A",
                        model="model_a",
                        usage=TokenUsage(
                            input_tokens=10, output_tokens=20, total_tokens=30
                        ),
                        cost_estimate=Decimal("0.05"),
                        provider="openrouter",
                    ),
                    ModelResponse(
                        content="Response B",
                        model="model_b",
                        usage=TokenUsage(
                            input_tokens=10, output_tokens=20, total_tokens=30
                        ),
                        cost_estimate=Decimal("0.02"),
                        provider="lmstudio",
                    ),
                ],
                "evaluation_results": [
                    {
                        "model_a": "model_a",
                        "model_b": "model_b",
                        "winner": "primary",  # model_a wins
                        "score_a": 8.0,
                        "score_b": 6.0,
                        "overall_score": 8.0,
                        "reasoning": "A is better",
                    }
                ],
            }
        ]

        # Test comprehensive scoring
        scores = _calculate_benchmark_scores(models, benchmark_results, "comprehensive")

        assert len(scores) == 2
        assert all("rank" in score for score in scores)
        assert all("overall_score" in score for score in scores)
        assert scores[0]["rank"] == 1  # Best model is rank 1

        # Verify sorting (highest score first)
        assert scores[0]["overall_score"] >= scores[1]["overall_score"]

        # Test different evaluation criteria
        accuracy_scores = _calculate_benchmark_scores(
            models, benchmark_results, "accuracy"
        )
        speed_scores = _calculate_benchmark_scores(models, benchmark_results, "speed")
        creativity_scores = _calculate_benchmark_scores(
            models, benchmark_results, "creativity"
        )

        # All should return valid scores
        for score_set in [accuracy_scores, speed_scores, creativity_scores]:
            assert len(score_set) == 2
            assert all(0 <= score["overall_score"] <= 100 for score in score_set)

    def test_get_benchmark_strength(self):
        """Test benchmark strength description generation."""
        from second_opinion.mcp.tools.model_benchmark import _get_benchmark_strength

        score = {
            "overall_win_rate": 85.0,
            "total_cost": Decimal("0.50"),
        }

        # Test different criteria
        assert "accuracy" in _get_benchmark_strength(score, "accuracy").lower()
        assert "creative" in _get_benchmark_strength(score, "creativity").lower()
        assert "speed" in _get_benchmark_strength(score, "speed").lower()
        assert "balance" in _get_benchmark_strength(score, "comprehensive").lower()

    def test_get_model_name_suggestions(self):
        """Test model name suggestion generation."""
        from second_opinion.mcp.tools.model_benchmark import _get_model_name_suggestions

        suggestions = _get_model_name_suggestions("invalid-model")

        assert "Cloud Models" in suggestions
        assert "Local Models" in suggestions
        assert "anthropic/claude-3-5-sonnet" in suggestions
        assert "qwen3-4b-mlx" in suggestions
        assert "Example model list" in suggestions

    def test_benchmark_tasks_structure(self):
        """Test that benchmark tasks are properly structured."""
        from second_opinion.mcp.tools.model_benchmark import BENCHMARK_TASKS

        # Verify all expected task types exist
        expected_types = ["coding", "reasoning", "creative", "analysis", "explanation"]
        for task_type in expected_types:
            assert task_type in BENCHMARK_TASKS
            assert isinstance(BENCHMARK_TASKS[task_type], list)
            assert len(BENCHMARK_TASKS[task_type]) >= 3  # At least 3 tasks per type

        # Verify tasks are not empty
        for _task_type, tasks in BENCHMARK_TASKS.items():
            for task in tasks:
                assert isinstance(task, str)
                assert len(task.strip()) > 10  # Reasonable task length
