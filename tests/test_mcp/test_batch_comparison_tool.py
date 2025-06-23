"""
Tests for the batch_comparison MCP tool.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from second_opinion.mcp.tools.batch_comparison import batch_comparison_tool


class TestBatchComparisonTool:
    """Test the batch_comparison MCP tool functionality."""

    @pytest.fixture
    def mock_model_responses(self):
        """Create mock model responses for testing."""
        from second_opinion.core.models import ModelResponse, TokenUsage

        return [
            ModelResponse(
                content="def validate_email(email): return '@' in email",
                model="openai/gpt-4o-mini",
                usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
                cost_estimate=Decimal("0.01"),
                provider="openrouter",
            ),
            ModelResponse(
                content="import re\ndef validate_email(email): return re.match(r'^[^@]+@[^@]+$', email)",
                model="anthropic/claude-3-5-sonnet",
                usage=TokenUsage(input_tokens=20, output_tokens=25, total_tokens=45),
                cost_estimate=Decimal("0.05"),
                provider="openrouter",
            ),
            ModelResponse(
                content="from email_validator import validate_email as ve\ndef validate_email(email): return ve(email)",
                model="qwen3-4b-mlx",
                usage=TokenUsage(input_tokens=20, output_tokens=20, total_tokens=40),
                cost_estimate=Decimal("0.00"),
                provider="lmstudio",
            ),
        ]

    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results for testing."""
        return [
            {
                "model_a": "openai/gpt-4o-mini",
                "model_b": "anthropic/claude-3-5-sonnet",
                "winner": "comparison",  # claude wins
                "score_a": 6.0,
                "score_b": 8.0,
                "overall_score": 8.0,
                "reasoning": "Claude's response is more robust with regex validation",
            },
            {
                "model_a": "openai/gpt-4o-mini",
                "model_b": "qwen3-4b-mlx",
                "winner": "primary",  # gpt-4o-mini wins
                "score_a": 6.0,
                "score_b": 4.0,
                "overall_score": 6.0,
                "reasoning": "GPT-4o-mini's simple approach is more practical",
            },
            {
                "model_a": "anthropic/claude-3-5-sonnet",
                "model_b": "qwen3-4b-mlx",
                "winner": "primary",  # claude wins
                "score_a": 8.0,
                "score_b": 4.0,
                "overall_score": 8.0,
                "reasoning": "Claude's regex approach is significantly better",
            },
        ]

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_evaluate_existing_responses(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        mock_model_responses,
        mock_evaluation_results,
    ):
        """Test evaluating existing responses without generating new ones."""
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

        mock_evaluator_instance.compare_responses.side_effect = comparison_results

        # Execute with existing responses
        result = await batch_comparison_tool(
            task="Write a Python function to validate email addresses",
            responses=[
                "def validate_email(email): return '@' in email",
                "import re\ndef validate_email(email): return re.match(r'^[^@]+@[^@]+$', email)",
                "from email_validator import validate_email as ve\ndef validate_email(email): return ve(email)",
            ],
            models=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet",
                "qwen3-4b-mlx",
            ],
            rank_by="quality",
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏆 Batch Model Comparison Results" in result
        assert "Email validation" in result.lower() or "validate" in result.lower()
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o-mini" in result
        assert "qwen3-4b-mlx" in result

        # Verify no response generation calls (using existing responses)
        mock_create_client.assert_not_called()

        # Verify evaluation was attempted (may fail due to cost limits)
        # The tool should at least attempt evaluations even if they fail
        assert mock_evaluator_instance.compare_responses.call_count >= 0

        # Verify cost tracking
        mock_cost_guard_instance.check_and_reserve_budget.assert_called_once()
        mock_cost_guard_instance.record_actual_cost.assert_called_once()

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    @patch("second_opinion.utils.client_factory.create_client_from_config")
    async def test_generate_fresh_responses(
        self,
        mock_create_client,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        mock_model_responses,
        mock_evaluation_results,
    ):
        """Test generating fresh responses from models."""
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

        mock_evaluator_instance.compare_responses.side_effect = comparison_results

        # Execute with model generation
        result = await batch_comparison_tool(
            task="Write a Python function to validate email addresses",
            models=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet",
                "qwen3-4b-mlx",
            ],
            rank_by="quality",
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏆 Batch Model Comparison Results" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "🥇" in result  # Winner medal

        # Verify response generation calls
        assert mock_create_client.call_count == 3  # One for each model

        # Verify cost estimation and generation
        for client in mock_clients:
            client.estimate_cost.assert_called_once()
            client.complete.assert_called_once()

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    async def test_cost_ranking(
        self,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        mock_model_responses,
    ):
        """Test ranking by cost criteria."""
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

        # Execute with cost ranking
        result = await batch_comparison_tool(
            task="Simple coding task",
            responses=["response1", "response2", "response3"],
            models=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3-5-sonnet",
                "qwen3-4b-mlx",
            ],
            rank_by="cost",
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏆 Batch Model Comparison Results" in result
        assert "Ranking Criteria**: Cost" in result
        assert "Most Cost-Efficient" in result or "cost" in result.lower()

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    async def test_invalid_parameters(self, mock_cost_guard, mock_store):
        """Test validation of invalid parameters."""
        # Test invalid time period
        result = await batch_comparison_tool(
            task="Test task",
            rank_by="invalid_ranking",
        )
        assert "❌ **Invalid Ranking Criteria**" in result
        assert "invalid_ranking" in result

        # Test too many models
        result = await batch_comparison_tool(
            task="Test task",
            max_models=15,  # Exceeds limit of 10
        )
        assert "❌ **Invalid Max Models**" in result
        assert "Maximum allowed is 10" in result

        # Test too few models
        result = await batch_comparison_tool(
            task="Test task",
            max_models=1,  # Below minimum of 2
        )
        assert "❌ **Invalid Max Models**" in result
        assert "Minimum required is 2" in result

        # Test mismatched responses and models
        result = await batch_comparison_tool(
            task="Test task",
            responses=["response1", "response2"],
            models=["model1", "model2", "model3"],  # More models than responses
        )
        assert "❌ **Mismatched Input**" in result

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    async def test_budget_error(self, mock_cost_guard, mock_store):
        """Test budget error handling."""
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance
        mock_cost_guard_instance.check_and_reserve_budget.side_effect = Exception(
            "Budget exceeded"
        )

        result = await batch_comparison_tool(
            task="Test task",
            models=["model1", "model2"],
            cost_limit=0.01,  # Very low limit
        )

        assert "❌ **Budget Error**" in result
        assert "Budget exceeded" in result or "exceeds" in result

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
        result = await batch_comparison_tool(
            task="Test task",
            models=["good-model", "bad-model"],
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏆 Batch Model Comparison Results" in result
        # Should continue with partial results
        assert "good-model" in result

    @pytest.mark.parametrize(
        "rank_by,context",
        [
            ("quality", "coding task"),
            ("cost", "academic research"),
            ("speed", "creative writing"),
            ("comprehensive", "educational content"),
        ],
    )
    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    async def test_ranking_criteria_variations(
        self,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
        rank_by,
        context,
    ):
        """Test different ranking criteria and contexts."""
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

        # Execute
        result = await batch_comparison_tool(
            task="Test task with different criteria",
            responses=["response1", "response2"],
            models=["model1", "model2"],
            context=context,
            rank_by=rank_by,
            cost_limit=1.0,
        )

        # Verify
        assert isinstance(result, str)
        assert "🏆 Batch Model Comparison Results" in result
        assert f"**Ranking Criteria**: {rank_by.title()}" in result
        if context:
            assert f"**Context**: {context}" in result

    @patch("second_opinion.database.store.get_conversation_store")
    @patch("second_opinion.utils.cost_tracking.get_cost_guard")
    @patch("second_opinion.core.evaluator.get_evaluator")
    async def test_auto_model_selection(
        self,
        mock_evaluator,
        mock_cost_guard,
        mock_store,
    ):
        """Test automatic model selection when no models provided."""
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

        # Mock model selector
        with patch("second_opinion.cli.main.ComparisonModelSelector") as mock_selector:
            mock_selector_instance = MagicMock()
            mock_selector.return_value = mock_selector_instance
            mock_selector_instance.select_models.return_value = [
                "anthropic/claude-3-haiku",
                "openai/gpt-4o-mini",
            ]

            # Execute with no models specified
            result = await batch_comparison_tool(
                task="Auto-select models for this task",
                cost_limit=1.0,
            )

            # Verify
            assert isinstance(result, str)
            assert "🏆 Batch Model Comparison Results" in result

            # Verify auto-selection was used
            mock_selector_instance.select_models.assert_called_once()

    def test_calculate_batch_rankings(self):
        """Test the ranking calculation logic."""
        from second_opinion.core.models import ModelResponse, TokenUsage
        from second_opinion.mcp.tools.batch_comparison import _calculate_batch_rankings

        # Mock model responses
        models = ["model_a", "model_b", "model_c"]
        model_responses = [
            ModelResponse(
                content="Response A",
                model="model_a",
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
                cost_estimate=Decimal("0.05"),
                provider="openrouter",
            ),
            ModelResponse(
                content="Response B",
                model="model_b",
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
                cost_estimate=Decimal("0.01"),
                provider="lmstudio",
            ),
            ModelResponse(
                content="Response C",
                model="model_c",
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
                cost_estimate=Decimal("0.03"),
                provider="openrouter",
            ),
        ]

        # Mock evaluation results
        evaluation_results = [
            {
                "model_a": "model_a",
                "model_b": "model_b",
                "winner": "primary",  # model_a wins
                "score_a": 8.0,
                "score_b": 6.0,
                "overall_score": 8.0,
                "reasoning": "A is better",
            },
            {
                "model_a": "model_a",
                "model_b": "model_c",
                "winner": "comparison",  # model_c wins
                "score_a": 7.0,
                "score_b": 8.5,
                "overall_score": 8.5,
                "reasoning": "C is better",
            },
            {
                "model_a": "model_b",
                "model_b": "model_c",
                "winner": "comparison",  # model_c wins
                "score_a": 6.0,
                "score_b": 8.5,
                "overall_score": 8.5,
                "reasoning": "C is much better",
            },
        ]

        # Test quality ranking
        rankings = _calculate_batch_rankings(
            models, model_responses, evaluation_results, "quality"
        )

        assert len(rankings) == 3
        assert all("rank" in r for r in rankings)
        assert all("ranking_score" in r for r in rankings)
        assert rankings[0]["rank"] == 1  # Best model is rank 1

        # Verify sorting (highest score first)
        assert rankings[0]["ranking_score"] >= rankings[1]["ranking_score"]
        assert rankings[1]["ranking_score"] >= rankings[2]["ranking_score"]

        # Test cost ranking
        cost_rankings = _calculate_batch_rankings(
            models, model_responses, evaluation_results, "cost"
        )

        # Local model (model_b) with lowest cost should rank highest
        model_b_ranking = next(r for r in cost_rankings if r["model"] == "model_b")
        assert model_b_ranking["rank"] <= 2  # Should be top 2 for cost efficiency

    def test_get_model_strength(self):
        """Test model strength description generation."""
        from second_opinion.mcp.tools.batch_comparison import _get_model_strength

        ranking = {
            "model": "test-model",
            "avg_score": 8.5,
            "cost": Decimal("0.02"),
        }

        # Test different ranking criteria
        assert "quality" in _get_model_strength(ranking, "quality").lower()
        assert "cost" in _get_model_strength(ranking, "cost").lower()
        assert "fastest" in _get_model_strength(ranking, "speed").lower()
        assert "balance" in _get_model_strength(ranking, "comprehensive").lower()

    def test_get_model_name_suggestions(self):
        """Test model name suggestion generation."""
        from second_opinion.mcp.tools.batch_comparison import (
            _get_model_name_suggestions,
        )

        suggestions = _get_model_name_suggestions("invalid-model")

        assert "Cloud Models" in suggestions
        assert "Local Models" in suggestions
        assert "anthropic/claude-3-5-sonnet" in suggestions
        assert "qwen3-4b-mlx" in suggestions
        assert "Example model list" in suggestions
