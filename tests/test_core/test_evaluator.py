"""
Tests for response evaluation and comparison engine.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from second_opinion.core.evaluator import (
    ResponseEvaluator,
    get_evaluator,
    set_evaluator,
)
from second_opinion.core.models import (
    ComparisonResult,
    EvaluationCriteria,
    ModelResponse,
    RecommendationResult,
    RecommendationType,
    TaskComplexity,
    TokenUsage,
)


@pytest.fixture
def sample_response_primary():
    """Create a sample primary model response."""
    return ModelResponse(
        content="This is a detailed response from the primary model. It covers the main points comprehensively and provides good explanations.",
        model="gpt-4o",
        usage=TokenUsage(input_tokens=100, output_tokens=150, total_tokens=250),
        cost_estimate=Decimal("0.01"),
        provider="openai"
    )


@pytest.fixture
def sample_response_comparison():
    """Create a sample comparison model response."""
    return ModelResponse(
        content="This is a response from the comparison model. It's shorter but still helpful.",
        model="gpt-3.5-turbo",
        usage=TokenUsage(input_tokens=80, output_tokens=80, total_tokens=160),
        cost_estimate=Decimal("0.002"),
        provider="openai"
    )


@pytest.fixture
def sample_evaluation_criteria():
    """Create sample evaluation criteria."""
    return EvaluationCriteria(
        accuracy_weight=0.4,
        completeness_weight=0.3,
        clarity_weight=0.2,
        usefulness_weight=0.1
    )


@pytest.fixture
def evaluator():
    """Create a ResponseEvaluator instance."""
    return ResponseEvaluator()


class TestResponseEvaluator:
    def test_initialization(self, evaluator):
        """Test ResponseEvaluator initialization."""
        assert evaluator.cost_guard is not None
        assert evaluator.default_criteria is not None
        assert isinstance(evaluator.model_tiers, dict)
        assert isinstance(evaluator.complexity_indicators, dict)

    def test_model_tier_classification(self, evaluator):
        """Test model tier classification."""
        assert evaluator._get_model_tier("gpt-3.5-turbo") == "budget"
        assert evaluator._get_model_tier("gpt-4o") == "premium"
        assert evaluator._get_model_tier("claude-3-haiku") == "budget"
        assert evaluator._get_model_tier("claude-3.5-sonnet") == "premium"
        assert evaluator._get_model_tier("unknown-model") == "mid_range"  # Default

    @pytest.mark.asyncio
    async def test_classify_task_complexity_simple(self, evaluator):
        """Test classification of simple tasks."""
        simple_tasks = [
            "What is Python?",
            "Define machine learning",
            "List the benefits of exercise",
            "When was Python created?"
        ]

        for task in simple_tasks:
            complexity = await evaluator.classify_task_complexity(task)
            assert complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]

    @pytest.mark.asyncio
    async def test_classify_task_complexity_moderate(self, evaluator):
        """Test classification of moderate tasks."""
        moderate_tasks = [
            "Analyze the benefits and drawbacks of remote work",
            "Compare Python and JavaScript for web development",
            "Explain how neural networks work",
            "Summarize the main points of this article"
        ]

        for task in moderate_tasks:
            complexity = await evaluator.classify_task_complexity(task)
            assert complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]

    @pytest.mark.asyncio
    async def test_classify_task_complexity_complex(self, evaluator):
        """Test classification of complex tasks."""
        complex_tasks = [
            "Design a distributed system for processing millions of transactions",
            "Evaluate the economic impact of artificial intelligence on employment",
            "Create a comprehensive marketing strategy for a new product",
            "Solve this optimization problem with multiple constraints"
        ]

        for task in complex_tasks:
            complexity = await evaluator.classify_task_complexity(task)
            assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]

    @pytest.mark.asyncio
    async def test_classify_task_complexity_expert(self, evaluator):
        """Test classification of expert-level tasks."""
        expert_tasks = [
            "Derive a mathematical proof for this theorem",
            "Conduct advanced research on quantum computing applications",
            "Formulate a comprehensive theory of economic behavior",
            "Develop sophisticated machine learning algorithms"
        ]

        for task in expert_tasks:
            complexity = await evaluator.classify_task_complexity(task)
            assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]

    @pytest.mark.asyncio
    async def test_classify_task_complexity_length_heuristics(self, evaluator):
        """Test that task length affects complexity classification."""
        short_task = "Hi"
        medium_task = "Please explain the concept of machine learning and its applications in modern technology."
        long_task = "Please provide a comprehensive analysis of the socioeconomic implications of artificial intelligence adoption across various industries, considering both positive and negative effects, potential regulatory frameworks, and long-term societal impacts. Include specific examples and cite relevant research studies."

        short_complexity = await evaluator.classify_task_complexity(short_task)
        medium_complexity = await evaluator.classify_task_complexity(medium_task)
        long_complexity = await evaluator.classify_task_complexity(long_task)

        # Short tasks should trend toward simple
        assert short_complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
        # Long tasks should trend toward complex
        assert long_complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT]

    @pytest.mark.asyncio
    @patch("second_opinion.core.evaluator.get_client_for_model")
    @patch("second_opinion.core.evaluator.create_client_from_config")
    async def test_compare_responses_basic(
        self,
        mock_create_client_from_config,
        mock_get_client_for_model,
        evaluator,
        sample_response_primary,
        sample_response_comparison,
        sample_evaluation_criteria
    ):
        """Test basic response comparison."""
        # Mock the evaluator client
        mock_client = AsyncMock()
        mock_client.estimate_cost.return_value = Decimal("0.01")
        mock_client.complete.return_value = MagicMock(
            content="Analysis: Response A wins in accuracy. Response B wins in clarity. Overall: Response A is better.",
            cost_estimate=Decimal("0.01")
        )
        mock_get_client_for_model.return_value = mock_client
        mock_create_client_from_config.return_value = mock_client
        
        # Mock the cost guard to avoid reservation issues
        evaluator.cost_guard.check_and_reserve_budget = AsyncMock()
        evaluator.cost_guard.record_actual_cost = AsyncMock()
        
        original_task = "Explain the benefits of exercise"

        result = await evaluator.compare_responses(
            sample_response_primary,
            sample_response_comparison,
            original_task,
            sample_evaluation_criteria
        )

        assert isinstance(result, ComparisonResult)
        assert result.primary_model == "gpt-4o"
        assert result.comparison_model == "gpt-3.5-turbo"
        assert 0.0 <= result.accuracy_score <= 10.0
        assert 0.0 <= result.completeness_score <= 10.0
        assert 0.0 <= result.clarity_score <= 10.0
        assert 0.0 <= result.usefulness_score <= 10.0
        assert 0.0 <= result.overall_score <= 10.0
        assert result.winner in ["primary", "comparison", "tie"]
        assert len(result.reasoning) > 0
        assert result.cost_analysis is not None

    @pytest.mark.asyncio
    @patch("second_opinion.core.evaluator.get_client_for_model")
    @patch("second_opinion.core.evaluator.create_client_from_config")
    async def test_compare_responses_default_criteria(
        self,
        mock_create_client_from_config,
        mock_get_client_for_model,
        evaluator,
        sample_response_primary,
        sample_response_comparison
    ):
        """Test response comparison with default criteria."""
        # Mock the evaluator client
        mock_client = AsyncMock()
        mock_client.estimate_cost.return_value = Decimal("0.01")
        mock_client.complete.return_value = MagicMock(
            content="Analysis: Response A wins in accuracy. Response B wins in clarity. Overall: Response A is better.",
            cost_estimate=Decimal("0.01")
        )
        mock_get_client_for_model.return_value = mock_client
        mock_create_client_from_config.return_value = mock_client
        
        # Mock the cost guard to avoid reservation issues
        evaluator.cost_guard.check_and_reserve_budget = AsyncMock()
        evaluator.cost_guard.record_actual_cost = AsyncMock()
        
        original_task = "What is machine learning?"

        result = await evaluator.compare_responses(
            sample_response_primary,
            sample_response_comparison,
            original_task
        )

        assert isinstance(result, ComparisonResult)
        assert result.cost_analysis.estimated_cost > 0

    @pytest.mark.asyncio
    async def test_recommend_model_tier_simple_task(self, evaluator):
        """Test model recommendation for simple tasks."""
        task = "What is 2 + 2?"
        current_model = "gpt-4o"

        result = await evaluator.recommend_model_tier(task, current_model)

        assert isinstance(result, RecommendationResult)
        assert result.current_model == current_model
        assert result.task_complexity == TaskComplexity.SIMPLE
        assert result.recommended_action == RecommendationType.DOWNGRADE
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning is not None

    @pytest.mark.asyncio
    async def test_recommend_model_tier_complex_task(self, evaluator):
        """Test model recommendation for complex tasks."""
        task = "Design a comprehensive distributed system architecture for a global social media platform"
        current_model = "gpt-3.5-turbo"

        result = await evaluator.recommend_model_tier(task, current_model)

        assert isinstance(result, RecommendationResult)
        assert result.current_model == current_model
        assert result.task_complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]
        assert result.recommended_action == RecommendationType.UPGRADE
        assert result.recommended_model is not None
        assert result.expected_improvement is not None
        assert result.expected_improvement > 0

    @pytest.mark.asyncio
    async def test_recommend_model_tier_with_response(self, evaluator):
        """Test model recommendation when current response is provided."""
        task = "Explain photosynthesis"
        current_model = "gpt-4o"
        current_response = "Photosynthesis is the process by which plants convert sunlight into energy."

        result = await evaluator.recommend_model_tier(
            task,
            current_model,
            current_response
        )

        assert isinstance(result, RecommendationResult)
        assert result.current_quality_score > 0

    @pytest.mark.asyncio
    async def test_recommend_model_tier_cost_constraint(self, evaluator):
        """Test model recommendation with cost constraints."""
        task = "Complex analysis task"
        current_model = "gpt-3.5-turbo"
        max_cost_increase = Decimal("0.001")  # Very small increase

        result = await evaluator.recommend_model_tier(
            task,
            current_model,
            max_cost_increase=max_cost_increase
        )

        assert isinstance(result, RecommendationResult)
        # Should consider cost constraints in recommendation
        assert result.cost_impact <= max_cost_increase or result.recommended_action == RecommendationType.MAINTAIN

    @pytest.mark.asyncio
    async def test_evaluate_cost_effectiveness_empty_list(self, evaluator):
        """Test cost-effectiveness evaluation with empty response list."""
        result = await evaluator.evaluate_cost_effectiveness([], TaskComplexity.MODERATE)

        assert "error" in result
        assert result["error"] == "No responses to evaluate"

    @pytest.mark.asyncio
    async def test_evaluate_cost_effectiveness_single_response(
        self,
        evaluator,
        sample_response_primary
    ):
        """Test cost-effectiveness evaluation with single response."""
        result = await evaluator.evaluate_cost_effectiveness(
            [sample_response_primary],
            TaskComplexity.MODERATE
        )

        assert result["task_complexity"] == "moderate"
        assert len(result["analyses"]) == 1
        assert result["most_cost_effective"] is not None
        assert result["most_expensive"] is not None

        analysis = result["analyses"][0]
        assert analysis["model"] == "gpt-4o"
        assert analysis["cost"] == float(sample_response_primary.cost_estimate)
        assert analysis["quality_score"] > 0
        assert analysis["cost_per_quality"] > 0

    @pytest.mark.asyncio
    async def test_evaluate_cost_effectiveness_multiple_responses(
        self,
        evaluator,
        sample_response_primary,
        sample_response_comparison
    ):
        """Test cost-effectiveness evaluation with multiple responses."""
        responses = [sample_response_primary, sample_response_comparison]

        result = await evaluator.evaluate_cost_effectiveness(
            responses,
            TaskComplexity.COMPLEX
        )

        assert result["task_complexity"] == "complex"
        assert len(result["analyses"]) == 2
        assert result["most_cost_effective"] is not None
        assert result["most_expensive"] is not None

        # Should be sorted by cost-effectiveness
        analyses = result["analyses"]
        assert analyses[0]["cost_per_quality"] <= analyses[1]["cost_per_quality"]

    @pytest.mark.asyncio
    async def test_evaluate_response_quality_basic(self, evaluator):
        """Test basic response quality evaluation."""
        task = "Explain machine learning"
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."

        quality = await evaluator._evaluate_response_quality(task, response, "gpt-4o-mini")

        assert 1.0 <= quality <= 10.0

    @pytest.mark.asyncio
    async def test_evaluate_response_quality_short_response(self, evaluator):
        """Test quality evaluation of short response."""
        task = "What is AI?"
        response = "AI is artificial intelligence."

        quality = await evaluator._evaluate_response_quality(task, response, "gpt-4o-mini")

        # Short responses should get lower scores
        assert quality < 7.0

    @pytest.mark.asyncio
    async def test_evaluate_response_quality_relevant_response(self, evaluator):
        """Test quality evaluation of relevant response."""
        task = "Explain machine learning algorithms"
        response = "Machine learning algorithms are computational methods that allow systems to learn patterns from data. Common algorithms include neural networks, decision trees, and support vector machines."

        quality = await evaluator._evaluate_response_quality(task, response, "gpt-4o-mini")

        # Relevant responses should get higher scores
        assert quality > 5.0

    @pytest.mark.asyncio
    async def test_calculate_cost_impact_no_recommendation(self, evaluator):
        """Test cost impact calculation with no recommendation."""
        impact = await evaluator._calculate_cost_impact("gpt-4o", None)
        assert impact == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_cost_impact_upgrade(self, evaluator):
        """Test cost impact calculation for upgrade."""
        impact = await evaluator._calculate_cost_impact("gpt-3.5-turbo", "gpt-4o")
        assert impact > 0  # Upgrade should cost more

    @pytest.mark.asyncio
    async def test_calculate_cost_impact_downgrade(self, evaluator):
        """Test cost impact calculation for downgrade."""
        impact = await evaluator._calculate_cost_impact("gpt-4o", "gpt-3.5-turbo")
        assert impact < 0  # Downgrade should cost less



class TestGlobalEvaluator:
    def test_global_evaluator_singleton(self):
        """Test that global evaluator maintains singleton behavior."""
        evaluator1 = get_evaluator()
        evaluator2 = get_evaluator()

        assert evaluator1 is evaluator2

    def test_set_global_evaluator(self):
        """Test setting a custom global evaluator."""
        custom_evaluator = ResponseEvaluator()
        set_evaluator(custom_evaluator)

        retrieved_evaluator = get_evaluator()
        assert retrieved_evaluator is custom_evaluator


class TestEvaluationCriteria:
    def test_custom_criteria_weights(self, evaluator, sample_response_primary, sample_response_comparison):
        """Test using custom evaluation criteria weights."""
        # Create criteria heavily weighted toward accuracy
        custom_criteria = EvaluationCriteria(
            accuracy_weight=0.7,
            completeness_weight=0.1,
            clarity_weight=0.1,
            usefulness_weight=0.1
        )

        # The comparison should work with custom criteria
        # (actual behavior depends on the simulation logic)
        assert custom_criteria.accuracy_weight == 0.7
        assert abs(sum([
            custom_criteria.accuracy_weight,
            custom_criteria.completeness_weight,
            custom_criteria.clarity_weight,
            custom_criteria.usefulness_weight
        ]) - 1.0) < 0.01


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_task_complexity(self, evaluator):
        """Test task complexity classification with empty task."""
        complexity = await evaluator.classify_task_complexity("")
        assert complexity == TaskComplexity.MODERATE  # Default

    @pytest.mark.asyncio
    async def test_very_long_task_complexity(self, evaluator):
        """Test task complexity classification with very long task."""
        long_task = "This is a very long task description. " * 50
        complexity = await evaluator.classify_task_complexity(long_task)
        assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT, TaskComplexity.MODERATE]

    @pytest.mark.asyncio
    async def test_multiple_questions_complexity(self, evaluator):
        """Test that multiple questions increase complexity."""
        multi_question_task = "What is AI? How does it work? What are the applications? What are the risks?"
        complexity = await evaluator.classify_task_complexity(multi_question_task)

        # Multiple questions should trend toward higher complexity
        assert complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT]

    @pytest.mark.asyncio
    async def test_recommend_with_poor_quality_response(self, evaluator):
        """Test recommendation when current response quality is poor."""
        task = "Simple question"
        current_model = "gpt-4o"
        poor_response = "Umm, I don't know."

        result = await evaluator.recommend_model_tier(
            task,
            current_model,
            poor_response
        )

        # Should recommend upgrade due to poor quality
        assert result.recommended_action in [RecommendationType.UPGRADE, RecommendationType.MAINTAIN]


# Test cleanup
@pytest.fixture(autouse=True)
def reset_global_evaluator():
    """Reset global evaluator state after each test."""
    yield
    # Reset to a fresh evaluator
    set_evaluator(ResponseEvaluator())
