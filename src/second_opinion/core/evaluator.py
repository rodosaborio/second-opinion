"""
Response evaluation and comparison engine.

This module provides the core logic for evaluating model responses,
comparing them across different criteria, and making recommendations
for model upgrades or downgrades.
"""

import logging
import re
from decimal import Decimal
from typing import Any

from ..clients import detect_model_provider
from ..prompts.manager import render_template
from ..utils.client_factory import create_client_from_config
from ..utils.cost_tracking import CostGuard
from .models import (
    ComparisonResult,
    CostAnalysis,
    EvaluationCriteria,
    EvaluationError,
    Message,
    ModelRequest,
    ModelResponse,
    RecommendationResult,
    RecommendationType,
    SecurityContext,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """
    Core logic for comparing and evaluating model responses.

    This class provides functionality for:
    - Comparing responses from different models
    - Evaluating response quality across multiple criteria
    - Making model tier recommendations
    - Task complexity assessment
    """

    def __init__(self, cost_guard: CostGuard | None = None):
        """
        Initialize the response evaluator.

        Args:
            cost_guard: Cost tracking and budget management instance
        """
        self.cost_guard = cost_guard or CostGuard()

        # Default evaluation criteria weights
        self.default_criteria = EvaluationCriteria()

        # Model tier mappings (can be configured)
        self.model_tiers = {
            "budget": ["gpt-3.5-turbo", "claude-3-haiku"],
            "mid_range": ["gpt-4o-mini", "claude-3.5-haiku"],
            "premium": ["gpt-4o", "claude-3.5-sonnet"],
            "reasoning": ["o1-preview", "o1-mini", "claude-3-opus"],
        }

        # Task complexity indicators
        self.complexity_indicators = {
            "simple": [
                "what is",
                "define",
                "explain briefly",
                "list",
                "when",
                "where",
                "who",
                "basic",
                "simple",
                "show me",
                "tell me",
            ],
            "moderate": [
                "analyze",
                "compare",
                "explain how",
                "why",
                "summarize",
                "outline",
                "describe",
                "discuss",
                "how does",
                "what are the",
            ],
            "complex": [
                "evaluate",
                "synthesize",
                "critique",
                "design",
                "develop",
                "create",
                "solve",
                "optimize",
                "recommend",
                "implement",
                "build",
            ],
            "expert": [
                "research",
                "prove",
                "derive",
                "formulate",
                "theorize",
                "advanced",
                "sophisticated",
                "comprehensive analysis",
                "mathematical",
            ],
        }

    async def compare_responses(
        self,
        primary_response: ModelResponse,
        comparison_response: ModelResponse,
        original_task: str,
        criteria: EvaluationCriteria | None = None,
        evaluator_model: str | None = None,
    ) -> ComparisonResult:
        """
        Compare two model responses across multiple evaluation criteria.

        Args:
            primary_response: Response from the primary model
            comparison_response: Response from the comparison model
            original_task: The original task/question
            criteria: Evaluation criteria (uses defaults if not provided)
            evaluator_model: Model to use for evaluation (defaults to primary model if None)

        Returns:
            Detailed comparison results

        Raises:
            CostLimitExceededError: If evaluation would exceed budget
            ValueError: If responses are invalid
        """
        if not criteria:
            criteria = self.default_criteria

        # Use primary model as evaluator if none specified
        if evaluator_model is None:
            evaluator_model = primary_response.model
            logger.info(f"Using primary model '{evaluator_model}' as evaluator")

        logger.info(
            f"Comparing responses: {primary_response.model} vs {comparison_response.model} (evaluator: {evaluator_model})"
        )

        # Prepare evaluation prompt (match template parameter names)
        evaluation_params = {
            "original_question": original_task,
            "model_a": primary_response.model,
            "response_a": primary_response.content,
            "model_b": comparison_response.model,
            "response_b": comparison_response.content,
            "accuracy_weight": criteria.accuracy_weight,
            "completeness_weight": criteria.completeness_weight,
            "clarity_weight": criteria.clarity_weight,
            "usefulness_weight": criteria.usefulness_weight,
        }

        # Render evaluation prompt
        evaluation_prompt = await render_template(
            "comparison",
            evaluation_params,
            model=evaluator_model,
            security_context=SecurityContext.SYSTEM_PROMPT,
        )

        # Make actual API call to evaluator model
        try:
            evaluation_result = await self._evaluate_with_model(
                evaluation_prompt,
                primary_response,
                comparison_response,
                criteria,
                evaluator_model,
            )
        except EvaluationError:
            # Re-raise evaluation errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to EvaluationError with context
            raise EvaluationError(
                f"Unexpected error during evaluation: {str(e)}",
                model=evaluator_model,
                cause=e,
            ) from e

        # Calculate cost analysis with real budget data
        total_cost = primary_response.cost_estimate + comparison_response.cost_estimate

        # Get real budget information from cost guard
        try:
            from ..utils.cost_tracking import BudgetPeriod

            budget_usage = await self.cost_guard.get_usage_summary(BudgetPeriod.DAILY)
            budget_remaining = budget_usage.available
        except Exception as e:
            logger.warning(f"Failed to get budget data: {e}. Using fallback value.")
            budget_remaining = Decimal("100.00")  # Fallback if cost guard fails

        cost_analysis = CostAnalysis(
            estimated_cost=total_cost,
            actual_cost=total_cost,  # Simplified for now
            cost_per_token=total_cost
            / max(
                1,
                primary_response.usage.total_tokens
                + comparison_response.usage.total_tokens,
            ),
            budget_remaining=budget_remaining,
        )

        # Generate recommendations based on evaluation results
        recommendations = self._generate_recommendations(
            primary_response, comparison_response, evaluation_result
        )

        # Create comparison result
        result = ComparisonResult(
            primary_response=primary_response.content,
            comparison_response=comparison_response.content,
            primary_model=primary_response.model,
            comparison_model=comparison_response.model,
            accuracy_score=evaluation_result["accuracy_score"],
            completeness_score=evaluation_result["completeness_score"],
            clarity_score=evaluation_result["clarity_score"],
            usefulness_score=evaluation_result["usefulness_score"],
            overall_score=evaluation_result["overall_score"],
            winner=evaluation_result["winner"],
            reasoning=evaluation_result["reasoning"],
            cost_analysis=cost_analysis,
            recommendations=recommendations,
        )

        logger.info(
            f"Comparison complete. Winner: {result.winner}, Score: {result.overall_score:.2f}"
        )
        return result

    async def recommend_model_tier(
        self,
        task: str,
        current_model: str,
        current_response: str | None = None,
        max_cost_increase: Decimal | None = None,
        evaluator_model: str = "gpt-4o-mini",
    ) -> RecommendationResult:
        """
        Recommend whether to upgrade, downgrade, or maintain current model.

        Args:
            task: The task description
            current_model: Currently used model
            current_response: Current model's response (optional)
            max_cost_increase: Maximum acceptable cost increase
            evaluator_model: Model to use for evaluation

        Returns:
            Model recommendation with reasoning
        """
        logger.info(f"Analyzing model recommendation for {current_model}")

        # Assess task complexity
        task_complexity = await self.classify_task_complexity(task)

        # Get current model tier
        current_tier = self._get_model_tier(current_model)

        # Evaluate current response quality if provided
        current_quality_score = 7.0  # Default assumption
        if current_response:
            current_quality_score = await self._evaluate_response_quality(
                task, current_response, evaluator_model
            )

        # Determine recommendation based on complexity and current quality
        recommendation = await self._determine_recommendation(
            task_complexity, current_tier, current_quality_score, max_cost_increase
        )

        # Calculate cost impact
        cost_impact = await self._calculate_cost_impact(
            current_model, recommendation.get("recommended_model")
        )

        # Create cost analysis
        cost_analysis = CostAnalysis(
            estimated_cost=Decimal("0.05"),  # Simplified
            actual_cost=Decimal("0.05"),
            cost_per_token=Decimal("0.0001"),
            budget_remaining=Decimal("100.00"),
        )

        # Create recommendation result
        result = RecommendationResult(
            current_model=current_model,
            recommended_action=recommendation["action"],
            recommended_model=recommendation.get("recommended_model"),
            task_complexity=task_complexity,
            confidence=recommendation["confidence"],
            current_quality_score=current_quality_score,
            expected_improvement=recommendation.get("expected_improvement"),
            cost_impact=cost_impact,
            reasoning=recommendation["reasoning"],
            cost_analysis=cost_analysis,
        )

        logger.info(
            f"Recommendation: {result.recommended_action} (confidence: {result.confidence:.2f})"
        )
        return result

    async def classify_task_complexity(self, task: str) -> TaskComplexity:
        """
        Classify the complexity of a given task.

        Args:
            task: Task description to classify

        Returns:
            Task complexity level
        """
        if not task or not task.strip():
            return TaskComplexity.MODERATE

        task_lower = task.lower()

        # Count indicators for each complexity level with weighted scoring
        complexity_scores = {
            TaskComplexity.SIMPLE: 0,
            TaskComplexity.MODERATE: 0,
            TaskComplexity.COMPLEX: 0,
            TaskComplexity.EXPERT: 0,
        }

        # Score based on keyword indicators with word boundaries
        import re

        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                # Use word boundaries for single words, or exact phrase match for multi-word indicators
                if len(indicator.split()) == 1:
                    pattern = r"\b" + re.escape(indicator) + r"\b"
                    if re.search(pattern, task_lower):
                        complexity_scores[TaskComplexity(complexity)] += 1
                else:
                    if indicator in task_lower:
                        weight = len(indicator.split())
                        complexity_scores[TaskComplexity(complexity)] += weight

        # Additional heuristics with lighter weighting
        if len(task) < 20:  # Only very short tasks get simple bonus
            complexity_scores[TaskComplexity.SIMPLE] += 1
        elif len(task) > 300:  # Only very long tasks get complex bonus
            complexity_scores[TaskComplexity.COMPLEX] += 1

        # Count questions (multiple questions = more complex)
        question_count = task.count("?")
        if question_count > 1:
            complexity_scores[TaskComplexity.COMPLEX] += question_count

        # Technical terms indicate higher complexity, but not for simple questions
        simple_question_patterns = [
            "what is",
            "define",
            "when was",
            "where is",
            "who is",
        ]
        is_simple_question = any(
            pattern in task_lower for pattern in simple_question_patterns
        )

        if not is_simple_question:
            technical_terms = [
                "algorithm",
                "system",
                "architecture",
                "framework",
                "optimization",
                "neural",
                "machine learning",
            ]
            for term in technical_terms:
                if term in task_lower:
                    complexity_scores[TaskComplexity.COMPLEX] += 2

        # Determine highest scoring complexity
        max_score = max(complexity_scores.values())
        if max_score == 0:
            return TaskComplexity.MODERATE  # Default

        # Find the highest complexity level with the max score
        for complexity in [
            TaskComplexity.EXPERT,
            TaskComplexity.COMPLEX,
            TaskComplexity.MODERATE,
            TaskComplexity.SIMPLE,
        ]:
            if complexity_scores[complexity] == max_score:
                logger.debug(
                    f"Task classified as {complexity.value} (score: {max_score}): {task[:100]}..."
                )
                return complexity

        return TaskComplexity.MODERATE

    async def evaluate_cost_effectiveness(
        self, responses: list[ModelResponse], task_complexity: TaskComplexity
    ) -> dict[str, Any]:
        """
        Evaluate cost-effectiveness of different model responses.

        Args:
            responses: List of model responses to evaluate
            task_complexity: Complexity of the task

        Returns:
            Cost-effectiveness analysis
        """
        if not responses:
            return {"error": "No responses to evaluate"}

        analyses = []
        for response in responses:
            # Simple quality heuristic (would be replaced with actual evaluation)
            quality_score = min(10.0, len(response.content) / 100)  # Simplified

            cost_per_quality = float(response.cost_estimate) / max(0.1, quality_score)

            analyses.append(
                {
                    "model": response.model,
                    "cost": float(response.cost_estimate),
                    "quality_score": quality_score,
                    "cost_per_quality": cost_per_quality,
                    "tokens": response.usage.total_tokens,
                    "cost_per_token": float(response.cost_estimate)
                    / max(1, response.usage.total_tokens),
                }
            )

        # Sort by cost-effectiveness (lower cost_per_quality is better)
        analyses.sort(key=lambda x: x["cost_per_quality"])

        return {
            "task_complexity": task_complexity.value,
            "analyses": analyses,
            "most_cost_effective": analyses[0] if analyses else None,
            "most_expensive": analyses[-1] if analyses else None,
        }

    async def _evaluate_response_quality(
        self, task: str, response: str, evaluator_model: str
    ) -> float:
        """
        Evaluate the quality of a single response.

        Args:
            task: Original task
            response: Response to evaluate
            evaluator_model: Model to use for evaluation

        Returns:
            Quality score (0-10)
        """
        # Simple heuristic for now (would use actual evaluation in production)
        base_score = 5.0

        # Length heuristic
        if 100 <= len(response) <= 1000:
            base_score += 1.0
        elif len(response) < 50:
            base_score -= 1.0

        # Relevance heuristic (check if response contains task keywords)
        task_words = set(task.lower().split())
        response_words = set(response.lower().split())
        overlap = len(task_words.intersection(response_words)) / max(1, len(task_words))
        base_score += overlap * 2.0

        return min(10.0, max(1.0, base_score))

    def _generate_recommendations(
        self,
        primary_response: ModelResponse,
        comparison_response: ModelResponse,
        evaluation_result: dict[str, Any],
    ) -> list[str]:
        """
        Generate recommendations based on comparison results.

        Args:
            primary_response: Primary model response
            comparison_response: Comparison model response
            evaluation_result: Evaluation results with scores and winner

        Returns:
            List of recommendation strings
        """
        recommendations = []

        winner = evaluation_result.get("winner", "tie")
        primary_cost = float(primary_response.cost_estimate)
        comparison_cost = float(comparison_response.cost_estimate)

        # Handle zero-cost scenarios (likely pricing data issues)
        if primary_cost == 0.0 and comparison_cost == 0.0:
            if winner == "comparison":
                recommendations.append(
                    f"Consider using {comparison_response.model} for potentially better quality "
                    f"(cost data unavailable for comparison)"
                )
            elif winner == "primary":
                recommendations.append(
                    f"Your primary model ({primary_response.model}) appears to provide better quality "
                    f"(cost data unavailable for detailed recommendation)"
                )
            else:
                recommendations.append(
                    "Both models provide similar quality. Choose based on your specific requirements "
                    "(cost data unavailable for detailed comparison)"
                )
            return recommendations

        # Normal cost comparison logic
        cost_threshold = 0.001  # Only consider differences above $0.001

        if winner == "comparison":
            # Comparison model won
            if (
                comparison_cost < primary_cost
                and (primary_cost - comparison_cost) > cost_threshold
            ):
                savings = primary_cost - comparison_cost
                recommendations.append(
                    f"Consider using {comparison_response.model} instead - it provides better quality "
                    f"at ${savings:.4f} lower cost per request"
                )
            elif (
                comparison_cost > primary_cost
                and (comparison_cost - primary_cost) > cost_threshold
            ):
                cost_increase = comparison_cost - primary_cost
                recommendations.append(
                    f"Consider upgrading to {comparison_response.model} for better quality "
                    f"(${cost_increase:.4f} more per request)"
                )
            else:
                recommendations.append(
                    f"Consider switching to {comparison_response.model} for better quality at similar cost"
                )
        elif winner == "primary":
            # Primary model won
            if (
                primary_cost > comparison_cost
                and (primary_cost - comparison_cost) > cost_threshold
            ):
                cost_difference = primary_cost - comparison_cost
                recommendations.append(
                    f"Your primary model ({primary_response.model}) provides better quality, "
                    f"justifying the ${cost_difference:.4f} higher cost"
                )
            else:
                recommendations.append(
                    f"Your primary model ({primary_response.model}) provides the best value - "
                    f"good quality at competitive cost"
                )
        else:
            # Tie
            if (
                primary_cost > comparison_cost
                and (primary_cost - comparison_cost) > cost_threshold
            ):
                savings = primary_cost - comparison_cost
                recommendations.append(
                    f"Both models provide similar quality. Consider {comparison_response.model} "
                    f"to save ${savings:.4f} per request"
                )
            elif (
                comparison_cost > primary_cost
                and (comparison_cost - primary_cost) > cost_threshold
            ):
                recommendations.append(
                    f"Both models provide similar quality. Your current model ({primary_response.model}) "
                    f"is more cost-effective"
                )
            else:
                recommendations.append(
                    "Both models provide similar quality and cost. Choose based on other factors "
                    "like response time or specific capabilities"
                )

        return recommendations

    def _get_model_tier(self, model: str) -> str:
        """Get the tier classification for a model."""
        for tier, models in self.model_tiers.items():
            if any(model_name in model for model_name in models):
                return tier
        return "mid_range"  # Default

    async def _determine_recommendation(
        self,
        task_complexity: TaskComplexity,
        current_tier: str,
        current_quality: float,
        max_cost_increase: Decimal | None,
    ) -> dict[str, Any]:
        """
        Determine model recommendation based on task complexity and current performance.
        """
        # Check quality first - if quality is poor, recommend upgrade regardless of task complexity
        if current_quality < 6.0:
            return {
                "action": RecommendationType.UPGRADE,
                "recommended_model": "gpt-4o",
                "confidence": 0.7,
                "expected_improvement": 1.5,
                "reasoning": "Current response quality is below acceptable threshold",
            }

        # Quality is acceptable, now consider task complexity and current tier
        if (
            task_complexity == TaskComplexity.SIMPLE
            and current_tier not in ["budget"]
            and current_quality >= 7.0
        ):
            return {
                "action": RecommendationType.DOWNGRADE,
                "recommended_model": "gpt-3.5-turbo",
                "confidence": 0.8,
                "reasoning": "Task is simple and can be handled by a budget model with good quality",
            }
        elif task_complexity == TaskComplexity.EXPERT and current_tier in [
            "budget",
            "mid_range",
        ]:
            return {
                "action": RecommendationType.UPGRADE,
                "recommended_model": "gpt-4o",
                "confidence": 0.9,
                "expected_improvement": 2.0,
                "reasoning": "Expert-level task requires premium model capabilities",
            }
        elif task_complexity == TaskComplexity.COMPLEX and current_tier == "budget":
            return {
                "action": RecommendationType.UPGRADE,
                "recommended_model": "gpt-4o",
                "confidence": 0.8,
                "expected_improvement": 1.0,
                "reasoning": "Complex task would benefit from premium model capabilities",
            }
        else:
            return {
                "action": RecommendationType.MAINTAIN,
                "confidence": 0.8,
                "reasoning": "Current model is appropriate for this task complexity and quality level",
            }

    async def _calculate_cost_impact(
        self, current_model: str, recommended_model: str | None
    ) -> Decimal:
        """Calculate the cost impact of switching models."""
        if not recommended_model:
            return Decimal("0")

        # Simplified cost calculation (would use actual pricing in production)
        model_costs = {
            "gpt-3.5-turbo": Decimal("0.001"),
            "gpt-4o-mini": Decimal("0.005"),
            "gpt-4o": Decimal("0.01"),
            "claude-3-haiku": Decimal("0.002"),
            "claude-3.5-sonnet": Decimal("0.008"),
        }

        current_cost = model_costs.get(current_model, Decimal("0.005"))
        recommended_cost = model_costs.get(recommended_model, Decimal("0.005"))

        return recommended_cost - current_cost

    async def _evaluate_with_model(
        self,
        evaluation_prompt: str,
        primary_response: ModelResponse,
        comparison_response: ModelResponse,
        criteria: EvaluationCriteria,
        evaluator_model: str,
    ) -> dict[str, Any]:
        """
        Use a real model to evaluate the comparison.

        Args:
            evaluation_prompt: The formatted evaluation prompt
            primary_response: Primary model response
            comparison_response: Comparison model response
            criteria: Evaluation criteria with weights
            evaluator_model: Model to use for evaluation

        Returns:
            Evaluation result dictionary with scores and reasoning
        """
        # Create client for the evaluator model (detect provider automatically)
        try:
            provider = detect_model_provider(evaluator_model)
            client = create_client_from_config(provider)
        except Exception as e:
            raise EvaluationError(
                f"Failed to create client for evaluator model '{evaluator_model}': {str(e)}",
                model=evaluator_model,
                cause=e,
            ) from e

        # Prepare the request
        messages = [Message(role="user", content=evaluation_prompt)]
        request = ModelRequest(
            model=evaluator_model, messages=messages, max_tokens=1000, temperature=0.1
        )

        # Check cost and make API call
        estimated_cost = await client.estimate_cost(request)
        budget_check = await self.cost_guard.check_and_reserve_budget(
            estimated_cost, "evaluation", evaluator_model
        )

        # Make the evaluation request
        response = await client.complete(request)

        # Record actual cost using the reservation ID from the budget check
        await self.cost_guard.record_actual_cost(
            reservation_id=budget_check.reservation_id,
            actual_cost=response.cost_estimate,
            model=evaluator_model,
            operation_type="evaluation",
        )

        # Parse the evaluation response
        return self._parse_evaluation_response(
            response.content, primary_response, comparison_response, criteria
        )

    def _parse_evaluation_response(
        self,
        evaluation_text: str,
        primary_response: ModelResponse,
        comparison_response: ModelResponse,
        criteria: EvaluationCriteria,
    ) -> dict[str, Any]:
        """
        Parse the evaluation model's response to extract scores and reasoning.

        Args:
            evaluation_text: The evaluator model's response
            primary_response: Primary model response
            comparison_response: Comparison model response
            criteria: Evaluation criteria with weights

        Returns:
            Dictionary with scores, winner, and reasoning
        """
        # Initialize with default scores
        scores = {
            "accuracy_primary": 7.0,
            "accuracy_comparison": 7.0,
            "completeness_primary": 7.0,
            "completeness_comparison": 7.0,
            "clarity_primary": 7.0,
            "clarity_comparison": 7.0,
            "usefulness_primary": 7.0,
            "usefulness_comparison": 7.0,
        }

        # Try to extract winners from the structured response
        response_lower = evaluation_text.lower()

        # Look for winner declarations in each section
        accuracy_winner = self._extract_section_winner(evaluation_text, "accuracy")
        completeness_winner = self._extract_section_winner(
            evaluation_text, "completeness"
        )
        clarity_winner = self._extract_section_winner(evaluation_text, "clarity")
        usefulness_winner = self._extract_section_winner(evaluation_text, "usefulness")

        # Convert winners to scores (winner gets 8.5, loser gets 6.5, tie gets 7.5)
        if accuracy_winner == "a":
            scores["accuracy_primary"] = 8.5
            scores["accuracy_comparison"] = 6.5
        elif accuracy_winner == "b":
            scores["accuracy_primary"] = 6.5
            scores["accuracy_comparison"] = 8.5
        else:  # tie or unclear
            scores["accuracy_primary"] = 7.5
            scores["accuracy_comparison"] = 7.5

        if completeness_winner == "a":
            scores["completeness_primary"] = 8.5
            scores["completeness_comparison"] = 6.5
        elif completeness_winner == "b":
            scores["completeness_primary"] = 6.5
            scores["completeness_comparison"] = 8.5
        else:
            scores["completeness_primary"] = 7.5
            scores["completeness_comparison"] = 7.5

        if clarity_winner == "a":
            scores["clarity_primary"] = 8.5
            scores["clarity_comparison"] = 6.5
        elif clarity_winner == "b":
            scores["clarity_primary"] = 6.5
            scores["clarity_comparison"] = 8.5
        else:
            scores["clarity_primary"] = 7.5
            scores["clarity_comparison"] = 7.5

        if usefulness_winner == "a":
            scores["usefulness_primary"] = 8.5
            scores["usefulness_comparison"] = 6.5
        elif usefulness_winner == "b":
            scores["usefulness_primary"] = 6.5
            scores["usefulness_comparison"] = 8.5
        else:
            scores["usefulness_primary"] = 7.5
            scores["usefulness_comparison"] = 7.5

        # Calculate weighted overall scores
        primary_overall = (
            scores["accuracy_primary"] * criteria.accuracy_weight
            + scores["completeness_primary"] * criteria.completeness_weight
            + scores["clarity_primary"] * criteria.clarity_weight
            + scores["usefulness_primary"] * criteria.usefulness_weight
        )

        comparison_overall = (
            scores["accuracy_comparison"] * criteria.accuracy_weight
            + scores["completeness_comparison"] * criteria.completeness_weight
            + scores["clarity_comparison"] * criteria.clarity_weight
            + scores["usefulness_comparison"] * criteria.usefulness_weight
        )

        # Determine overall winner
        if abs(primary_overall - comparison_overall) < 0.5:
            winner = "tie"
        elif primary_overall > comparison_overall:
            winner = "primary"
        else:
            winner = "comparison"

        # Extract overall recommendation if present
        reasoning = self._extract_reasoning(
            evaluation_text, winner, primary_overall, comparison_overall
        )

        return {
            "accuracy_score": max(
                scores["accuracy_primary"], scores["accuracy_comparison"]
            ),
            "completeness_score": max(
                scores["completeness_primary"], scores["completeness_comparison"]
            ),
            "clarity_score": max(
                scores["clarity_primary"], scores["clarity_comparison"]
            ),
            "usefulness_score": max(
                scores["usefulness_primary"], scores["usefulness_comparison"]
            ),
            "overall_score": max(primary_overall, comparison_overall),
            "winner": winner,
            "reasoning": reasoning,
        }

    def _extract_section_winner(self, text: str, section: str) -> str:
        """Extract winner for a specific evaluation section."""
        # Look for patterns like "Winner: Response A" or "Winner: A" in the section
        section_patterns = [
            rf"{section}.*?winner:?\s*(response\s*)?([ab])",
            rf"{section}.*?:\s*response\s*([ab])",
            rf"winner.*?{section}.*?([ab])",
        ]

        text_lower = text.lower()
        for pattern in section_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                # Get the last group (which should be a or b)
                winner_letter = match.groups()[-1].lower()
                if winner_letter in ["a", "b"]:
                    return winner_letter

        return "tie"  # Default to tie if no clear winner found

    def _extract_reasoning(
        self, text: str, winner: str, primary_score: float, comparison_score: float
    ) -> str:
        """Extract or generate reasoning from the evaluation response."""
        # Look for "Overall Recommendation" or "Overall Assessment" section
        reasoning_patterns = [
            r"overall\s+recommendation:?\s*(.+?)(?:\n\n|\n#|\Z)",
            r"overall\s+assessment:?\s*(.+?)(?:\n\n|\n#|\Z)",
            r"summary\s+comparison:?\s*(.+?)(?:\n\n|\n#|\Z)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 20:  # Ensure we got meaningful content
                    return reasoning

        # Fallback to generated reasoning
        score_diff = abs(primary_score - comparison_score)
        if winner == "tie":
            return f"Both responses are very similar in quality (scores within {score_diff:.1f} points). The choice between them depends on specific user preferences and context."
        else:
            better_model = "primary" if winner == "primary" else "comparison"
            return f"The {better_model} model provided a superior response with a {score_diff:.1f} point advantage, showing better performance across the evaluated criteria."


# Global evaluator instance
_global_evaluator: ResponseEvaluator | None = None


def get_evaluator() -> ResponseEvaluator:
    """
    Get the global evaluator instance.

    Returns:
        Global ResponseEvaluator instance
    """
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = ResponseEvaluator()
    return _global_evaluator


def set_evaluator(evaluator: ResponseEvaluator) -> None:
    """
    Set the global evaluator instance.

    Args:
        evaluator: ResponseEvaluator instance to set as global
    """
    global _global_evaluator
    _global_evaluator = evaluator


def get_client_for_model(model: str):
    """
    Create a client for a specific model by detecting its provider.

    This function is used by tests and provides a convenient interface
    for creating clients based on model names.

    Args:
        model: Model name (e.g., "gpt-4", "claude-3-sonnet")

    Returns:
        Configured client instance for the model's provider

    Raises:
        EvaluationError: If client creation fails
    """
    try:
        provider = detect_model_provider(model)
        client = create_client_from_config(provider)
        return client
    except Exception as e:
        raise EvaluationError(
            f"Failed to create client for model '{model}': {str(e)}",
            model=model,
            cause=e,
        ) from e
