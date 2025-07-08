"""
Batch Comparison MCP tool implementation.

This module provides the `batch_comparison` tool for comparing multiple AI responses
to the same task, helping evaluate model performance patterns and identify the best
approach across different models.
"""

import logging
from decimal import Decimal
from typing import Any

from ...cli.main import filter_think_tags
from ...clients import detect_model_provider
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, Message, ModelRequest, TaskComplexity
from ...orchestration import get_conversation_orchestrator
from ...orchestration.types import StorageContext
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import BudgetPeriod, get_cost_guard
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)

logger = logging.getLogger(__name__)


async def batch_comparison_tool(
    task: str,
    responses: list[str] | None = None,
    models: list[str] | None = None,
    context: str | None = None,
    rank_by: str = "quality",
    max_models: int = 5,
    cost_limit: float | None = None,
    session_id: str | None = None,
) -> str:
    """
    Compare multiple AI responses to the same task and rank them by various criteria.

    This tool is designed for systematic model evaluation where you want to test multiple
    models against the same task and get a comprehensive ranking and analysis. Perfect for
    model selection, benchmarking, and understanding model strengths for specific use cases.

    USAGE PATTERNS:
    1. **Evaluate Existing Responses**: Compare responses you already have
    2. **Model Benchmarking**: Generate fresh responses from multiple models
    3. **Mixed Evaluation**: Some existing responses + generate from additional models

    Args:
        task: The task or question to evaluate responses for
        responses: List of existing responses to compare (optional).
                  If provided, these will be evaluated without additional API calls.
        models: List of models to use. Format:
               - Cloud models: "anthropic/claude-3-5-sonnet", "openai/gpt-4o", etc.
               - Local models: "qwen3-4b-mlx", "codestral-22b-v0.1", etc.
               If responses are provided, this should match the response order.
               If no responses provided, will generate responses from these models.
        context: Additional context about the task domain for better evaluation.
        rank_by: Ranking criteria - "quality", "cost", "speed", or "comprehensive"
        max_models: Maximum number of models to compare (default: 5, max: 10)
        cost_limit: Maximum cost for this operation in USD (default: $0.50)
        session_id: Session ID for conversation tracking

    Returns:
        Comprehensive comparison report with:
        - Ranked responses with detailed scoring
        - Model performance analysis
        - Cost-effectiveness recommendations
        - Best model suggestions for similar tasks

    USAGE EXAMPLES:

        # Compare existing responses from different models
        result = await batch_comparison_tool(
            task="Write a Python function to validate email addresses",
            responses=[
                "def validate_email(email): return '@' in email",
                "import re\ndef validate_email(email): return re.match(r'^[^@]+@[^@]+$', email)",
                "from email_validator import validate_email as ve\ndef validate_email(email): return ve(email)"
            ],
            models=["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet", "openai/gpt-4o"],
            rank_by="quality"
        )

        # Generate and compare fresh responses
        result = await batch_comparison_tool(
            task="Explain quantum computing to a 10-year-old",
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "google/gemini-pro-1.5"],
            context="educational content",
            rank_by="comprehensive"
        )

        # Cost-focused comparison with local models
        result = await batch_comparison_tool(
            task="Debug this JavaScript code",
            models=["qwen3-4b-mlx", "codestral-22b-v0.1", "anthropic/claude-3-haiku"],
            rank_by="cost",
            context="coding task"
        )
    """
    try:
        # Input validation and sanitization
        clean_task = sanitize_prompt(task, SecurityContext.USER_PROMPT)

        # Validate context if provided
        clean_context = None
        if context:
            clean_context = sanitize_prompt(context, SecurityContext.USER_PROMPT)

        # Validate max_models limit
        if max_models > 10:
            return "❌ **Invalid Max Models**: Maximum allowed is 10 models per comparison.\n\nThis limit prevents excessive costs and ensures meaningful comparisons."

        if max_models < 2:
            return "❌ **Invalid Max Models**: Minimum required is 2 models for comparison.\n\nPlease specify at least 2 models to compare."

        # Validate ranking criteria
        valid_rank_by = ["quality", "cost", "speed", "comprehensive"]
        if rank_by not in valid_rank_by:
            return f"❌ **Invalid Ranking Criteria**: '{rank_by}'\n\n**Valid options**: {', '.join(valid_rank_by)}"

        # Validate responses and models alignment
        if responses and models:
            if len(responses) != len(models):
                return f"❌ **Mismatched Input**: {len(responses)} responses provided but {len(models)} models specified.\n\nWhen providing existing responses, the number of responses must match the number of models."

        if responses and len(responses) > max_models:
            return f"❌ **Too Many Responses**: {len(responses)} responses provided but max_models is {max_models}.\n\nEither reduce responses or increase max_models limit."

        # Validate and normalize model names
        if models:
            try:
                models = [validate_model_name(model) for model in models[:max_models]]
            except Exception as e:
                suggestions = _get_model_name_suggestions(str(e))
                return f"❌ **Invalid Model Name**: {str(e)}\n\n**Suggested formats:**\n{suggestions}"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            cost_limit_decimal = Decimal("0.50")  # Higher default for batch operations

        logger.info(
            f"Starting batch_comparison: task length={len(clean_task)}, "
            f"responses_provided={responses is not None}, models={models}, "
            f"rank_by={rank_by}, max_models={max_models}"
        )

        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()

        # Classify task complexity
        task_complexity = TaskComplexity.MODERATE  # Default fallback
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_task)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")

        # Determine operation mode and setup
        if responses and models:
            # Mode 1: Evaluate existing responses
            operation_mode = "evaluate_existing"
            if len(responses) != len(models):
                return f"❌ **Input Mismatch**: {len(responses)} responses but {len(models)} models specified."

            # Truncate to max_models if needed
            responses = responses[:max_models]
            models = models[:max_models]

        elif models and not responses:
            # Mode 2: Generate fresh responses
            operation_mode = "generate_fresh"
            models = models[:max_models]
            responses = []  # Will be populated

        elif responses and not models:
            # Mode 3: Evaluate responses without model info
            operation_mode = "evaluate_anonymous"
            responses = responses[:max_models]
            models = [f"Model_{i + 1}" for i in range(len(responses))]

        else:
            # Mode 4: Auto-select models and generate
            operation_mode = "auto_generate"
            from ...cli.main import ComparisonModelSelector

            selector = ComparisonModelSelector()
            # Get a primary model first
            primary_model = "anthropic/claude-3-5-sonnet"

            models = selector.select_models(
                primary_model=primary_model,
                tool_name="batch_comparison",
                explicit_models=None,
                task_complexity=task_complexity,
                max_models=max_models,
            )
            # Include the primary model
            models = [primary_model] + models
            models = list(dict.fromkeys(models))[
                :max_models
            ]  # Remove duplicates, limit to max
            responses = []

        logger.info(f"Operation mode: {operation_mode}, Models: {models}")

        # Estimate costs
        estimated_cost = Decimal("0.0")

        if operation_mode in ["generate_fresh", "auto_generate"]:
            # Need to generate responses
            for model in models:
                try:
                    provider = detect_model_provider(model)
                    client = create_client_from_config(provider)
                    request = ModelRequest(
                        model=model,
                        messages=[Message(role="user", content=clean_task)],
                        max_tokens=2000,
                        temperature=0.1,
                        system_prompt=clean_context,
                    )
                    model_cost = await client.estimate_cost(request)
                    estimated_cost += model_cost
                except Exception as e:
                    logger.warning(f"Failed to estimate cost for {model}: {e}")
                    estimated_cost += Decimal("0.05")  # Conservative estimate

        # Add evaluation costs (pairwise comparisons)
        num_comparisons = len(models) * (len(models) - 1) // 2  # n choose 2
        evaluation_cost_per_comparison = Decimal("0.02")
        estimated_cost += evaluation_cost_per_comparison * num_comparisons

        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "batch_comparison",
                models[0] if models else "unknown",
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Generate responses if needed
        actual_cost = Decimal("0.0")
        model_responses = []  # Will store ModelResponse objects

        if operation_mode in ["generate_fresh", "auto_generate"]:
            logger.info("Generating fresh responses from models")
            for model in models:
                try:
                    provider = detect_model_provider(model)
                    client = create_client_from_config(provider)
                    request = ModelRequest(
                        model=model,
                        messages=[Message(role="user", content=clean_task)],
                        max_tokens=2000,
                        temperature=0.1,
                        system_prompt=clean_context,
                    )
                    response = await client.complete(request)
                    model_responses.append(response)
                    responses.append(filter_think_tags(response.content))
                    actual_cost += response.cost_estimate
                    logger.info(
                        f"Generated response from {model}: ${response.cost_estimate:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to generate response from {model}: {e}")
                    # Create error response
                    from ...core.models import ModelResponse, TokenUsage

                    error_response = ModelResponse(
                        content=f"Error: Failed to generate response from {model}: {str(e)}",
                        model=model,
                        usage=TokenUsage(
                            input_tokens=0, output_tokens=0, total_tokens=0
                        ),
                        cost_estimate=Decimal("0.0"),
                        provider=detect_model_provider(model),
                    )
                    model_responses.append(error_response)
                    responses.append(error_response.content)
        else:
            # Convert existing responses to ModelResponse objects for consistency
            logger.info("Converting existing responses to evaluation format")
            from ...core.models import ModelResponse, TokenUsage

            for model, response_text in zip(models, responses, strict=False):
                # Estimate cost for existing responses (for reporting purposes)
                estimated_cost_for_response = Decimal("0.0")
                if operation_mode != "evaluate_anonymous":
                    try:
                        provider = detect_model_provider(model)
                        client = create_client_from_config(provider)
                        # Estimate based on response length
                        estimated_tokens = len(response_text.split()) * 1.3
                        request = ModelRequest(
                            model=model,
                            messages=[Message(role="user", content=clean_task)],
                            max_tokens=int(estimated_tokens),
                            temperature=0.1,
                        )
                        estimated_cost_for_response = await client.estimate_cost(
                            request
                        )
                    except Exception:
                        estimated_cost_for_response = Decimal("0.01")

                model_response = ModelResponse(
                    content=response_text,
                    model=model,
                    usage=TokenUsage(
                        input_tokens=int(len(clean_task.split()) * 1.3),
                        output_tokens=int(len(response_text.split()) * 1.3),
                        total_tokens=int(len(clean_task.split()) * 1.3)
                        + int(len(response_text.split()) * 1.3),
                    ),
                    cost_estimate=estimated_cost_for_response,
                    provider=detect_model_provider(model)
                    if operation_mode != "evaluate_anonymous"
                    else "unknown",
                )
                model_responses.append(model_response)
                actual_cost += estimated_cost_for_response

        # Perform comprehensive pairwise evaluation
        logger.info("Performing comprehensive pairwise evaluation")
        evaluation_results = []
        evaluation_cost = Decimal("0.0")

        try:
            evaluation_criteria = EvaluationCriteria(
                accuracy_weight=0.3,
                completeness_weight=0.25,
                clarity_weight=0.25,
                usefulness_weight=0.2,
            )

            # Use cost-effective model for evaluation
            evaluator_model = "openai/gpt-4o-mini"

            # Perform pairwise comparisons
            for i in range(len(model_responses)):
                for j in range(i + 1, len(model_responses)):
                    try:
                        response_a = model_responses[i]
                        response_b = model_responses[j]

                        # Skip error responses
                        if response_a.content.startswith(
                            "Error:"
                        ) or response_b.content.startswith("Error:"):
                            continue

                        comparison_result = await evaluator.compare_responses(
                            response_a,
                            response_b,
                            original_task=clean_task,
                            criteria=evaluation_criteria,
                            evaluator_model=evaluator_model,
                        )

                        evaluation_results.append(
                            {
                                "model_a": response_a.model,
                                "model_b": response_b.model,
                                "winner": comparison_result.winner,
                                "score_a": comparison_result.score_a,
                                "score_b": comparison_result.score_b,
                                "overall_score": comparison_result.overall_score,
                                "reasoning": comparison_result.reasoning,
                            }
                        )

                        evaluation_cost += Decimal("0.01")

                    except Exception as e:
                        logger.warning(
                            f"Evaluation failed for {response_a.model} vs {response_b.model}: {e}"
                        )
                        continue

            actual_cost += evaluation_cost

        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            evaluation_results = []

        # Calculate rankings based on criteria
        rankings = _calculate_batch_rankings(
            models=models,
            model_responses=model_responses,
            evaluation_results=evaluation_results,
            rank_by=rank_by,
        )

        # Record actual cost
        await cost_guard.record_actual_cost(
            reservation_id,
            actual_cost,
            models[0] if models else "unknown",
            "batch_comparison",
        )

        # Store conversation (optional, non-fatal)
        try:
            orchestrator = get_conversation_orchestrator()
            storage_context = StorageContext(
                interface_type="mcp",
                tool_name="batch_comparison",
                session_id=session_id,
                context=clean_context,
                save_conversation=True,
            )

            await orchestrator.handle_interaction(
                prompt=clean_task,
                responses=model_responses,
                storage_context=storage_context,
                evaluation_result={
                    "operation_mode": operation_mode,
                    "ranking_criteria": rank_by,
                    "evaluation_results": evaluation_results,
                    "rankings": rankings,
                    "task_complexity": task_complexity.value,
                    "cost_analysis": {
                        "actual_cost": float(actual_cost),
                        "cost_limit": float(cost_limit_decimal),
                        "model_costs": [
                            float(r.cost_estimate) for r in model_responses
                        ],
                    },
                },
            )
        except Exception as storage_error:
            logger.warning(f"Conversation storage failed (non-fatal): {storage_error}")

        # Generate comprehensive report
        return await _format_batch_comparison_report(
            task=clean_task,
            models=models,
            responses=responses,
            model_responses=model_responses,
            evaluation_results=evaluation_results,
            rankings=rankings,
            rank_by=rank_by,
            operation_mode=operation_mode,
            task_complexity=task_complexity,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
            context=clean_context,
        )

    except Exception as e:
        logger.error(f"Unexpected error in batch_comparison tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs and try again with simpler parameters."


def _calculate_batch_rankings(
    models: list[str],
    model_responses: list[Any],
    evaluation_results: list[dict[str, Any]],
    rank_by: str,
) -> list[dict[str, Any]]:
    """Calculate model rankings based on specified criteria."""
    rankings = []

    # Initialize scores for each model
    model_scores = {
        model: {
            "wins": 0,
            "total_comparisons": 0,
            "avg_score": 0.0,
            "cost": Decimal("0.0"),
        }
        for model in models
    }

    # Process evaluation results
    for result in evaluation_results:
        model_a = result["model_a"]
        model_b = result["model_b"]
        winner = result["winner"]

        model_scores[model_a]["total_comparisons"] += 1
        model_scores[model_b]["total_comparisons"] += 1

        if winner == "primary":  # model_a won
            model_scores[model_a]["wins"] += 1
        elif winner == "comparison":  # model_b won
            model_scores[model_b]["wins"] += 1

        # Track average scores
        model_scores[model_a]["avg_score"] += result["score_a"]
        model_scores[model_b]["avg_score"] += result["score_b"]

    # Calculate final metrics
    for model, response in zip(models, model_responses, strict=False):
        score_data = model_scores[model]

        # Win rate
        win_rate = (
            (score_data["wins"] / score_data["total_comparisons"]) * 100
            if score_data["total_comparisons"] > 0
            else 0
        )

        # Average score
        avg_score = (
            score_data["avg_score"] / score_data["total_comparisons"]
            if score_data["total_comparisons"] > 0
            else 5.0
        )

        # Cost per response
        cost = response.cost_estimate

        # Calculate ranking score based on criteria
        if rank_by == "quality":
            ranking_score = (win_rate * 0.6) + (
                avg_score * 4
            )  # Scale avg_score from 0-10 to 0-40
        elif rank_by == "cost":
            # Invert cost (lower cost = higher ranking)
            max_cost = (
                max(r.cost_estimate for r in model_responses)
                if model_responses
                else Decimal("1.0")
            )
            cost_score = float(1 - (cost / max_cost)) * 100 if max_cost > 0 else 100
            ranking_score = cost_score
        elif rank_by == "speed":
            # For now, use a simple heuristic based on model type
            provider = detect_model_provider(model)
            if provider == "lmstudio":
                speed_score = 90  # Local models are fast
            elif "mini" in model.lower() or "flash" in model.lower():
                speed_score = 80  # Smaller models are faster
            else:
                speed_score = 60  # Default for larger models
            ranking_score = speed_score
        else:  # comprehensive
            # Balanced scoring: quality + cost efficiency + speed estimation
            quality_component = (win_rate * 0.4) + (avg_score * 3)
            max_cost = (
                max(r.cost_estimate for r in model_responses)
                if model_responses
                else Decimal("1.0")
            )
            cost_component = float(1 - (cost / max_cost)) * 30 if max_cost > 0 else 0
            # Speed component (simple heuristic)
            provider = detect_model_provider(model)
            speed_component = (
                20 if provider == "lmstudio" else 15 if "mini" in model.lower() else 10
            )

            ranking_score = quality_component + cost_component + speed_component

        rankings.append(
            {
                "model": model,
                "ranking_score": ranking_score,
                "win_rate": win_rate,
                "avg_score": avg_score,
                "cost": cost,
                "response_length": len(response.content),
                "provider": detect_model_provider(model),
            }
        )

    # Sort by ranking score (highest first)
    rankings.sort(key=lambda x: x["ranking_score"], reverse=True)

    # Add rank positions
    for i, ranking in enumerate(rankings):
        ranking["rank"] = i + 1

    return rankings


async def _format_batch_comparison_report(
    task: str,
    models: list[str],
    responses: list[str],
    model_responses: list[Any],
    evaluation_results: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    rank_by: str,
    operation_mode: str,
    task_complexity: Any,
    actual_cost: Decimal,
    cost_limit: Decimal,
    context: str | None,
) -> str:
    """Format comprehensive batch comparison report."""

    report = []

    # Header
    report.append("# 🏆 Batch Model Comparison Results")
    report.append("")

    # Task summary
    report.append("## 📝 Task Summary")
    report.append(f"**Task**: {task[:300]}{'...' if len(task) > 300 else ''}")
    if context:
        report.append(f"**Context**: {context}")
    report.append(f"**Task Complexity**: {task_complexity.value}")
    report.append(f"**Models Compared**: {len(models)}")
    report.append(f"**Ranking Criteria**: {rank_by.title()}")
    report.append(f"**Operation Mode**: {operation_mode.replace('_', ' ').title()}")
    report.append("")

    # Cost analysis
    report.append("## 💰 Cost Analysis")
    report.append(f"**Total Cost**: ${actual_cost:.4f}")
    report.append(f"**Cost Limit**: ${cost_limit:.2f}")
    report.append(f"**Average Cost per Model**: ${actual_cost / len(models):.4f}")

    # Get budget info
    try:
        cost_guard = get_cost_guard()
        budget_usage = await cost_guard.get_usage_summary(BudgetPeriod.DAILY)
        report.append(f"**Daily Budget Remaining**: ${budget_usage.available:.2f}")
    except Exception:
        report.append("**Daily Budget**: Not available")

    report.append("")

    # Rankings
    report.append(f"## 🏆 Model Rankings (by {rank_by.title()})")
    report.append("")

    for ranking in rankings:
        rank = ranking["rank"]
        model = ranking["model"]
        score = ranking["ranking_score"]
        win_rate = ranking["win_rate"]
        avg_score = ranking["avg_score"]
        cost = ranking["cost"]
        provider = ranking["provider"]

        # Medal emoji for top 3
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}."

        report.append(f"### {medal} {model}")
        report.append(f"**Provider**: {provider.title()}")
        report.append(f"**Ranking Score**: {score:.1f}")
        report.append(f"**Win Rate**: {win_rate:.1f}%")
        report.append(f"**Average Quality**: {avg_score:.1f}/10")
        report.append(f"**Cost**: ${cost:.4f}")
        report.append("")

    # Detailed responses
    report.append("## 📋 Complete Responses")
    for model, response in zip(models, responses, strict=False):
        ranking = next(r for r in rankings if r["model"] == model)
        rank = ranking["rank"]
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"

        report.append(f"### {medal} {model}")
        report.append("")
        if response.startswith("Error:"):
            report.append(f"❌ {response}")
        else:
            # Truncate very long responses
            max_length = 1200
            if len(response) > max_length:
                report.append(f"{response[:max_length]}")
                report.append("")
                report.append(
                    f"*[Response truncated - showing first {max_length} characters]*"
                )
            else:
                report.append(response)
        report.append("")

    # Performance insights
    report.append("## 📊 Performance Insights")

    # Best model analysis
    if rankings:
        best_model = rankings[0]
        report.append(f"**🎯 Winner**: {best_model['model']}")
        report.append(f"- **Strength**: {_get_model_strength(best_model, rank_by)}")
        report.append(f"- **Win Rate**: {best_model['win_rate']:.1f}%")
        report.append(f"- **Quality Score**: {best_model['avg_score']:.1f}/10")
        report.append("")

    # Cost efficiency analysis
    if len(rankings) > 1:
        most_cost_efficient = min(rankings, key=lambda x: x["cost"])
        if most_cost_efficient != rankings[0]:
            report.append(f"**💰 Most Cost-Efficient**: {most_cost_efficient['model']}")
            report.append(f"- **Cost**: ${most_cost_efficient['cost']:.4f}")
            report.append(f"- **Rank**: #{most_cost_efficient['rank']}")
            report.append("")

    # Local vs Cloud analysis
    local_models = [r for r in rankings if r["provider"] == "lmstudio"]
    cloud_models = [r for r in rankings if r["provider"] == "openrouter"]

    if local_models and cloud_models:
        best_local = local_models[0]
        best_cloud = cloud_models[0]

        report.append("**🔄 Local vs Cloud Comparison**:")
        report.append(
            f"- **Best Local**: {best_local['model']} (#{best_local['rank']}, ${best_local['cost']:.4f})"
        )
        report.append(
            f"- **Best Cloud**: {best_cloud['model']} (#{best_cloud['rank']}, ${best_cloud['cost']:.4f})"
        )

        if best_local["rank"] < best_cloud["rank"]:
            report.append(
                "- **💡 Insight**: Local model outperformed cloud alternatives!"
            )
        else:
            savings = best_cloud["cost"] - best_local["cost"]
            report.append(
                f"- **💡 Insight**: Cloud quality advantage costs ${savings:.4f} extra"
            )
        report.append("")

    # Recommendations
    report.append("## 🎯 Recommendations")

    if rankings:
        top_model = rankings[0]

        if rank_by == "quality":
            report.append(f"**For Best Quality**: Use **{top_model['model']}**")
            report.append(
                f"- Consistently wins {top_model['win_rate']:.1f}% of comparisons"
            )
            report.append(f"- Average quality score: {top_model['avg_score']:.1f}/10")
        elif rank_by == "cost":
            report.append(f"**For Cost Optimization**: Use **{top_model['model']}**")
            report.append(
                f"- Most cost-effective at ${top_model['cost']:.4f} per request"
            )
            if top_model["provider"] == "lmstudio":
                report.append("- **Bonus**: Zero marginal cost for local inference!")
        elif rank_by == "comprehensive":
            report.append(f"**Best Overall Choice**: **{top_model['model']}**")
            report.append("- Optimal balance of quality, cost, and speed")
            report.append(f"- Comprehensive score: {top_model['ranking_score']:.1f}")

        # Alternative suggestions
        if len(rankings) > 1:
            runner_up = rankings[1]
            report.append("")
            report.append(f"**Alternative**: {runner_up['model']}")
            report.append(
                f"- Close second choice with {runner_up['ranking_score']:.1f} score"
            )

            if top_model["cost"] > runner_up["cost"]:
                savings = top_model["cost"] - runner_up["cost"]
                report.append(
                    f"- **Cost savings**: ${savings:.4f} per request vs winner"
                )

    report.append("")

    # Next steps
    report.append("## 🚀 Next Steps")
    if rankings:
        report.append(
            f"1. **Adopt** {rankings[0]['model']} for similar {task_complexity.value.lower()} tasks"
        )
        report.append(
            "2. **Test** the winning model on a few more examples to validate consistency"
        )

        if local_models:
            best_local = local_models[0]
            report.append(
                f"3. **Consider** {best_local['model']} for cost-free development iterations"
            )

        report.append(
            "4. **Document** the performance patterns for future model selection"
        )
    else:
        report.append("1. **Review** any error messages in the responses above")
        report.append("2. **Try again** with different models or simplified task")
        report.append("3. **Check** model availability and cost limits")

    report.append("")
    report.append("---")
    report.append("*Batch Comparison Complete - Choose your champion! 🏆*")

    return "\n".join(report)


def _get_model_strength(ranking: dict[str, Any], rank_by: str) -> str:
    """Get the primary strength description for a model based on ranking criteria."""
    if rank_by == "quality":
        return f"Highest quality responses ({ranking['avg_score']:.1f}/10 average)"
    elif rank_by == "cost":
        return f"Most cost-effective (${ranking['cost']:.4f} per response)"
    elif rank_by == "speed":
        return "Fastest response generation"
    else:  # comprehensive
        return "Best overall balance of quality, cost, and speed"


def _get_model_name_suggestions(invalid_model: str) -> str:
    """Generate helpful model name suggestions for batch comparison."""
    suggestions = []

    suggestions.append("**Cloud Models (OpenRouter format):**")
    suggestions.append("- `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`")
    suggestions.append("- `openai/gpt-4o`, `openai/gpt-4o-mini`")
    suggestions.append("- `google/gemini-pro-1.5`, `google/gemini-flash-1.5`")
    suggestions.append("")
    suggestions.append("**Local Models:**")
    suggestions.append("- `qwen3-4b-mlx`, `codestral-22b-v0.1`")
    suggestions.append("")
    suggestions.append("**Example model list for batch comparison:**")
    suggestions.append("```")
    suggestions.append(
        '["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"]'
    )
    suggestions.append("```")

    return "\n".join(suggestions)
