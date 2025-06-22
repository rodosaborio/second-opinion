"""
Compare Responses MCP tool implementation.

This module provides the `compare_responses` tool for detailed side-by-side
analysis of two AI responses across quality criteria, with cost analysis
and actionable recommendations for model selection.
"""

import logging
from decimal import Decimal
from typing import Any

from ...cli.main import filter_think_tags
from ...clients import detect_model_provider

# from ...config.model_configs import model_config_manager  # Not yet implemented
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, Message, ModelRequest, TaskComplexity
from ...orchestration import get_conversation_orchestrator
from ...orchestration.types import StorageContext
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import get_cost_guard
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)
from .shared import (
    get_model_name_suggestions,
    get_model_tier,
)

logger = logging.getLogger(__name__)


async def compare_responses_tool(
    response_a: str,
    response_b: str,
    task: str,
    model_a: str | None = None,
    model_b: str | None = None,
    cost_limit: float | None = None,
    session_id: str | None = None,
) -> str:
    """
    Compare two AI responses with detailed side-by-side analysis.

    This tool provides comprehensive comparison of two responses across multiple
    quality criteria, enabling informed decisions about model selection, response
    quality, and cost optimization. Perfect for A/B testing different models
    or comparing responses from different AI systems.

    USAGE PATTERN:
    1. User has two responses to the same task/question
    2. User wants detailed quality comparison and model recommendations
    3. Tool provides side-by-side analysis with scoring and insights
    4. User gets actionable recommendations for future model selection

    Args:
        response_a: The first response to compare
        response_b: The second response to compare
        task: The original task/question that generated both responses
        model_a: The model that generated response_a. Use OpenRouter format:
                 - Claude: "anthropic/claude-3-5-sonnet", "anthropic/claude-3-haiku"
                 - ChatGPT: "openai/gpt-4o", "openai/gpt-4o-mini"
                 - Gemini: "google/gemini-pro-1.5", "google/gemini-flash-1.5"
                 - Local: "qwen3-4b-mlx", "codestral-22b-v0.1"
        model_b: The model that generated response_b (same format as model_a)
        cost_limit: Maximum cost limit for evaluation in USD (default: $0.25)

    Returns:
        A detailed comparison report with:
        - Side-by-side response analysis
        - Quality scoring across multiple criteria (accuracy, completeness, clarity, usefulness)
        - Winner determination with detailed reasoning
        - Cost analysis and model tier comparison
        - Actionable recommendations for model selection
        - Use case specific guidance (budget vs premium scenarios)

    RECOMMENDED USAGE:
        # Compare responses from two different models
        result = await compare_responses_tool(
            response_a="Response from Model A...",
            response_b="Response from Model B...",
            task="Write a Python function to calculate fibonacci",
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o"
        )

        # Compare local vs cloud model responses
        result = await compare_responses_tool(
            response_a="Local model response...",
            response_b="Cloud model response...",
            task="Debug this code snippet",
            model_a="qwen3-4b-mlx",
            model_b="anthropic/claude-3-haiku"
        )

    COST OPTIMIZATION FOCUS:
        # Zero additional API cost when both responses are provided
        - Uses existing responses for analysis (no new generation needed)
        - Cost-efficient evaluation using budget evaluation models
        - Provides cost analysis to guide future model selection decisions
    """
    try:
        # Input validation and sanitization

        # Sanitize inputs with user context (most permissive for code/technical content)
        clean_response_a = sanitize_prompt(response_a, SecurityContext.USER_PROMPT)
        clean_response_b = sanitize_prompt(response_b, SecurityContext.USER_PROMPT)
        clean_task = sanitize_prompt(task, SecurityContext.USER_PROMPT)

        # Validate and normalize model names if provided
        if model_a:
            try:
                model_a = validate_model_name(model_a)
            except Exception as e:
                suggestions = get_model_name_suggestions(model_a, "general")
                suggestion_text = (
                    f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                )
                return f"❌ **Invalid Model A**: {str(e)}{suggestion_text}"

        if model_b:
            try:
                model_b = validate_model_name(model_b)
            except Exception as e:
                suggestions = get_model_name_suggestions(model_b, "general")
                suggestion_text = (
                    f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                )
                return f"❌ **Invalid Model B**: {str(e)}{suggestion_text}"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            # Get default from configuration
            try:
                # For now, use a default since tool-specific config isn't implemented yet
                cost_limit_decimal = Decimal("0.25")  # Default for comparison analysis
            except Exception:
                cost_limit_decimal = Decimal("0.25")

        logger.info(
            f"Starting compare_responses tool: response_a length={len(clean_response_a)}, "
            f"response_b length={len(clean_response_b)}, task length={len(clean_task)}"
        )
        logger.info(f"Model A: {model_a}, Model B: {model_b}")
        logger.info(f"Cost limit: ${cost_limit_decimal:.2f}")

        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()

        # Infer models if not provided (use generic placeholders for cost analysis)
        if not model_a:
            model_a = "unknown/model-a"
            logger.info("No model A specified, using placeholder for analysis")

        if not model_b:
            model_b = "unknown/model-b"
            logger.info("No model B specified, using placeholder for analysis")

        # Classify task complexity for evaluation criteria weighting
        task_complexity = TaskComplexity.MODERATE  # Default fallback
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_task)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")

        # Estimate cost (zero for response analysis, small cost for evaluation)
        estimated_cost = Decimal("0.0")  # No API calls needed for existing responses

        # Add evaluation cost (uses small evaluation model)
        try:
            evaluation_model = "openai/gpt-4o-mini"  # Cost-effective evaluation model
            eval_provider = detect_model_provider(evaluation_model)
            eval_client = create_client_from_config(eval_provider)
            eval_request = ModelRequest(
                model=evaluation_model,
                messages=[Message(role="user", content="comparison evaluation task")],
                max_tokens=500,  # Small evaluation request
                temperature=0.1,
                system_prompt="",
            )
            eval_cost = await eval_client.estimate_cost(eval_request)
            estimated_cost += eval_cost  # Only one evaluation needed
            logger.info(f"Evaluation cost estimate: ${eval_cost:.4f}")
        except Exception as e:
            logger.warning(f"Failed to estimate evaluation cost: {e}")
            estimated_cost += Decimal("0.01")  # Conservative fallback

        # Check budget
        try:
            primary_model_for_budget = (
                model_a if model_a != "unknown/model-a" else "comparison-analysis"
            )
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "compare_responses",
                primary_model_for_budget,
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Get cost estimates for both models (for cost analysis reporting)
        model_a_cost_estimate = await _estimate_model_cost(
            model_a, clean_task, len(clean_response_a.split())
        )
        model_b_cost_estimate = await _estimate_model_cost(
            model_b, clean_task, len(clean_response_b.split())
        )

        # Create ModelResponse objects for both responses

        model_a_response = _create_model_response(
            content=clean_response_a,
            model=model_a,
            task=clean_task,
            cost_estimate=model_a_cost_estimate,
        )

        model_b_response = _create_model_response(
            content=clean_response_b,
            model=model_b,
            task=clean_task,
            cost_estimate=model_b_cost_estimate,
        )

        # Perform detailed comparison evaluation
        actual_cost = Decimal("0.0")

        try:
            logger.info("Performing detailed comparison evaluation")
            evaluation_criteria = EvaluationCriteria(
                accuracy_weight=0.3,  # Balanced weights for comprehensive comparison
                completeness_weight=0.25,
                clarity_weight=0.25,
                usefulness_weight=0.2,
            )

            evaluator_model = (
                "openai/gpt-4o-mini"  # Use cost-effective model for evaluation
            )

            # Compare Response A vs Response B
            comparison_result = await evaluator.compare_responses(
                model_a_response,  # Response A as primary
                model_b_response,  # Response B as comparison
                original_task=clean_task,
                criteria=evaluation_criteria,
                evaluator_model=evaluator_model,
            )

            # Convert ComparisonResult to analysis format
            analysis_result = {
                "winner": comparison_result.winner,
                "overall_score": comparison_result.overall_score,
                "reasoning": comparison_result.reasoning,
                "criteria_scores": {
                    "accuracy": getattr(comparison_result, "accuracy_score", None),
                    "completeness": getattr(
                        comparison_result, "completeness_score", None
                    ),
                    "clarity": getattr(comparison_result, "clarity_score", None),
                    "usefulness": getattr(comparison_result, "usefulness_score", None),
                },
            }

            actual_cost += Decimal("0.01")  # Small evaluation cost
            logger.info("✓ Comparison evaluation completed successfully")

        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            # Create fallback analysis
            analysis_result = {
                "winner": "tie",
                "overall_score": 5.0,
                "reasoning": f"Evaluation unavailable due to system error: {str(e)}. Manual comparison recommended.",
                "criteria_scores": {
                    "accuracy": None,
                    "completeness": None,
                    "clarity": None,
                    "usefulness": None,
                },
            }

        # Record actual cost
        await cost_guard.record_actual_cost(
            reservation_id, actual_cost, primary_model_for_budget, "compare_responses"
        )
        logger.info(f"Total operation cost: ${actual_cost:.4f}")

        # Store conversation (optional, non-fatal if it fails)
        try:
            orchestrator = get_conversation_orchestrator()
            storage_context = StorageContext(
                interface_type="mcp",  # This tool is called from MCP
                tool_name="compare_responses",
                session_id=session_id,
                context=None,  # No additional context for this tool
                save_conversation=True,  # TODO: Make this configurable
            )

            # Prepare responses for storage (both responses as ModelResponse objects)
            all_responses = [model_a_response, model_b_response]

            await orchestrator.handle_interaction(
                prompt=clean_task,
                responses=all_responses,
                storage_context=storage_context,
                evaluation_result={
                    "comparison_result": comparison_result,
                    "cost_analysis": {
                        "actual_cost": float(actual_cost),
                        "response_a_cost": float(model_a_cost_estimate),
                        "response_b_cost": float(model_b_cost_estimate),
                    },
                },
            )
            logger.debug("Conversation storage completed successfully")
        except Exception as storage_error:
            # Storage failure is non-fatal - continue with normal tool execution
            logger.warning(f"Conversation storage failed (non-fatal): {storage_error}")

        # Generate detailed comparison report
        return await _format_comparison_report(
            task=clean_task,
            response_a=clean_response_a,
            response_b=clean_response_b,
            model_a=model_a,
            model_b=model_b,
            model_a_cost=model_a_cost_estimate,
            model_b_cost=model_b_cost_estimate,
            analysis_result=analysis_result,
            task_complexity=task_complexity,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
        )

    except Exception as e:
        logger.error(f"Unexpected error in compare_responses tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs for more details and try again with simpler parameters."


async def _estimate_model_cost(
    model: str, task: str, response_length_words: int
) -> Decimal:
    """Estimate the cost for a model to generate a response of given length."""
    if model.startswith("unknown/"):
        return Decimal("0.0")  # Can't estimate cost for unknown models

    try:
        provider = detect_model_provider(model)
        client = create_client_from_config(provider)
        request = ModelRequest(
            model=model,
            messages=[Message(role="user", content=task)],
            max_tokens=response_length_words * 2,  # Rough estimate
            temperature=0.1,
            system_prompt="",
        )
        return await client.estimate_cost(request)
    except Exception as e:
        logger.warning(f"Failed to estimate cost for {model}: {e}")
        # Fallback estimates based on model tier
        tier = get_model_tier(model)
        if tier == "local":
            return Decimal("0.0")
        elif tier == "budget":
            return Decimal("0.01")
        elif tier == "mid-tier":
            return Decimal("0.03")
        elif tier == "premium":
            return Decimal("0.05")
        else:
            return Decimal("0.02")  # Default estimate


def _create_model_response(content: str, model: str, task: str, cost_estimate: Decimal):
    """Create a ModelResponse object from response content and metadata."""
    from ...core.models import ModelResponse, TokenUsage

    # Estimate token usage based on text length
    estimated_input_tokens = int(len(task.split()) * 1.3)
    estimated_output_tokens = int(len(content.split()) * 1.3)
    total_tokens = estimated_input_tokens + estimated_output_tokens

    provider = (
        detect_model_provider(model) if not model.startswith("unknown/") else "unknown"
    )

    return ModelResponse(
        content=content,
        model=model,
        usage=TokenUsage(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            total_tokens=total_tokens,
        ),
        cost_estimate=cost_estimate,
        provider=provider,
    )


async def _format_comparison_report(
    task: str,
    response_a: str,
    response_b: str,
    model_a: str,
    model_b: str,
    model_a_cost: Decimal,
    model_b_cost: Decimal,
    analysis_result: dict[str, Any],
    task_complexity: TaskComplexity,
    actual_cost: Decimal,
    cost_limit: Decimal,
) -> str:
    """Format the detailed comparison report for MCP client display."""

    report = []

    # Header with comparison framing
    report.append("# ⚖️ Response Comparison: Side-by-Side Analysis")
    report.append("")

    # Task and comparison context
    report.append("## 📝 Comparison Context")
    report.append(f"**Task**: {task[:200]}{'...' if len(task) > 200 else ''}")
    report.append(f"**Task Complexity**: {task_complexity.value}")
    report.append(f"**Model A**: {model_a} (${model_a_cost:.4f})")
    report.append(f"**Model B**: {model_b} (${model_b_cost:.4f})")
    report.append("")

    # Overall winner and score
    winner = analysis_result.get("winner", "tie")
    overall_score = analysis_result.get("overall_score", 5.0)
    reasoning = analysis_result.get("reasoning", "No detailed analysis available")

    report.append("## 🏆 Overall Winner")

    if winner == "primary":
        report.append(f"**🥇 Model A ({model_a}) WINS**")
        report.append(f"**Quality Score**: {overall_score:.1f}/10")
    elif winner == "comparison":
        report.append(f"**🥇 Model B ({model_b}) WINS**")
        report.append(f"**Quality Score**: {overall_score:.1f}/10")
    else:
        report.append("**🤝 TIE / Close Competition**")
        report.append(f"**Quality Score**: {overall_score:.1f}/10 (very close)")

    report.append("")
    report.append(f"**Analysis**: {reasoning}")
    report.append("")

    # Detailed criteria breakdown
    criteria_scores = analysis_result.get("criteria_scores", {})
    if any(score is not None for score in criteria_scores.values()):
        report.append("## 📊 Quality Criteria Breakdown")

        criteria_names = {
            "accuracy": "Accuracy & Correctness",
            "completeness": "Completeness & Coverage",
            "clarity": "Clarity & Readability",
            "usefulness": "Usefulness & Practicality",
        }

        for criterion, score in criteria_scores.items():
            if score is not None:
                criterion_name = criteria_names.get(criterion, criterion.title())
                report.append(f"**{criterion_name}**: {score:.1f}/10")

        report.append("")

    # Side-by-side response comparison
    report.append("## 🔀 Side-by-Side Responses")

    # Response A
    report.append(f"### Response A: {model_a}")
    report.append("")
    clean_response_a = filter_think_tags(response_a)
    report.append(
        clean_response_a[:1000] + ("..." if len(clean_response_a) > 1000 else "")
    )
    report.append("")

    # Response B
    report.append(f"### Response B: {model_b}")
    report.append("")
    clean_response_b = filter_think_tags(response_b)
    report.append(
        clean_response_b[:1000] + ("..." if len(clean_response_b) > 1000 else "")
    )
    report.append("")

    # Cost and model tier analysis
    report.append("## 💰 Cost & Model Analysis")

    tier_a = (
        get_model_tier(model_a) if not model_a.startswith("unknown/") else "unknown"
    )
    tier_b = (
        get_model_tier(model_b) if not model_b.startswith("unknown/") else "unknown"
    )

    report.append(
        f"**Model A**: {model_a} ({tier_a.title()} Tier) - ${model_a_cost:.4f}"
    )
    report.append(
        f"**Model B**: {model_b} ({tier_b.title()} Tier) - ${model_b_cost:.4f}"
    )

    # Cost difference analysis
    if model_a_cost > 0 or model_b_cost > 0:
        cost_diff = abs(model_b_cost - model_a_cost)
        if cost_diff > Decimal("0.001"):  # Meaningful difference
            if model_a_cost > model_b_cost:
                savings_pct = (
                    (cost_diff / model_a_cost * 100) if model_a_cost > 0 else 0
                )
                report.append(
                    f"**Cost Difference**: Model B saves ${cost_diff:.4f} ({savings_pct:.0f}% cheaper)"
                )
            else:
                increase_pct = (
                    (cost_diff / model_b_cost * 100) if model_b_cost > 0 else 0
                )
                report.append(
                    f"**Cost Difference**: Model A saves ${cost_diff:.4f} ({increase_pct:.0f}% cheaper)"
                )
        else:
            report.append("**Cost Difference**: Similar cost (< $0.001 difference)")

    report.append(f"**Analysis Cost**: ${actual_cost:.4f}")
    report.append("")

    # Actionable recommendations
    report.append("## 🎯 Actionable Recommendations")

    # Determine best recommendation based on winner, cost, and tiers
    if winner == "primary":
        winning_model = model_a
    elif winner == "comparison":
        winning_model = model_b
    else:
        # For ties, recommend based on cost efficiency
        if model_a_cost <= model_b_cost:
            winning_model = model_a
        else:
            winning_model = model_b

    if overall_score >= 7.5:
        report.append(f"**✅ RECOMMENDED: {winning_model}**")
        report.append("Clear quality advantage with good cost efficiency.")
    elif overall_score >= 6.0:
        report.append(f"**💡 CONSIDER: {winning_model}**")
        report.append("Modest quality advantage - consider your specific priorities.")
    else:
        report.append("**🤔 CLOSE CALL: Choose based on your priorities**")
        if model_a_cost < model_b_cost:
            report.append(f"- **For cost efficiency**: {model_a}")
        elif model_b_cost < model_a_cost:
            report.append(f"- **For cost efficiency**: {model_b}")
        report.append("- **For quality**: Test both on your specific use cases")

    report.append("")

    # Use case specific guidance
    report.append("## 🚀 Next Steps")

    report.append("**Immediate Actions:**")
    report.append(f"1. **Use {winning_model}** for similar tasks")
    report.append("2. **Test on your specific use cases** to validate performance")
    report.append("3. **Monitor cost vs quality** trade-offs over time")

    if tier_a != tier_b:
        report.append("")
        report.append("**Strategic Considerations:**")
        if any(tier in ["local"] for tier in [tier_a, tier_b]):
            local_model = model_a if tier_a == "local" else model_b
            report.append(
                f"- **{local_model}** offers zero marginal cost for high-volume use"
            )
        if any(tier in ["premium"] for tier in [tier_a, tier_b]):
            premium_model = model_a if tier_a == "premium" else model_b
            report.append(
                f"- **{premium_model}** provides premium quality for critical tasks"
            )
        report.append(
            "- **Consider task-based model selection** (budget for drafts, premium for finals)"
        )

    report.append("")
    report.append("---")
    report.append(
        "*Response comparison complete - Choose wisely based on your priorities! ⚖️*"
    )

    return "\n".join(report)
