"""
Should Downgrade MCP tool implementation.

This module provides the `should_downgrade` tool for analyzing whether cheaper
model alternatives could achieve similar quality for a given response, with
focus on cost optimization through local models and budget cloud alternatives.
"""

import logging
from decimal import Decimal
from typing import Any

from ...cli.main import filter_think_tags
from ...clients import detect_model_provider

# from ...config.model_configs import model_config_manager  # Not yet implemented
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, Message, ModelRequest, TaskComplexity
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import get_cost_guard
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)

logger = logging.getLogger(__name__)


async def should_downgrade_tool(
    current_response: str,
    task: str,
    current_model: str | None = None,
    downgrade_candidates: list[str] | None = None,
    test_local: bool = True,
    cost_limit: float | None = None,
) -> str:
    """
    Analyze whether cheaper model alternatives could achieve similar quality.

    This tool helps optimize AI costs by testing if cheaper models (especially local ones)
    can provide similar quality responses to more expensive cloud models. It focuses on
    cost reduction while maintaining acceptable quality standards.

    USAGE PATTERN:
    1. User has a response from an expensive model
    2. User wants to know if cheaper alternatives could work
    3. Tool tests local models and budget cloud options
    4. Provides specific downgrade recommendations with cost savings

    Args:
        current_response: The response to analyze for potential cost savings
        task: The original task/question that generated the response
        current_model: The model that generated the current response. Use OpenRouter format:
                      - Claude: "anthropic/claude-3-5-sonnet", "anthropic/claude-3-opus"
                      - ChatGPT: "openai/gpt-4o", "openai/gpt-4"
                      - Gemini: "google/gemini-pro-1.5"
                      - Local: "qwen3-4b-mlx", "codestral-22b-v0.1"
        downgrade_candidates: Specific cheaper models to test instead of auto-selection.
                             Use OpenRouter format for cloud models or local model names.
                             Examples: ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"]
                             If None, will auto-select based on current model and complexity.
        test_local: Whether to include local models in the comparison (recommended: True)
        cost_limit: Maximum cost limit for testing in USD (default: $0.15)

    Returns:
        A cost optimization report with:
        - Quality comparison between current and cheaper alternatives
        - Specific downgrade recommendations with cost savings
        - Local model alternatives for maximum cost reduction
        - Risk assessment for quality degradation
        - Actionable cost optimization steps

    RECOMMENDED USAGE:
        # Test if expensive model can be downgraded (auto-selection)
        result = await should_downgrade_tool(
            current_response="<response from expensive model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True
        )

        # Test specific cheaper models (custom selection)
        result = await should_downgrade_tool(
            current_response="<response from expensive model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-5-sonnet",
            downgrade_candidates=["anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
            test_local=False
        )

    COST OPTIMIZATION FOCUS:
        # Prioritizes testing these cheaper alternatives:
        - Local models: qwen3-4b-mlx, codestral-22b-v0.1 ($0.00 cost)
        - Budget cloud: anthropic/claude-3-haiku, openai/gpt-4o-mini
        - Mid-tier options: google/gemini-flash-1.5
    """
    try:
        # Input validation and sanitization

        # Sanitize inputs with user context (most permissive for code/technical content)
        clean_current_response = sanitize_prompt(
            current_response, SecurityContext.USER_PROMPT
        )
        clean_task = sanitize_prompt(task, SecurityContext.USER_PROMPT)

        # Validate and normalize model name if provided
        if current_model:
            try:
                current_model = validate_model_name(current_model)
            except Exception as e:
                suggestions = _get_model_name_suggestions(current_model)
                suggestion_text = (
                    f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                )
                return f"❌ **Invalid Current Model**: {str(e)}{suggestion_text}"

        # Validate downgrade candidates if provided
        validated_downgrade_candidates = None
        if downgrade_candidates is not None and len(downgrade_candidates) > 0:
            validated_downgrade_candidates = []
            for i, model in enumerate(downgrade_candidates):
                try:
                    validated_model = validate_model_name(model)
                    validated_downgrade_candidates.append(validated_model)
                except Exception as e:
                    suggestions = _get_model_name_suggestions(model)
                    suggestion_text = (
                        f"\n\n**Suggested formats:**\n{suggestions}"
                        if suggestions
                        else ""
                    )
                    return f"❌ **Invalid Downgrade Candidate #{i + 1}** ({model}): {str(e)}{suggestion_text}"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            # Get default from configuration (lower default for downgrade testing)
            try:
                # For now, use a default since tool-specific config isn't implemented yet
                cost_limit_decimal = Decimal(
                    "0.15"
                )  # Lower default for downgrade testing
            except Exception:
                # Use lower default for downgrade testing
                cost_limit_decimal = Decimal("0.15")

        logger.info(
            f"Starting should_downgrade tool: response length={len(clean_current_response)}, "
            f"current_model={current_model}, test_local={test_local}"
        )
        logger.info(f"Custom downgrade candidates: {validated_downgrade_candidates}")
        logger.info(f"Cost limit: ${cost_limit_decimal:.2f}")

        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()

        # Infer current model if not provided
        if not current_model:
            # Default to a common expensive model for comparison baseline
            current_model = "anthropic/claude-3-5-sonnet"
            logger.info(f"No current model specified, using baseline: {current_model}")

        # Classify task complexity for better model selection
        task_complexity = TaskComplexity.MODERATE  # Default fallback
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_task)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")

        # Select cheaper alternative models for testing
        if validated_downgrade_candidates and len(validated_downgrade_candidates) > 0:
            # Use user-specified models for testing
            downgrade_candidates = validated_downgrade_candidates
            logger.info(f"Using custom downgrade candidates: {downgrade_candidates}")
        else:
            # Auto-select based on current model and task complexity
            downgrade_candidates = _select_downgrade_candidates(
                current_model=current_model,
                test_local=test_local,
                task_complexity=task_complexity,
            )
            logger.info(f"Auto-selected downgrade candidates: {downgrade_candidates}")

        # Estimate total cost for testing
        estimated_cost = Decimal("0.0")

        # Estimate cost for each downgrade candidate
        for model in downgrade_candidates:
            try:
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_task)],
                    max_tokens=len(clean_current_response.split())
                    * 2,  # Estimate based on current response
                    temperature=0.1,
                    system_prompt="",
                )
                model_cost = await client.estimate_cost(request)
                estimated_cost += model_cost
                logger.info(
                    f"Downgrade candidate {model} cost estimate: ${model_cost:.4f}"
                )
            except Exception as e:
                logger.warning(f"Failed to estimate cost for {model}: {e}")
                # Local models have zero cost, cloud models get conservative estimate
                if detect_model_provider(model) == "lmstudio":
                    estimated_cost += Decimal("0.0")
                else:
                    estimated_cost += Decimal(
                        "0.02"
                    )  # Conservative budget model estimate

        # Add evaluation cost (small model for comparison)
        evaluation_cost = Decimal("0.01") * len(downgrade_candidates)
        estimated_cost += evaluation_cost

        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "should_downgrade",
                current_model,
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Generate responses from downgrade candidates
        actual_cost = Decimal("0.0")
        downgrade_responses = []  # Will store ModelResponse objects
        downgrade_costs = []

        for model in downgrade_candidates:
            try:
                logger.info(f"Generating downgrade response with {model}")
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_task)],
                    max_tokens=len(clean_current_response.split()) * 2,
                    temperature=0.1,
                    system_prompt="",
                )
                response = await client.complete(request)
                downgrade_responses.append(response)
                downgrade_costs.append(response.cost_estimate)
                actual_cost += response.cost_estimate
            except Exception as e:
                logger.error(f"Failed to get response from {model}: {e}")
                # Create error response for consistency
                from ...core.models import ModelResponse, TokenUsage

                error_response = ModelResponse(
                    content=f"Error: Failed to get response from {model}: {str(e)}",
                    model=model,
                    usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    cost_estimate=Decimal("0.0"),
                    provider=detect_model_provider(model),
                )
                downgrade_responses.append(error_response)
                downgrade_costs.append(Decimal("0.0"))

        # Perform evaluation against current response
        evaluation_results: list[tuple[str, dict[str, Any]]] = []
        evaluation_cost_actual = Decimal("0.0")

        # Create ModelResponse object for current response to use in evaluation
        current_provider = detect_model_provider(current_model)
        try:
            current_client = create_client_from_config(current_provider)
            current_request = ModelRequest(
                model=current_model,
                messages=[Message(role="user", content=clean_task)],
                max_tokens=len(clean_current_response.split()) * 2,
                temperature=0.1,
                system_prompt="",
            )
            current_cost_estimate = await current_client.estimate_cost(current_request)
        except Exception:
            current_cost_estimate = Decimal(
                "0.05"
            )  # Conservative estimate for expensive models

        from ...core.models import ModelResponse, TokenUsage

        estimated_input_tokens = int(len(clean_task.split()) * 1.3)
        estimated_output_tokens = int(len(clean_current_response.split()) * 1.3)
        total_tokens = (
            estimated_input_tokens + estimated_output_tokens
        )  # Ensure exact sum
        current_model_response = ModelResponse(
            content=clean_current_response,
            model=current_model,
            usage=TokenUsage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                total_tokens=total_tokens,
            ),
            cost_estimate=current_cost_estimate,
            provider=current_provider,
        )

        # Compare each downgrade candidate against current response
        try:
            logger.info("Performing downgrade evaluation")
            evaluation_criteria = EvaluationCriteria(
                accuracy_weight=0.35,  # Higher weight on accuracy for downgrade decisions
                completeness_weight=0.25,
                clarity_weight=0.2,
                usefulness_weight=0.2,
            )

            evaluator_model = (
                "openai/gpt-4o-mini"  # Use cost-effective model for evaluation
            )

            for response in downgrade_responses:
                try:
                    if response.content.startswith("Error:"):
                        fallback_result = {
                            "overall_winner": "current",
                            "overall_score": 0.0,
                            "reasoning": f"Downgrade candidate {response.model} failed to respond.",
                        }
                        evaluation_results.append((response.model, fallback_result))
                        continue

                    # Compare downgrade candidate vs current (reverse order from second_opinion)
                    result = await evaluator.compare_responses(
                        response,  # downgrade candidate as primary
                        current_model_response,  # current as comparison
                        original_task=clean_task,
                        criteria=evaluation_criteria,
                        evaluator_model=evaluator_model,
                    )

                    # Convert to dict and adjust for downgrade context
                    result_dict = {
                        "overall_winner": (
                            "downgrade" if result.winner == "primary" else "current"
                        ),
                        "overall_score": result.overall_score,
                        "reasoning": result.reasoning,
                        "quality_drop": _calculate_quality_drop(result.overall_score),
                    }
                    evaluation_results.append((response.model, result_dict))
                    evaluation_cost_actual += Decimal("0.005")  # Small evaluation cost

                except Exception as e:
                    logger.warning(f"Evaluation failed for {response.model}: {e}")
                    fallback_result = {
                        "overall_winner": "current",
                        "overall_score": 5.0,
                        "reasoning": f"Evaluation failed for {response.model}: {str(e)}",
                        "quality_drop": "unknown",
                    }
                    evaluation_results.append((response.model, fallback_result))

            actual_cost += evaluation_cost_actual

        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            evaluation_results = [
                (
                    model,
                    {
                        "overall_winner": "current",
                        "overall_score": 5.0,
                        "reasoning": "Evaluation unavailable",
                        "quality_drop": "unknown",
                    },
                )
                for model in downgrade_candidates
            ]

        # Record actual cost
        await cost_guard.record_actual_cost(
            reservation_id, actual_cost, current_model, "should_downgrade"
        )
        logger.info(f"Total operation cost: ${actual_cost:.4f}")

        # Generate cost optimization report
        return await _format_downgrade_report(
            task=clean_task,
            current_response=clean_current_response,
            current_model=current_model,
            current_cost_estimate=current_cost_estimate,
            downgrade_candidates=downgrade_candidates,
            downgrade_responses=[
                filter_think_tags(r.content) for r in downgrade_responses
            ],
            downgrade_costs=downgrade_costs,
            evaluation_results=evaluation_results,
            task_complexity=task_complexity,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
            test_local=test_local,
        )

    except Exception as e:
        logger.error(f"Unexpected error in should_downgrade tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs for more details and try again with simpler parameters."


def _select_downgrade_candidates(
    current_model: str,
    test_local: bool,
    task_complexity: TaskComplexity,
    max_candidates: int = 3,
) -> list[str]:
    """
    Select cheaper alternative models for downgrade testing.

    Prioritizes local models for maximum cost savings, then budget cloud options.
    """
    candidates = []

    # Always include local models if requested (zero cost)
    if test_local:
        if task_complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
            candidates.extend(["qwen3-4b-mlx", "codestral-22b-v0.1"])
        else:
            # For complex tasks, include better local models
            candidates.extend(["qwen3-4b-mlx"])

    # Add budget cloud alternatives based on current model tier
    current_lower = current_model.lower()

    if "claude-3-5-sonnet" in current_lower or "claude-3-opus" in current_lower:
        # Downgrade from premium Claude
        candidates.extend(["anthropic/claude-3-haiku"])
    elif "gpt-4o" in current_lower and "mini" not in current_lower:
        # Downgrade from premium GPT-4
        candidates.extend(["openai/gpt-4o-mini"])
    elif "gemini-pro" in current_lower:
        # Downgrade from premium Gemini
        candidates.extend(["google/gemini-flash-1.5"])

    # Add cross-provider budget options if we have room
    if len(candidates) < max_candidates:
        budget_options = [
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
        ]
        for option in budget_options:
            if option not in candidates and option.lower() not in current_lower:
                candidates.append(option)
                if len(candidates) >= max_candidates:
                    break

    # Remove duplicates and limit
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
        if len(unique_candidates) >= max_candidates:
            break

    return unique_candidates


def _calculate_quality_drop(score: float) -> str:
    """Calculate quality drop assessment from comparison score."""
    if score >= 7.5:
        return "minimal"
    elif score >= 6.0:
        return "moderate"
    elif score >= 4.0:
        return "significant"
    else:
        return "major"


async def _format_downgrade_report(
    task: str,
    current_response: str,
    current_model: str,
    current_cost_estimate: Decimal,
    downgrade_candidates: list[str],
    downgrade_responses: list[str],
    downgrade_costs: list[Decimal],
    evaluation_results: list[tuple[str, dict[str, Any]]],
    task_complexity: TaskComplexity,
    actual_cost: Decimal,
    cost_limit: Decimal,
    test_local: bool,
) -> str:
    """Format the downgrade analysis report for MCP client display."""

    report = []

    # Header with cost optimization framing
    report.append("# 💰 Should You Downgrade? Cost Optimization Analysis")
    report.append("")

    # Task and current model info
    report.append("## 📝 Current Situation")
    report.append(f"**Task**: {task[:200]}{'...' if len(task) > 200 else ''}")
    report.append(f"**Current Model**: {current_model}")
    report.append(f"**Current Cost per Request**: ${current_cost_estimate:.4f}")
    report.append(f"**Task Complexity**: {task_complexity.value}")
    report.append(f"**Testing Local Models**: {'Yes' if test_local else 'No'}")
    report.append("")

    # Current response preview
    report.append("## 🎯 Current Response Quality")
    report.append(f"**Model**: {current_model}")
    report.append("")
    report.append(
        current_response[:800] + ("..." if len(current_response) > 800 else "")
    )
    report.append("")

    # Downgrade candidates analysis
    report.append("## 🔻 Cheaper Alternatives Tested")

    best_downgrade = None
    best_downgrade_score = 0.0
    local_options = []

    for model, response, cost, (_, eval_result) in zip(
        downgrade_candidates,
        downgrade_responses,
        downgrade_costs,
        evaluation_results,
        strict=False,
    ):
        provider = detect_model_provider(model)
        is_local = provider == "lmstudio"

        if is_local:
            local_options.append(model)

        score = eval_result.get("overall_score", 0.0)
        quality_drop = eval_result.get("quality_drop", "unknown")
        winner = eval_result.get("overall_winner", "current")
        reasoning = eval_result.get("reasoning", "No analysis available")

        if score > best_downgrade_score and winner != "current":
            best_downgrade = model
            best_downgrade_score = score

        # Cost savings calculation
        if is_local:
            cost_savings = current_cost_estimate  # 100% savings
            savings_percent_str = "100%"
        else:
            cost_savings = current_cost_estimate - cost
            savings_percent_str = (
                f"{(cost_savings / current_cost_estimate * 100):.0f}%"
                if current_cost_estimate > 0
                else "N/A"
            )

        report.append(f"### {model}")
        report.append(
            f"**Cost**: ${cost:.4f} ({'FREE' if is_local else f'${cost:.4f}'}) | **Savings**: {savings_percent_str}"
        )
        report.append(
            f"**Quality vs Current**: {score:.1f}/10 ({quality_drop} degradation)"
        )
        report.append(f"**Winner**: {winner}")
        report.append("")

        if not response.startswith("Error:"):
            report.append(response[:600] + ("..." if len(response) > 600 else ""))
        else:
            report.append(f"❌ {response}")

        report.append("")
        report.append(f"**Analysis**: {reasoning}")
        report.append("")

    # Cost analysis summary
    report.append("## 💰 Cost Savings Analysis")
    report.append(f"**Current Cost per Request**: ${current_cost_estimate:.4f}")
    if local_options:
        report.append("**Local Models Cost**: $0.00 (100% savings)")
    if downgrade_costs:
        cheapest_cloud_cost = min(
            [c for c in downgrade_costs if c > 0] or [Decimal("0")]
        )
        if cheapest_cloud_cost > 0:
            cloud_savings = current_cost_estimate - cheapest_cloud_cost
            cloud_savings_percent = (
                (cloud_savings / current_cost_estimate * 100)
                if current_cost_estimate > 0
                else 0
            )
            report.append(
                f"**Cheapest Cloud Alternative**: ${cheapest_cloud_cost:.4f} ({cloud_savings_percent:.0f}% savings)"
            )

    report.append(f"**Testing Cost**: ${actual_cost:.4f}")

    # Monthly savings projection
    report.append("")
    report.append("**💡 Potential Monthly Savings (100 requests):**")
    if local_options:
        monthly_current = current_cost_estimate * 100
        report.append(f"- Current: ${monthly_current:.2f}")
        report.append(f"- With Local Models: $0.00 (Save ${monthly_current:.2f}/month)")

    report.append("")

    # Main recommendation
    report.append("## 🎯 My Recommendation")

    if best_downgrade and best_downgrade_score >= 6.5:
        # Good downgrade option found
        provider = detect_model_provider(best_downgrade)
        if provider == "lmstudio":
            report.append(f"**✅ DOWNGRADE to {best_downgrade}** (Local Model)")
            report.append(
                f"Quality score: {best_downgrade_score:.1f}/10 with 100% cost savings"
            )
            report.append("This local model provides acceptable quality at zero cost.")
        else:
            cost_savings = current_cost_estimate - next(
                c
                for m, c in zip(downgrade_candidates, downgrade_costs, strict=False)
                if m == best_downgrade
            )
            savings_percent = (
                (cost_savings / current_cost_estimate * 100)
                if current_cost_estimate > 0
                else 0
            )
            report.append(f"**💡 CONSIDER downgrading to {best_downgrade}**")
            report.append(
                f"Quality score: {best_downgrade_score:.1f}/10 with {savings_percent:.0f}% cost savings"
            )
    elif local_options and any(
        eval_result.get("overall_score", 0) >= 5.0
        for _, eval_result in evaluation_results
    ):
        # Local options with reasonable quality
        report.append("**💰 CONSIDER local models for development/testing**")
        report.append(
            "Local models show reasonable quality for this task type at zero cost."
        )
        report.append(f"Keep {current_model} for production, use local for iteration.")
    else:
        # No good downgrade options
        report.append(f"**✅ KEEP {current_model}**")
        report.append("Cheaper alternatives show significant quality degradation.")
        report.append("The cost premium appears justified for this task complexity.")

    report.append("")

    # Actionable next steps
    report.append("## 🚀 Next Steps")

    if best_downgrade:
        provider = detect_model_provider(best_downgrade)
        if provider == "lmstudio":
            report.append(
                f"1. **Test {best_downgrade}** for similar tasks in development"
            )
            report.append("2. **Compare outputs** on your specific use cases")
            report.append(
                f"3. **Use for iteration**, keep {current_model} for final outputs"
            )
        else:
            report.append(f"1. **Trial {best_downgrade}** for similar tasks")
            report.append("2. **Monitor quality** vs cost trade-offs")
            report.append(
                "3. **Switch gradually** for tasks where quality is acceptable"
            )
    else:
        report.append(f"1. **Continue using {current_model}** for this task type")
        report.append("2. **Test local models** for simpler tasks")
        report.append("3. **Re-evaluate** as local models improve")

    if local_options:
        report.append("4. **Set up local inference** for maximum cost savings")

    report.append("")
    report.append("---")
    report.append(
        "*Cost optimization analysis complete - Every penny saved is a penny earned! 💰*"
    )

    return "\n".join(report)


def _get_model_name_suggestions(invalid_model: str) -> str:
    """Generate helpful model name suggestions for downgrade context."""
    suggestions = []

    suggestions.append("**Expensive Models (for comparison baseline):**")
    suggestions.append(
        "- Claude: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-opus`"
    )
    suggestions.append("- ChatGPT: `openai/gpt-4o`, `openai/gpt-4`")
    suggestions.append("- Gemini: `google/gemini-pro-1.5`")
    suggestions.append("")
    suggestions.append("**Budget Alternatives (cheaper cloud):**")
    suggestions.append("- Claude: `anthropic/claude-3-haiku`")
    suggestions.append("- ChatGPT: `openai/gpt-4o-mini`")
    suggestions.append("- Gemini: `google/gemini-flash-1.5`")
    suggestions.append("")
    suggestions.append("**Local Models (zero cost):**")
    suggestions.append("- Qwen: `qwen3-4b-mlx`, `qwen3-0.6b-mlx`")
    suggestions.append("- Codestral: `codestral-22b-v0.1`")

    return "\n".join(suggestions)
