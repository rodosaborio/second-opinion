"""
Should Upgrade MCP tool implementation.

This module provides the `should_upgrade` tool for analyzing whether premium
model alternatives could provide quality improvements that justify additional
cost, with focus on quality enhancement through better models.
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
from .shared import (
    calculate_quality_assessment,
    format_cost_comparison,
    get_cross_provider_alternatives,
    get_model_name_suggestions,
    get_model_tier,
    should_recommend_change,
    validate_model_candidates,
)

logger = logging.getLogger(__name__)


async def should_upgrade_tool(
    current_response: str,
    task: str,
    current_model: str | None = None,
    upgrade_candidates: list[str] | None = None,
    include_premium: bool = True,
    cost_limit: float | None = None,
) -> str:
    """
    Analyze whether premium model alternatives could provide quality improvements.

    This tool helps optimize AI quality by testing if premium models can provide
    significant quality improvements that justify additional cost. It focuses on
    quality enhancement while providing transparent cost analysis.

    USAGE PATTERN:
    1. User has a response from a budget/mid-tier model
    2. User wants to know if premium alternatives would be better
    3. Tool tests premium models and cross-provider alternatives
    4. Provides specific upgrade recommendations with quality vs cost analysis

    Args:
        current_response: The response to analyze for potential quality improvements
        task: The original task/question that generated the response
        current_model: The model that generated the current response. Use OpenRouter format:
                      - Budget: "anthropic/claude-3-haiku", "openai/gpt-4o-mini"
                      - Mid-tier: "anthropic/claude-3-5-sonnet", "google/gemini-pro"
                      - Local: "qwen3-4b-mlx", "codestral-22b-v0.1"
        upgrade_candidates: Specific premium models to test instead of auto-selection.
                           Use OpenRouter format for cloud models.
                           Examples: ["anthropic/claude-3-opus", "openai/gpt-4o"]
                           If None, will auto-select based on current model and complexity.
        include_premium: Whether to include premium models in testing (recommended: True)
        cost_limit: Maximum cost limit for testing in USD (default: $0.50)

    Returns:
        A quality enhancement report with:
        - Quality comparison between current and premium alternatives
        - Specific upgrade recommendations with cost vs quality analysis
        - Premium model alternatives for maximum quality improvement
        - ROI assessment for quality vs cost trade-offs
        - Actionable quality optimization steps

    RECOMMENDED USAGE:
        # Test if budget model can be upgraded (auto-selection)
        result = await should_upgrade_tool(
            current_response="<response from budget model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-haiku",
            include_premium=True
        )

        # Test specific premium models (custom selection)
        result = await should_upgrade_tool(
            current_response="<response from budget model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-haiku",
            upgrade_candidates=["anthropic/claude-3-opus", "openai/gpt-4o"],
            include_premium=True
        )

    QUALITY OPTIMIZATION FOCUS:
        # Prioritizes testing these premium alternatives:
        - Premium cloud: anthropic/claude-3-opus, openai/gpt-4o, google/gemini-pro-1.5
        - Mid-tier upgrades: anthropic/claude-3-5-sonnet, google/gemini-pro
        - Cross-provider options for diverse perspective
    """
    try:
        # Input validation and sanitization

        # Sanitize inputs with user context (most permissive for code/technical content)
        clean_current_response = sanitize_prompt(current_response, SecurityContext.USER_PROMPT)
        clean_task = sanitize_prompt(task, SecurityContext.USER_PROMPT)

        # Validate and normalize model name if provided
        if current_model:
            try:
                current_model = validate_model_name(current_model)
            except Exception as e:
                suggestions = get_model_name_suggestions(current_model, "upgrade")
                suggestion_text = f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                return f"❌ **Invalid Current Model**: {str(e)}{suggestion_text}"

        # Validate upgrade candidates if provided
        try:
            validated_upgrade_candidates = validate_model_candidates(upgrade_candidates, "upgrade")
        except Exception as e:
            return f"❌ **Invalid Upgrade Candidates**: {str(e)}"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            # Get default from configuration (higher default for upgrade testing)
            try:
                # For now, use a default since tool-specific config isn't implemented yet
                cost_limit_decimal = Decimal("0.50")  # Higher default for upgrade testing
            except Exception:
                # Use higher default for upgrade testing
                cost_limit_decimal = Decimal("0.50")

        logger.info(f"Starting should_upgrade tool: response length={len(clean_current_response)}, "
                   f"current_model={current_model}, include_premium={include_premium}")
        logger.info(f"Custom upgrade candidates: {validated_upgrade_candidates}")
        logger.info(f"Cost limit: ${cost_limit_decimal:.2f}")

        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()

        # Infer current model if not provided
        if not current_model:
            # Default to a common budget model for upgrade baseline
            current_model = "anthropic/claude-3-haiku"
            logger.info(f"No current model specified, using baseline: {current_model}")

        # Classify task complexity for better model selection
        task_complexity = TaskComplexity.MODERATE  # Default fallback
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_task)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")

        # Select premium alternative models for testing
        if validated_upgrade_candidates and len(validated_upgrade_candidates) > 0:
            # Use user-specified models for testing
            upgrade_candidates = validated_upgrade_candidates
            logger.info(f"Using custom upgrade candidates: {upgrade_candidates}")
        else:
            # Auto-select based on current model and task complexity
            upgrade_candidates = _select_upgrade_candidates(
                current_model=current_model,
                include_premium=include_premium,
                task_complexity=task_complexity
            )
            logger.info(f"Auto-selected upgrade candidates: {upgrade_candidates}")

        # Estimate total cost for testing
        estimated_cost = Decimal("0.0")

        # Estimate cost for each upgrade candidate
        for model in upgrade_candidates:
            try:
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_task)],
                    max_tokens=len(clean_current_response.split()) * 2,  # Estimate based on current response
                    temperature=0.1,
                    system_prompt=""
                )
                model_cost = await client.estimate_cost(request)
                estimated_cost += model_cost
                logger.info(f"Upgrade candidate {model} cost estimate: ${model_cost:.4f}")
            except Exception as e:
                logger.warning(f"Failed to estimate cost for {model}: {e}")
                # Premium models get conservative estimate
                estimated_cost += Decimal("0.05")  # Conservative premium model estimate

        # Add evaluation cost (small model for comparison)
        evaluation_cost = Decimal("0.01") * len(upgrade_candidates)
        estimated_cost += evaluation_cost

        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost, "should_upgrade", current_model, per_request_override=cost_limit_decimal
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Generate responses from upgrade candidates
        actual_cost = Decimal("0.0")
        upgrade_responses = []  # Will store ModelResponse objects
        upgrade_costs = []

        for model in upgrade_candidates:
            try:
                logger.info(f"Generating upgrade response with {model}")
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_task)],
                    max_tokens=len(clean_current_response.split()) * 2,
                    temperature=0.1,
                    system_prompt=""
                )
                response = await client.complete(request)
                upgrade_responses.append(response)
                upgrade_costs.append(response.cost_estimate)
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
                    provider=detect_model_provider(model)
                )
                upgrade_responses.append(error_response)
                upgrade_costs.append(Decimal("0.0"))

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
                system_prompt=""
            )
            current_cost_estimate = await current_client.estimate_cost(current_request)
        except Exception:
            # Get cost estimate based on model tier
            current_tier = get_model_tier(current_model)
            if current_tier == "local":
                current_cost_estimate = Decimal("0.0")
            elif current_tier == "budget":
                current_cost_estimate = Decimal("0.01")
            else:
                current_cost_estimate = Decimal("0.03")

        from ...core.models import ModelResponse, TokenUsage
        estimated_input_tokens = int(len(clean_task.split()) * 1.3)
        estimated_output_tokens = int(len(clean_current_response.split()) * 1.3)
        total_tokens = estimated_input_tokens + estimated_output_tokens
        current_model_response = ModelResponse(
            content=clean_current_response,
            model=current_model,
            usage=TokenUsage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                total_tokens=total_tokens
            ),
            cost_estimate=current_cost_estimate,
            provider=current_provider
        )

        # Compare each upgrade candidate against current response
        try:
            logger.info("Performing upgrade evaluation")
            evaluation_criteria = EvaluationCriteria(
                accuracy_weight=0.3,     # Balanced weights for comprehensive quality assessment
                completeness_weight=0.3,
                clarity_weight=0.2,
                usefulness_weight=0.2
            )

            evaluator_model = "openai/gpt-4o-mini"  # Use cost-effective model for evaluation

            for response in upgrade_responses:
                try:
                    if response.content.startswith("Error:"):
                        fallback_result = {
                            'overall_winner': 'current',
                            'overall_score': 0.0,
                            'reasoning': f"Upgrade candidate {response.model} failed to respond."
                        }
                        evaluation_results.append((response.model, fallback_result))
                        continue

                    # Compare upgrade candidate vs current
                    result = await evaluator.compare_responses(
                        response,  # upgrade candidate as primary
                        current_model_response,  # current as comparison
                        original_task=clean_task,
                        criteria=evaluation_criteria,
                        evaluator_model=evaluator_model
                    )

                    # Convert to dict and adjust for upgrade context
                    result_dict = {
                        'overall_winner': 'upgrade' if result.winner == 'primary' else 'current',
                        'overall_score': result.overall_score,
                        'reasoning': result.reasoning,
                        'quality_improvement': calculate_quality_assessment(result.overall_score, "upgrade")
                    }
                    evaluation_results.append((response.model, result_dict))
                    evaluation_cost_actual += Decimal("0.005")  # Small evaluation cost

                except Exception as e:
                    logger.warning(f"Evaluation failed for {response.model}: {e}")
                    fallback_result = {
                        'overall_winner': 'current',
                        'overall_score': 5.0,
                        'reasoning': f"Evaluation failed for {response.model}: {str(e)}",
                        'quality_improvement': 'unknown'
                    }
                    evaluation_results.append((response.model, fallback_result))

            actual_cost += evaluation_cost_actual

        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            evaluation_results = [
                (model, {
                    'overall_winner': 'current',
                    'overall_score': 5.0,
                    'reasoning': 'Evaluation unavailable',
                    'quality_improvement': 'unknown'
                }) for model in upgrade_candidates
            ]

        # Record actual cost
        await cost_guard.record_actual_cost(reservation_id, actual_cost, current_model, "should_upgrade")
        logger.info(f"Total operation cost: ${actual_cost:.4f}")

        # Generate quality enhancement report
        return await _format_upgrade_report(
            task=clean_task,
            current_response=clean_current_response,
            current_model=current_model,
            current_cost_estimate=current_cost_estimate,
            upgrade_candidates=upgrade_candidates,
            upgrade_responses=[filter_think_tags(r.content) for r in upgrade_responses],
            upgrade_costs=upgrade_costs,
            evaluation_results=evaluation_results,
            task_complexity=task_complexity,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
            include_premium=include_premium
        )

    except Exception as e:
        logger.error(f"Unexpected error in should_upgrade tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs for more details and try again with simpler parameters."


def _select_upgrade_candidates(
    current_model: str,
    include_premium: bool,
    task_complexity: TaskComplexity,
    max_candidates: int = 3
) -> list[str]:
    """
    Select premium alternative models for upgrade testing.

    Prioritizes premium models for maximum quality improvement, then cross-provider alternatives.
    """
    candidates = []
    current_tier = get_model_tier(current_model)
    current_lower = current_model.lower()

    # Include premium models if requested
    if include_premium:
        if current_tier in ["local", "budget"]:
            # Major upgrade to premium tier
            candidates.extend([
                "anthropic/claude-3-opus",
                "openai/gpt-4o",
                "google/gemini-pro-1.5"
            ])
        elif current_tier == "mid-tier":
            # Upgrade to top premium models
            if "claude" not in current_lower:
                candidates.append("anthropic/claude-3-opus")
            if "gpt" not in current_lower and "openai" not in current_lower:
                candidates.append("openai/gpt-4o")
            if "gemini" not in current_lower:
                candidates.append("google/gemini-pro-1.5")

    # Add tier-appropriate upgrades based on current model
    if current_tier == "local":
        # Upgrade from local to any cloud model
        candidates.extend([
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o-mini",
            "google/gemini-pro"
        ])
    elif current_tier == "budget":
        # Upgrade from budget to mid-tier
        if "claude" not in current_lower:
            candidates.append("anthropic/claude-3-5-sonnet")
        if "gpt" not in current_lower and "openai" not in current_lower:
            candidates.append("openai/gpt-4o")
        if "gemini" not in current_lower:
            candidates.append("google/gemini-pro")

    # Add cross-provider alternatives
    cross_provider = get_cross_provider_alternatives(current_model, "premium")
    for option in cross_provider:
        if option not in candidates:
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


async def _format_upgrade_report(
    task: str,
    current_response: str,
    current_model: str,
    current_cost_estimate: Decimal,
    upgrade_candidates: list[str],
    upgrade_responses: list[str],
    upgrade_costs: list[Decimal],
    evaluation_results: list[tuple[str, dict[str, Any]]],
    task_complexity: TaskComplexity,
    actual_cost: Decimal,
    cost_limit: Decimal,
    include_premium: bool
) -> str:
    """Format the upgrade analysis report for MCP client display."""

    report = []

    # Header with quality enhancement framing
    report.append("# 🚀 Should You Upgrade? Quality Enhancement Analysis")
    report.append("")

    # Task and current model info
    report.append("## 📝 Current Situation")
    report.append(f"**Task**: {task[:200]}{'...' if len(task) > 200 else ''}")
    report.append(f"**Current Model**: {current_model}")
    report.append(f"**Current Cost per Request**: ${current_cost_estimate:.4f}")
    report.append(f"**Task Complexity**: {task_complexity.value}")
    report.append(f"**Testing Premium Models**: {'Yes' if include_premium else 'No'}")
    report.append("")

    # Current response preview
    report.append("## 🎯 Current Response Quality")
    report.append(f"**Model**: {current_model}")
    report.append("")
    report.append(current_response[:800] + ("..." if len(current_response) > 800 else ""))
    report.append("")

    # Upgrade candidates analysis
    report.append("## ⬆️ Premium Alternatives Tested")

    best_upgrade = None
    best_upgrade_score = 0.0
    premium_options = []

    for (model, response, cost, (_, eval_result)) in zip(
        upgrade_candidates, upgrade_responses, upgrade_costs, evaluation_results, strict=False
    ):
        tier = get_model_tier(model)
        is_premium = tier == "premium"

        if is_premium:
            premium_options.append(model)

        score = eval_result.get('overall_score', 0.0)
        quality_improvement = eval_result.get('quality_improvement', 'unknown')
        winner = eval_result.get('overall_winner', 'current')
        reasoning = eval_result.get('reasoning', 'No analysis available')

        if score > best_upgrade_score and winner == 'upgrade':
            best_upgrade = model
            best_upgrade_score = score

        # Cost comparison
        cost_diff, cost_percentage, cost_description = format_cost_comparison(
            current_cost_estimate, cost, is_upgrade=True
        )

        report.append(f"### {model}")
        report.append(f"**Cost**: ${cost:.4f} | **Change**: {cost_description}")
        report.append(f"**Quality vs Current**: {score:.1f}/10 ({quality_improvement} improvement)")
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
    report.append("## 💰 Cost vs Quality Analysis")
    report.append(f"**Current Cost per Request**: ${current_cost_estimate:.4f}")

    if upgrade_costs:
        cheapest_upgrade_cost = min(upgrade_costs)
        most_expensive_cost = max(upgrade_costs)

        cheapest_diff, cheapest_pct, cheapest_desc = format_cost_comparison(
            current_cost_estimate, cheapest_upgrade_cost, is_upgrade=True
        )
        expensive_diff, expensive_pct, expensive_desc = format_cost_comparison(
            current_cost_estimate, most_expensive_cost, is_upgrade=True
        )

        report.append(f"**Cheapest Upgrade**: {cheapest_desc}")
        report.append(f"**Most Expensive Upgrade**: {expensive_desc}")

    report.append(f"**Testing Cost**: ${actual_cost:.4f}")

    # Monthly cost projection
    report.append("")
    report.append("**💡 Potential Monthly Costs (100 requests):**")
    monthly_current = current_cost_estimate * 100
    report.append(f"- Current: ${monthly_current:.2f}")

    if best_upgrade:
        best_upgrade_cost = next(c for m, c in zip(upgrade_candidates, upgrade_costs, strict=False) if m == best_upgrade)
        monthly_upgrade = best_upgrade_cost * 100
        monthly_diff = monthly_upgrade - monthly_current
        report.append(f"- With {best_upgrade}: ${monthly_upgrade:.2f} (+${monthly_diff:.2f}/month)")

    report.append("")

    # Main recommendation
    report.append("## 🎯 My Recommendation")

    should_upgrade, reasoning = should_recommend_change(
        best_upgrade_score if best_upgrade else 0,
        next(c for m, c in zip(upgrade_candidates, upgrade_costs, strict=False) if m == best_upgrade) - current_cost_estimate if best_upgrade else Decimal("0"),
        current_cost_estimate,
        is_upgrade=True,
        quality_threshold=7.0,
        cost_ratio_threshold=3.0
    )

    if should_upgrade and best_upgrade:
        # Good upgrade option found
        tier = get_model_tier(best_upgrade)
        cost_diff, cost_pct, cost_desc = format_cost_comparison(
            current_cost_estimate,
            next(c for m, c in zip(upgrade_candidates, upgrade_costs, strict=False) if m == best_upgrade),
            is_upgrade=True
        )

        report.append(f"**✅ UPGRADE to {best_upgrade}** ({tier.title()} Model)")
        report.append(f"Quality score: {best_upgrade_score:.1f}/10 with {cost_desc}")
        report.append("The quality improvement justifies the additional cost.")
    elif premium_options and any(eval_result.get('overall_score', 0) >= 6.5 for _, eval_result in evaluation_results):
        # Premium options with reasonable improvement
        report.append("**💡 CONSIDER premium models for critical tasks**")
        report.append("Premium models show quality improvements for complex tasks.")
        report.append(f"Keep {current_model} for routine work, upgrade for important tasks.")
    else:
        # No clear upgrade benefit
        report.append(f"**✅ KEEP {current_model}**")
        report.append("Premium alternatives don't provide sufficient quality improvement.")
        report.append("The current model appears well-suited for this task complexity.")

    report.append("")
    report.append(f"**Reasoning**: {reasoning}")
    report.append("")

    # Actionable next steps
    report.append("## 🚀 Next Steps")

    if should_upgrade and best_upgrade:
        tier = get_model_tier(best_upgrade)
        report.append(f"1. **Trial {best_upgrade}** for similar tasks")
        report.append("2. **Compare outputs** on your most important use cases")
        report.append("3. **Monitor quality vs cost** trade-offs")
        if tier == "premium":
            report.append("4. **Use selectively** for critical tasks to optimize ROI")
    else:
        report.append(f"1. **Continue using {current_model}** for this task type")
        report.append("2. **Re-evaluate** when task complexity increases")
        report.append("3. **Test premium models** for mission-critical work")

    if premium_options:
        report.append("4. **Consider task-based model selection** (budget for drafts, premium for finals)")

    report.append("")
    report.append("---")
    report.append("*Quality enhancement analysis complete - Invest in quality when it matters! 🚀*")

    return "\n".join(report)
