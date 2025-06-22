"""
Shared utilities for MCP tools.

This module provides common functions and patterns used by both should_downgrade
and should_upgrade tools to reduce code duplication and ensure consistency.
"""

import logging
from decimal import Decimal

from ...clients import detect_model_provider
from ...utils.sanitization import validate_model_name

logger = logging.getLogger(__name__)


def get_model_name_suggestions(invalid_model: str, context: str = "general") -> str:
    """Generate helpful model name suggestions based on context."""
    suggestions = []

    if context == "downgrade":
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
    elif context == "upgrade":
        suggestions.append("**Budget Models (starting point):**")
        suggestions.append("- Claude: `anthropic/claude-3-haiku`")
        suggestions.append("- ChatGPT: `openai/gpt-4o-mini`")
        suggestions.append("- Gemini: `google/gemini-flash-1.5`")
        suggestions.append("- Local: `qwen3-4b-mlx`, `codestral-22b-v0.1`")
        suggestions.append("")
        suggestions.append("**Premium Alternatives (upgrade targets):**")
        suggestions.append(
            "- Claude: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-opus`"
        )
        suggestions.append("- ChatGPT: `openai/gpt-4o`, `openai/gpt-4`")
        suggestions.append("- Gemini: `google/gemini-pro-1.5`")
    else:
        suggestions.append("**Popular Models:**")
        suggestions.append(
            "- Claude: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`"
        )
        suggestions.append("- ChatGPT: `openai/gpt-4o`, `openai/gpt-4o-mini`")
        suggestions.append(
            "- Gemini: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`"
        )
        suggestions.append("- Local: `qwen3-4b-mlx`, `codestral-22b-v0.1`")

    return "\n".join(suggestions)


def validate_model_candidates(
    candidates: list[str] | None, context: str = "general"
) -> list[str] | None:
    """
    Validate a list of model candidates.

    Args:
        candidates: List of model names to validate
        context: Context for error suggestions ("downgrade", "upgrade", "general")

    Returns:
        List of validated model names or None if input was None

    Raises:
        Exception with helpful suggestions if validation fails
    """
    if candidates is None or len(candidates) == 0:
        return None

    validated_candidates = []
    for i, model in enumerate(candidates):
        try:
            validated_model = validate_model_name(model)
            validated_candidates.append(validated_model)
        except Exception as e:
            suggestions = get_model_name_suggestions(model, context)
            suggestion_text = (
                f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
            )
            raise Exception(
                f"Invalid model candidate #{i + 1} ({model}): {str(e)}{suggestion_text}"
            ) from e

    return validated_candidates


def calculate_quality_assessment(score: float, context: str = "general") -> str:
    """Calculate quality assessment from comparison score based on context."""
    if context == "downgrade":
        # For downgrade: focus on quality drop
        if score >= 7.5:
            return "minimal"
        elif score >= 6.0:
            return "moderate"
        elif score >= 4.0:
            return "significant"
        else:
            return "major"
    elif context == "upgrade":
        # For upgrade: focus on quality improvement
        if score >= 8.5:
            return "excellent"
        elif score >= 7.5:
            return "significant"
        elif score >= 6.5:
            return "moderate"
        elif score >= 5.5:
            return "minor"
        else:
            return "negligible"
    else:
        # General quality assessment
        if score >= 8.0:
            return "excellent"
        elif score >= 7.0:
            return "good"
        elif score >= 6.0:
            return "fair"
        elif score >= 4.0:
            return "poor"
        else:
            return "very poor"


def get_model_tier(model: str) -> str:
    """
    Determine the tier of a model for upgrade/downgrade logic.

    Returns:
        "local", "budget", "mid-tier", "premium", or "unknown"
    """
    model_lower = model.lower()
    provider = detect_model_provider(model)

    # Local models
    if provider == "lmstudio":
        return "local"

    # Budget tier
    if any(
        budget in model_lower
        for budget in ["haiku", "mini", "flash", "gemini-flash", "gpt-3.5"]
    ):
        return "budget"

    # Premium tier
    if any(
        premium in model_lower
        for premium in ["opus", "claude-3-opus", "gpt-4o", "gpt-4", "gemini-pro-1.5"]
    ):
        return "premium"

    # Mid-tier (default for sonnet and similar)
    if any(mid in model_lower for mid in ["sonnet", "claude-3-5-sonnet", "gemini-pro"]):
        return "mid-tier"

    return "unknown"


def get_cross_provider_alternatives(current_model: str, target_tier: str) -> list[str]:
    """
    Get cross-provider alternatives for a given tier.

    Args:
        current_model: The current model to find alternatives for
        target_tier: Target tier ("budget", "mid-tier", "premium")

    Returns:
        List of alternative models from different providers
    """
    current_provider = detect_model_provider(current_model)
    current_lower = current_model.lower()
    alternatives = []

    if target_tier == "budget":
        budget_options = [
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
        ]
        # Exclude current provider's model
        for option in budget_options:
            option_provider = detect_model_provider(option)
            if (
                option_provider != current_provider
                and option.lower() not in current_lower
            ):
                alternatives.append(option)

    elif target_tier == "mid-tier":
        mid_options = ["anthropic/claude-3-5-sonnet", "google/gemini-pro"]
        for option in mid_options:
            option_provider = detect_model_provider(option)
            if (
                option_provider != current_provider
                and option.lower() not in current_lower
            ):
                alternatives.append(option)

    elif target_tier == "premium":
        premium_options = [
            "anthropic/claude-3-opus",
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4",
            "google/gemini-pro-1.5",
        ]
        for option in premium_options:
            option_provider = detect_model_provider(option)
            if (
                option_provider != current_provider
                and option.lower() not in current_lower
            ):
                alternatives.append(option)

    return alternatives


def format_cost_comparison(
    current_cost: Decimal, alternative_cost: Decimal, is_upgrade: bool = True
) -> tuple[Decimal, str, str]:
    """
    Format cost comparison between current and alternative model.

    Returns:
        (cost_difference, percentage_change, formatted_description)
    """
    if current_cost == 0 and alternative_cost == 0:
        return Decimal("0"), "0%", "No cost change"

    if current_cost == 0:
        # Upgrading from free (local) model
        return alternative_cost, "∞%", f"${alternative_cost:.4f} (from free)"

    cost_diff = alternative_cost - current_cost

    if alternative_cost == 0:
        # Downgrading to free (local) model
        return cost_diff, "100%", f"Save ${abs(cost_diff):.4f} (to free)"

    percentage = (abs(cost_diff) / current_cost * 100) if current_cost > 0 else 0

    if is_upgrade and cost_diff > 0:
        return (
            cost_diff,
            f"+{percentage:.0f}%",
            f"+${cost_diff:.4f} ({percentage:.0f}% increase)",
        )
    elif not is_upgrade and cost_diff < 0:
        return (
            cost_diff,
            f"-{percentage:.0f}%",
            f"Save ${abs(cost_diff):.4f} ({percentage:.0f}% savings)",
        )
    else:
        return (
            cost_diff,
            f"{percentage:.0f}%",
            f"${abs(cost_diff):.4f} ({'increase' if cost_diff > 0 else 'savings'})",
        )


def should_recommend_change(
    score: float,
    cost_diff: Decimal,
    current_cost: Decimal,
    is_upgrade: bool = True,
    quality_threshold: float = 7.0,
    cost_ratio_threshold: float = 2.0,
) -> tuple[bool, str]:
    """
    Determine if a model change should be recommended based on quality and cost.

    Args:
        score: Quality comparison score (1-10)
        cost_diff: Cost difference (positive = more expensive)
        current_cost: Current model cost for ratio calculation
        is_upgrade: Whether this is an upgrade recommendation
        quality_threshold: Minimum quality score for recommendation
        cost_ratio_threshold: Maximum cost ratio increase to accept

    Returns:
        (should_recommend, reasoning)
    """
    if is_upgrade:
        # Upgrade logic: quality improvement should justify cost increase
        if score < quality_threshold:
            return (
                False,
                f"Quality improvement insufficient ({score:.1f}/10 < {quality_threshold})",
            )

        if cost_diff <= 0:
            return True, f"Quality improvement ({score:.1f}/10) with cost savings"

        if current_cost > 0:
            cost_ratio = (current_cost + cost_diff) / current_cost
            if cost_ratio > cost_ratio_threshold:
                return (
                    False,
                    f"Cost increase too high ({cost_ratio:.1f}x > {cost_ratio_threshold}x)",
                )

        return True, f"Quality improvement ({score:.1f}/10) justifies cost increase"

    else:
        # Downgrade logic: maintain acceptable quality while reducing cost
        if score < quality_threshold - 1.0:  # More lenient for downgrades
            return False, f"Quality degradation too significant ({score:.1f}/10)"

        if cost_diff >= 0:
            return False, "No cost savings achieved"

        return True, f"Acceptable quality ({score:.1f}/10) with cost savings"
