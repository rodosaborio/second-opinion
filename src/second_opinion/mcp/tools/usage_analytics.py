"""
Usage Analytics MCP tool implementation.

This module provides analytics and insights on model usage patterns, cost optimization
opportunities, and performance trends based on stored conversation data.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ...database.store import get_conversation_store
from ...utils.cost_tracking import BudgetPeriod, get_cost_guard
from ...utils.sanitization import validate_cost_limit

logger = logging.getLogger(__name__)


async def usage_analytics_tool(
    time_period: str = "week",
    breakdown_by: str = "model",
    interface_type: str | None = None,
    include_trends: bool = True,
    include_recommendations: bool = True,
    cost_limit: float | None = None,
) -> str:
    """
    Analyze model usage patterns and provide cost optimization insights.

    This tool provides comprehensive analytics on your AI model usage, including:
    - Cost breakdown by model, time period, and interface
    - Usage patterns and trends over time
    - Cost optimization recommendations
    - Model performance insights
    - Budget analysis and projections

    Args:
        time_period: Analysis period - "day", "week", "month", "quarter", "year", or "all"
        breakdown_by: Primary breakdown dimension - "model", "interface", "tool", "cost", or "time"
        interface_type: Filter by interface - "cli", "mcp", or None for all
        include_trends: Whether to include trend analysis over time
        include_recommendations: Whether to include optimization recommendations
        cost_limit: Maximum cost for this analytics operation (default: $0.10)

    Returns:
        Comprehensive usage analytics report with insights and recommendations

    Usage Examples:
        # Weekly cost breakdown by model
        usage_analytics_tool(time_period="week", breakdown_by="model")

        # Monthly MCP tool usage analysis with trends
        usage_analytics_tool(time_period="month", interface_type="mcp", include_trends=True)

        # Cost optimization analysis for all time
        usage_analytics_tool(time_period="all", breakdown_by="cost", include_recommendations=True)
    """
    try:
        # Input validation
        valid_periods = ["day", "week", "month", "quarter", "year", "all"]
        if time_period not in valid_periods:
            return f"❌ **Invalid Time Period**: '{time_period}'\n\n**Valid options**: {', '.join(valid_periods)}"

        valid_breakdowns = ["model", "interface", "tool", "cost", "time"]
        if breakdown_by not in valid_breakdowns:
            return f"❌ **Invalid Breakdown**: '{breakdown_by}'\n\n**Valid options**: {', '.join(valid_breakdowns)}"

        if interface_type and interface_type not in ["cli", "mcp"]:
            return f"❌ **Invalid Interface Type**: '{interface_type}'\n\n**Valid options**: cli, mcp, or None for all"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            cost_limit_decimal = Decimal("0.10")  # Low cost for analytics

        logger.info(
            f"Starting usage analytics: period={time_period}, breakdown={breakdown_by}"
        )

        # Cost guard check (minimal cost for analytics)
        cost_guard = get_cost_guard()
        estimated_cost = Decimal("0.01")  # Very low cost for database queries

        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "usage_analytics",
                "local-analytics",  # Not using external models
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Calculate date range
        end_date = datetime.now()
        start_date = _calculate_start_date(time_period, end_date)

        # Get analytics data from conversation store
        store = get_conversation_store()
        analytics_data = await store.get_usage_analytics(
            start_date=start_date, end_date=end_date, interface_type=interface_type
        )

        # Get current budget status
        budget_status = await _get_budget_status(cost_guard)

        # Generate trends if requested
        trends_data = None
        if include_trends and time_period != "day":
            trends_data = await _generate_trends_analysis(
                store, start_date, end_date, interface_type
            )

        # Generate recommendations if requested
        recommendations = None
        if include_recommendations:
            recommendations = await _generate_optimization_recommendations(
                analytics_data, trends_data, budget_status
            )

        # Record minimal actual cost
        actual_cost = Decimal("0.01")
        await cost_guard.record_actual_cost(
            reservation_id, actual_cost, "local-analytics", "usage_analytics"
        )

        # Format comprehensive report
        return await _format_analytics_report(
            analytics_data=analytics_data,
            breakdown_by=breakdown_by,
            time_period=time_period,
            interface_type=interface_type,
            budget_status=budget_status,
            trends_data=trends_data,
            recommendations=recommendations,
            start_date=start_date,
            end_date=end_date,
        )

    except Exception as e:
        logger.error(f"Unexpected error in usage_analytics tool: {e}")
        return f"❌ **Analytics Error**: {str(e)}\n\nPlease check the logs and try again with simpler parameters."


def _calculate_start_date(time_period: str, end_date: datetime) -> datetime | None:
    """Calculate start date based on time period."""
    if time_period == "all":
        return None
    elif time_period == "day":
        return end_date - timedelta(days=1)
    elif time_period == "week":
        return end_date - timedelta(days=7)
    elif time_period == "month":
        return end_date - timedelta(days=30)
    elif time_period == "quarter":
        return end_date - timedelta(days=90)
    elif time_period == "year":
        return end_date - timedelta(days=365)
    else:
        return end_date - timedelta(days=7)  # Default to week


async def _get_budget_status(cost_guard) -> dict[str, Any]:
    """Get current budget status across different periods."""
    try:
        daily_usage = await cost_guard.get_usage_summary(BudgetPeriod.DAILY)
        weekly_usage = await cost_guard.get_usage_summary(BudgetPeriod.WEEKLY)
        monthly_usage = await cost_guard.get_usage_summary(BudgetPeriod.MONTHLY)

        return {
            "daily": {
                "used": daily_usage.used,
                "limit": daily_usage.limit,
                "available": daily_usage.available,
                "percentage_used": (daily_usage.used / daily_usage.limit * 100)
                if daily_usage.limit > 0
                else 0,
            },
            "weekly": {
                "used": weekly_usage.used,
                "limit": weekly_usage.limit,
                "available": weekly_usage.available,
                "percentage_used": (weekly_usage.used / weekly_usage.limit * 100)
                if weekly_usage.limit > 0
                else 0,
            },
            "monthly": {
                "used": monthly_usage.used,
                "limit": monthly_usage.limit,
                "available": monthly_usage.available,
                "percentage_used": (monthly_usage.used / monthly_usage.limit * 100)
                if monthly_usage.limit > 0
                else 0,
            },
        }
    except Exception as e:
        logger.warning(f"Failed to get budget status: {e}")
        return {
            "daily": {
                "used": Decimal("0"),
                "limit": Decimal("0"),
                "available": Decimal("0"),
                "percentage_used": 0,
            },
            "weekly": {
                "used": Decimal("0"),
                "limit": Decimal("0"),
                "available": Decimal("0"),
                "percentage_used": 0,
            },
            "monthly": {
                "used": Decimal("0"),
                "limit": Decimal("0"),
                "available": Decimal("0"),
                "percentage_used": 0,
            },
        }


async def _generate_trends_analysis(
    store, start_date: datetime | None, end_date: datetime, interface_type: str | None
) -> dict[str, Any]:
    """Generate trends analysis by breaking down the period into segments."""
    if not start_date:
        # For "all" period, use last 30 days for trends
        start_date = end_date - timedelta(days=30)

    # Break period into 5 segments for trend analysis
    total_days = (end_date - start_date).days
    segment_days = max(1, total_days // 5)

    segments = []
    current_start = start_date

    for _i in range(5):
        current_end = min(current_start + timedelta(days=segment_days), end_date)

        segment_data = await store.get_usage_analytics(
            start_date=current_start,
            end_date=current_end,
            interface_type=interface_type,
        )

        segments.append(
            {
                "period": f"{current_start.strftime('%m/%d')} - {current_end.strftime('%m/%d')}",
                "conversations": segment_data["summary"]["total_conversations"],
                "cost": segment_data["summary"]["total_cost"],
                "avg_cost": segment_data["summary"]["average_cost_per_conversation"],
            }
        )

        current_start = current_end + timedelta(days=1)
        if current_start > end_date:
            break

    # Calculate trends
    conversations_trend = _calculate_trend([s["conversations"] for s in segments])
    cost_trend = _calculate_trend([float(s["cost"]) for s in segments])

    return {
        "segments": segments,
        "conversations_trend": conversations_trend,
        "cost_trend": cost_trend,
    }


def _calculate_trend(values: list[float]) -> dict[str, Any]:
    """Calculate trend direction and percentage change."""
    if len(values) < 2:
        return {"direction": "stable", "change_percent": 0}

    # Compare first half vs second half
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
    second_half_avg = (
        sum(values[mid:]) / (len(values) - mid) if len(values) - mid > 0 else 0
    )

    if first_half_avg == 0:
        if second_half_avg > 0:
            return {"direction": "increasing", "change_percent": 100}
        else:
            return {"direction": "stable", "change_percent": 0}

    change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    if change_percent > 10:
        direction = "increasing"
    elif change_percent < -10:
        direction = "decreasing"
    else:
        direction = "stable"

    return {"direction": direction, "change_percent": change_percent}


async def _generate_optimization_recommendations(
    analytics_data: dict[str, Any],
    trends_data: dict[str, Any] | None,
    budget_status: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate cost optimization and usage recommendations."""
    recommendations = []

    # Budget-based recommendations
    daily_usage = budget_status["daily"]["percentage_used"]
    if daily_usage > 80:
        recommendations.append(
            {
                "type": "budget_warning",
                "title": "High Daily Budget Usage",
                "description": f"You've used {daily_usage:.1f}% of your daily budget. Consider optimizing high-cost models or using local alternatives.",
                "priority": "high",
            }
        )

    # Model usage recommendations
    model_usage = analytics_data.get("model_usage", [])
    if len(model_usage) > 0:
        # Find most expensive model
        most_expensive = max(model_usage, key=lambda x: float(x["total_cost"]))
        if float(most_expensive["total_cost"]) > 1.0:  # More than $1 spent
            recommendations.append(
                {
                    "type": "cost_optimization",
                    "title": "High-Cost Model Usage",
                    "description": f"'{most_expensive['model']}' has cost ${most_expensive['total_cost']:.2f}. Consider testing cheaper alternatives like 'openai/gpt-4o-mini' for similar tasks.",
                    "priority": "medium",
                }
            )

        # Check for local model opportunity
        has_local_models = any(
            "lmstudio" in model.get("model", "").lower()
            or "/" not in model.get("model", "")
            for model in model_usage
        )
        if not has_local_models:
            recommendations.append(
                {
                    "type": "local_models",
                    "title": "Local Model Opportunity",
                    "description": "You haven't used any local models. Consider testing 'qwen3-4b-mlx' or 'codestral-22b-v0.1' for cost-free development work.",
                    "priority": "low",
                }
            )

    # Trend-based recommendations
    if trends_data:
        cost_trend = trends_data.get("cost_trend", {})
        if (
            cost_trend.get("direction") == "increasing"
            and cost_trend.get("change_percent", 0) > 25
        ):
            recommendations.append(
                {
                    "type": "trend_warning",
                    "title": "Rising Cost Trend",
                    "description": f"Your costs are trending upward ({cost_trend['change_percent']:.1f}% increase). Review recent usage patterns and consider cost controls.",
                    "priority": "medium",
                }
            )

    # Interface optimization
    interface_breakdown = analytics_data.get("interface_breakdown", [])
    if len(interface_breakdown) > 1:
        cli_usage = next(
            (i for i in interface_breakdown if i["interface"] == "cli"), None
        )
        mcp_usage = next(
            (i for i in interface_breakdown if i["interface"] == "mcp"), None
        )

        if cli_usage and mcp_usage:
            cli_cost = float(cli_usage["total_cost"])
            mcp_cost = float(mcp_usage["total_cost"])

            if cli_cost > mcp_cost * 2:  # CLI costs significantly more
                recommendations.append(
                    {
                        "type": "interface_optimization",
                        "title": "CLI vs MCP Usage",
                        "description": "CLI usage is costing significantly more than MCP tools. Consider using MCP tools for routine comparisons to leverage response reuse.",
                        "priority": "low",
                    }
                )

    return recommendations


async def _format_analytics_report(
    analytics_data: dict[str, Any],
    breakdown_by: str,
    time_period: str,
    interface_type: str | None,
    budget_status: dict[str, Any],
    trends_data: dict[str, Any] | None,
    recommendations: list[dict[str, str]] | None,
    start_date: datetime | None,
    end_date: datetime,
) -> str:
    """Format comprehensive analytics report."""

    report = []

    # Header
    report.append("# 📊 Second Opinion Usage Analytics")
    report.append("")

    # Period info
    if start_date:
        report.append(
            f"**Analysis Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
    else:
        report.append(
            f"**Analysis Period**: All time (through {end_date.strftime('%Y-%m-%d')})"
        )

    if interface_type:
        report.append(f"**Interface Filter**: {interface_type.upper()}")

    report.append(f"**Primary Breakdown**: {breakdown_by.title()}")
    report.append("")

    # Executive Summary
    summary = analytics_data["summary"]
    report.append("## 📈 Executive Summary")
    report.append(f"**Total Conversations**: {summary['total_conversations']:,}")
    report.append(f"**Total Cost**: ${summary['total_cost']:.4f}")
    report.append(
        f"**Average Cost per Conversation**: ${summary['average_cost_per_conversation']:.4f}"
    )
    report.append("")

    # Budget Status
    report.append("## 💰 Budget Status")
    daily = budget_status["daily"]
    weekly = budget_status["weekly"]
    monthly = budget_status["monthly"]

    report.append(
        f"**Daily**: ${daily['used']:.4f} / ${daily['limit']:.2f} ({daily['percentage_used']:.1f}% used)"
    )
    report.append(
        f"**Weekly**: ${weekly['used']:.4f} / ${weekly['limit']:.2f} ({weekly['percentage_used']:.1f}% used)"
    )
    report.append(
        f"**Monthly**: ${monthly['used']:.4f} / ${monthly['limit']:.2f} ({monthly['percentage_used']:.1f}% used)"
    )
    report.append("")

    # Primary breakdown section
    if breakdown_by == "model":
        await _add_model_breakdown(report, analytics_data["model_usage"])
    elif breakdown_by == "interface":
        await _add_interface_breakdown(report, analytics_data["interface_breakdown"])
    elif breakdown_by == "cost":
        await _add_cost_breakdown(report, analytics_data)
    elif breakdown_by == "time" and trends_data:
        await _add_time_breakdown(report, trends_data)

    # Trends analysis
    if trends_data:
        report.append("## 📈 Trends Analysis")
        conversations_trend = trends_data["conversations_trend"]
        cost_trend = trends_data["cost_trend"]

        report.append(
            f"**Conversation Volume**: {conversations_trend['direction']} ({conversations_trend['change_percent']:+.1f}%)"
        )
        report.append(
            f"**Cost Trend**: {cost_trend['direction']} ({cost_trend['change_percent']:+.1f}%)"
        )
        report.append("")

        # Add trend segments
        report.append("**Period Breakdown**:")
        for segment in trends_data["segments"]:
            report.append(
                f"- {segment['period']}: {segment['conversations']} conversations, ${segment['cost']:.4f}"
            )
        report.append("")

    # Recommendations
    if recommendations:
        report.append("## 💡 Optimization Recommendations")

        high_priority = [r for r in recommendations if r["priority"] == "high"]
        medium_priority = [r for r in recommendations if r["priority"] == "medium"]
        low_priority = [r for r in recommendations if r["priority"] == "low"]

        for priority_level, recs in [
            ("🚨 High Priority", high_priority),
            ("⚠️ Medium Priority", medium_priority),
            ("💡 Suggestions", low_priority),
        ]:
            if recs:
                report.append(f"### {priority_level}")
                for rec in recs:
                    report.append(f"**{rec['title']}**: {rec['description']}")
                    report.append("")

    # Quick Actions
    report.append("## 🚀 Quick Actions")
    if float(summary["total_cost"]) > 0:
        report.append("- Run `should_downgrade` on your most expensive model responses")
        report.append("- Test local models with `second_opinion` for development work")
        report.append("- Use `compare_responses` to validate model switching decisions")
    else:
        report.append("- Start using Second Opinion tools to begin tracking usage")
        report.append(
            "- Try the `second_opinion` tool with `--existing-response` to save costs"
        )

    report.append("")
    report.append("---")
    report.append("*Analytics Complete - Optimize your AI usage! 📊*")

    return "\n".join(report)


async def _add_model_breakdown(
    report: list[str], model_usage: list[dict[str, Any]]
) -> None:
    """Add model usage breakdown to report."""
    report.append("## 🤖 Model Usage Breakdown")

    if not model_usage:
        report.append("No model usage data available for this period.")
        report.append("")
        return

    # Sort by usage count
    sorted_models = sorted(model_usage, key=lambda x: x["usage_count"], reverse=True)

    for i, model in enumerate(sorted_models[:10]):  # Top 10 models
        rank = i + 1
        report.append(f"**{rank}. {model['model']}**")
        report.append(f"   - Usage: {model['usage_count']} calls")
        report.append(f"   - Total Cost: ${model['total_cost']:.4f}")
        report.append(
            f"   - Avg Cost: ${float(model['total_cost']) / model['usage_count']:.4f} per call"
        )
        report.append("")


async def _add_interface_breakdown(
    report: list[str], interface_breakdown: list[dict[str, Any]]
) -> None:
    """Add interface usage breakdown to report."""
    report.append("## 🖥️ Interface Usage Breakdown")

    if not interface_breakdown:
        report.append("No interface usage data available for this period.")
        report.append("")
        return

    for interface in interface_breakdown:
        interface_name = interface["interface"].upper()
        report.append(f"**{interface_name}**")
        report.append(f"   - Conversations: {interface['conversation_count']}")
        report.append(f"   - Total Cost: ${interface['total_cost']:.4f}")
        report.append("")


async def _add_cost_breakdown(
    report: list[str], analytics_data: dict[str, Any]
) -> None:
    """Add cost-focused breakdown to report."""
    report.append("## 💰 Cost Analysis Breakdown")

    model_usage = analytics_data.get("model_usage", [])
    if not model_usage:
        report.append("No cost data available for this period.")
        report.append("")
        return

    # Sort by total cost
    sorted_by_cost = sorted(
        model_usage, key=lambda x: float(x["total_cost"]), reverse=True
    )

    # Cost distribution
    total_cost = sum(float(m["total_cost"]) for m in model_usage)

    report.append(f"**Total Spend**: ${total_cost:.4f}")
    report.append("")
    report.append("**Top Cost Contributors**:")

    for i, model in enumerate(sorted_by_cost[:5]):
        percentage = (
            (float(model["total_cost"]) / total_cost * 100) if total_cost > 0 else 0
        )
        report.append(
            f"{i + 1}. {model['model']}: ${model['total_cost']:.4f} ({percentage:.1f}%)"
        )

    report.append("")


async def _add_time_breakdown(report: list[str], trends_data: dict[str, Any]) -> None:
    """Add time-based breakdown to report."""
    report.append("## ⏰ Time-Based Analysis")

    segments = trends_data.get("segments", [])
    if not segments:
        report.append("No time breakdown data available.")
        report.append("")
        return

    report.append("**Usage Over Time**:")
    for segment in segments:
        avg_cost = float(segment["avg_cost"]) if segment["avg_cost"] else 0
        report.append(
            f"- {segment['period']}: {segment['conversations']} conversations (${segment['cost']:.4f}, avg ${avg_cost:.4f})"
        )

    report.append("")
