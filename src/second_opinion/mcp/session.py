"""
Lightweight session management for MCP interactions.

This module provides the MCPSession class for tracking costs, caching
model information, and maintaining conversation context across tool calls
within an MCP session.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from ..clients.base import ModelInfo

logger = logging.getLogger(__name__)


class MCPSession:
    """
    Lightweight session management for MCP interactions.

    This class provides:
    - Cost tracking across multiple tool calls
    - Model capability and pricing caching
    - Simple conversation context for recommendations
    - Session-based budget management

    Design Philosophy:
    - Explicit over implicit (no complex model detection)
    - Cost efficiency through caching and reuse
    - Lightweight state management
    """

    def __init__(self, session_id: str | None = None):
        """
        Initialize a new MCP session.

        Args:
            session_id: Optional session identifier. If None, generates a new UUID.
        """
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.now(UTC)
        self.last_activity = self.created_at

        # Cost tracking for this session
        self.total_cost: Decimal = Decimal("0.0")
        self.tool_costs: dict[str, Decimal] = {}
        self.operation_count = 0

        # Model information cache
        self.cached_model_info: dict[str, ModelInfo] = {}
        self.pricing_cache: dict[str, tuple[Decimal, Decimal]] = (
            {}
        )  # model -> (input_cost, output_cost)

        # Simple conversation context
        self.conversation_history: list[dict[str, Any]] = []
        self.last_used_model: str | None = None
        self.user_preferences: dict[str, Any] = {}

        logger.debug(f"Created MCP session {self.session_id}")

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now(UTC)

    def record_cost(self, tool_name: str, cost: Decimal, model: str) -> None:
        """
        Record cost for a tool operation.

        Args:
            tool_name: Name of the tool that incurred the cost
            cost: Cost amount in USD
            model: Model that was used
        """
        self.update_activity()
        self.total_cost += cost

        if tool_name not in self.tool_costs:
            self.tool_costs[tool_name] = Decimal("0.0")
        self.tool_costs[tool_name] += cost

        self.operation_count += 1
        self.last_used_model = model

        logger.debug(
            f"Session {self.session_id}: Recorded ${cost} for {tool_name} using {model}"
        )

    def cache_model_info(self, model: str, model_info: ModelInfo) -> None:
        """
        Cache model information for reuse.

        Args:
            model: Model identifier
            model_info: Model information to cache
        """
        self.cached_model_info[model] = model_info

        # Also cache pricing for quick lookups
        self.pricing_cache[model] = (
            model_info.input_cost_per_1k,
            model_info.output_cost_per_1k,
        )

        logger.debug(f"Session {self.session_id}: Cached info for model {model}")

    def get_cached_model_info(self, model: str) -> ModelInfo | None:
        """
        Get cached model information.

        Args:
            model: Model identifier

        Returns:
            Cached ModelInfo or None if not cached
        """
        return self.cached_model_info.get(model)

    def get_cached_pricing(self, model: str) -> tuple[Decimal, Decimal] | None:
        """
        Get cached pricing information.

        Args:
            model: Model identifier

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) or None if not cached
        """
        return self.pricing_cache.get(model)

    def add_conversation_context(
        self,
        tool_name: str,
        prompt: str,
        primary_model: str,
        comparison_models: list[str],
        result_summary: str,
    ) -> None:
        """
        Add conversation context for better recommendations.

        Args:
            tool_name: Name of the tool used
            prompt: User prompt or task
            primary_model: Primary model used
            comparison_models: List of comparison models
            result_summary: Brief summary of results
        """
        self.update_activity()

        context_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "tool": tool_name,
            "prompt": prompt[:200],  # Truncate for privacy
            "primary_model": primary_model,
            "comparison_models": comparison_models,
            "result_summary": result_summary,
        }

        self.conversation_history.append(context_entry)

        # Keep only recent history (last 10 interactions)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        logger.debug(
            f"Session {self.session_id}: Added conversation context for {tool_name}"
        )

    def get_model_usage_patterns(self) -> dict[str, int]:
        """
        Get model usage patterns from conversation history.

        Returns:
            Dictionary mapping model names to usage counts
        """
        usage: dict[str, int] = {}

        for entry in self.conversation_history:
            primary = entry.get("primary_model")
            if primary:
                usage[primary] = usage.get(primary, 0) + 1

            for model in entry.get("comparison_models", []):
                usage[model] = usage.get(model, 0) + 1

        return usage

    def suggest_primary_model(self) -> str | None:
        """
        Suggest primary model based on session history.

        Returns:
            Most frequently used model or None if no history
        """
        if self.last_used_model:
            return self.last_used_model

        usage_patterns = self.get_model_usage_patterns()
        if not usage_patterns:
            return None

        # Return most frequently used model
        return max(usage_patterns.items(), key=lambda x: x[1])[0]

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get a summary of session activity.

        Returns:
            Dictionary with session statistics and summaries
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "total_cost": float(self.total_cost),
            "operation_count": self.operation_count,
            "tool_costs": {tool: float(cost) for tool, cost in self.tool_costs.items()},
            "cached_models": list(self.cached_model_info.keys()),
            "conversation_entries": len(self.conversation_history),
            "last_used_model": self.last_used_model,
            "model_usage_patterns": self.get_model_usage_patterns(),
        }

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """
        Check if session has expired.

        Args:
            timeout_hours: Session timeout in hours

        Returns:
            True if session has expired
        """
        now = datetime.now(UTC)
        elapsed = (now - self.last_activity).total_seconds() / 3600
        return elapsed > timeout_hours

    def clear_conversation_history(self) -> None:
        """Clear conversation history for privacy."""
        self.conversation_history.clear()
        logger.debug(f"Session {self.session_id}: Cleared conversation history")

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking for the session."""
        self.total_cost = Decimal("0.0")
        self.tool_costs.clear()
        self.operation_count = 0
        self.last_used_model = None
        logger.debug(f"Session {self.session_id}: Reset cost tracking")
