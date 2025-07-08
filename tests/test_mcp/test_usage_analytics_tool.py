"""
Tests for the usage_analytics MCP tool.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from second_opinion.mcp.tools.usage_analytics import usage_analytics_tool


class TestUsageAnalyticsTool:
    """Test the usage_analytics MCP tool functionality."""

    @pytest.fixture
    def mock_analytics_data(self):
        """Create mock analytics data for testing."""
        return {
            "period": {
                "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
                "end_date": datetime.now().isoformat(),
            },
            "summary": {
                "total_conversations": 25,
                "total_cost": Decimal("1.25"),
                "average_cost_per_conversation": Decimal("0.05"),
            },
            "model_usage": [
                {
                    "model": "anthropic/claude-3-5-sonnet",
                    "usage_count": 15,
                    "total_cost": Decimal("0.75"),
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "usage_count": 10,
                    "total_cost": Decimal("0.50"),
                },
            ],
            "interface_breakdown": [
                {
                    "interface": "mcp",
                    "conversation_count": 15,
                    "total_cost": Decimal("0.75"),
                },
                {
                    "interface": "cli",
                    "conversation_count": 10,
                    "total_cost": Decimal("0.50"),
                },
            ],
        }

    @pytest.fixture
    def mock_budget_status(self):
        """Create mock budget status for testing."""
        return {
            "daily": {
                "used": Decimal("0.50"),
                "limit": Decimal("5.00"),
                "available": Decimal("4.50"),
                "percentage_used": 10.0,
            },
            "weekly": {
                "used": Decimal("1.25"),
                "limit": Decimal("25.00"),
                "available": Decimal("23.75"),
                "percentage_used": 5.0,
            },
            "monthly": {
                "used": Decimal("3.75"),
                "limit": Decimal("100.00"),
                "available": Decimal("96.25"),
                "percentage_used": 3.75,
            },
        }

    @pytest.fixture
    def mock_trends_data(self):
        """Create mock trends data for testing."""
        return {
            "segments": [
                {
                    "period": "01/15 - 01/16",
                    "conversations": 5,
                    "cost": Decimal("0.25"),
                    "avg_cost": Decimal("0.05"),
                },
                {
                    "period": "01/17 - 01/18",
                    "conversations": 8,
                    "cost": Decimal("0.40"),
                    "avg_cost": Decimal("0.05"),
                },
                {
                    "period": "01/19 - 01/20",
                    "conversations": 12,
                    "cost": Decimal("0.60"),
                    "avg_cost": Decimal("0.05"),
                },
            ],
            "conversations_trend": {"direction": "increasing", "change_percent": 20.0},
            "cost_trend": {"direction": "increasing", "change_percent": 15.0},
        }

    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_basic_analytics(
        self, mock_cost_guard, mock_store, mock_analytics_data
    ):
        """Test basic analytics functionality."""
        # Setup mocks
        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.return_value = mock_analytics_data

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        # Mock budget check
        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        # Mock budget status
        mock_cost_guard_instance.get_usage_summary.return_value = MagicMock(
            used=Decimal("0.50"), limit=Decimal("5.00"), available=Decimal("4.50")
        )

        # Execute
        result = await usage_analytics_tool(
            time_period="week",
            breakdown_by="model",
            interface_type=None,
            include_trends=False,
            include_recommendations=False,
            cost_limit=0.10,
        )

        # Verify
        assert isinstance(result, str)
        assert "📊 Second Opinion Usage Analytics" in result
        assert "**Total Conversations**: 25" in result
        assert "**Total Cost**: $1.2500" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "openai/gpt-4o-mini" in result

        # Verify store was called correctly
        mock_store_instance.get_usage_analytics.assert_called_once()
        call_args = mock_store_instance.get_usage_analytics.call_args
        assert call_args.kwargs["interface_type"] is None

        # Verify cost tracking
        mock_cost_guard_instance.check_and_reserve_budget.assert_called_once()
        mock_cost_guard_instance.record_actual_cost.assert_called_once()

    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_interface_filtering(
        self, mock_cost_guard, mock_store, mock_analytics_data
    ):
        """Test interface filtering functionality."""
        # Setup mocks
        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.return_value = mock_analytics_data

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_cost_guard_instance.get_usage_summary.return_value = MagicMock(
            used=Decimal("0.50"), limit=Decimal("5.00"), available=Decimal("4.50")
        )

        # Execute with MCP interface filter
        result = await usage_analytics_tool(
            time_period="month",
            breakdown_by="interface",
            interface_type="mcp",
            include_trends=False,
            include_recommendations=False,
        )

        # Verify
        assert isinstance(result, str)
        assert "**Interface Filter**: MCP" in result

        # Verify store was called with interface filter
        call_args = mock_store_instance.get_usage_analytics.call_args
        assert call_args.kwargs["interface_type"] == "mcp"

    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_trends_analysis(
        self, mock_cost_guard, mock_store, mock_analytics_data, mock_trends_data
    ):
        """Test trends analysis functionality."""
        # Setup mocks
        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.return_value = mock_analytics_data

        # Mock multiple calls for trends
        def mock_analytics_side_effect(*args, **kwargs):
            call_count: int = getattr(mock_analytics_side_effect, "call_count", 0)
            call_count += 1
            mock_analytics_side_effect.call_count = call_count  # type: ignore
            if call_count == 1:
                return mock_analytics_data
            # Return segment data for trends
            return {
                "summary": {
                    "total_conversations": 5 + call_count,
                    "total_cost": Decimal("0.25") * call_count,
                    "average_cost_per_conversation": Decimal("0.05"),
                }
            }

        mock_store_instance.get_usage_analytics.side_effect = mock_analytics_side_effect

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_cost_guard_instance.get_usage_summary.return_value = MagicMock(
            used=Decimal("0.50"), limit=Decimal("5.00"), available=Decimal("4.50")
        )

        # Execute with trends enabled
        result = await usage_analytics_tool(
            time_period="week",
            breakdown_by="time",
            include_trends=True,
            include_recommendations=False,
        )

        # Verify
        assert isinstance(result, str)
        assert "📈 Trends Analysis" in result
        assert "**Conversation Volume**:" in result
        assert "**Cost Trend**:" in result

        # Verify multiple store calls for trend segments
        assert mock_store_instance.get_usage_analytics.call_count > 1

    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_recommendations_generation(
        self, mock_cost_guard, mock_store, mock_analytics_data, mock_budget_status
    ):
        """Test recommendations generation."""
        # Setup mocks with high-cost scenario
        high_cost_data = mock_analytics_data.copy()
        high_cost_data["model_usage"] = [
            {
                "model": "anthropic/claude-3-opus",
                "usage_count": 20,
                "total_cost": Decimal("5.00"),  # High cost to trigger recommendation
            }
        ]

        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.return_value = high_cost_data

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        # Mock high budget usage to trigger warning
        high_usage_budget = MagicMock()
        high_usage_budget.used = Decimal("4.50")
        high_usage_budget.limit = Decimal("5.00")
        high_usage_budget.available = Decimal("0.50")
        mock_cost_guard_instance.get_usage_summary.return_value = high_usage_budget

        # Execute with recommendations enabled
        result = await usage_analytics_tool(
            time_period="week",
            breakdown_by="cost",
            include_trends=False,
            include_recommendations=True,
        )

        # Verify
        assert isinstance(result, str)
        assert "💡 Optimization Recommendations" in result
        assert "High Daily Budget Usage" in result or "High-Cost Model Usage" in result

    async def test_invalid_time_period(self):
        """Test invalid time period handling."""
        result = await usage_analytics_tool(
            time_period="invalid_period",
            breakdown_by="model",
        )

        assert "❌ **Invalid Time Period**" in result
        assert "invalid_period" in result

    async def test_invalid_breakdown(self):
        """Test invalid breakdown handling."""
        result = await usage_analytics_tool(
            time_period="week",
            breakdown_by="invalid_breakdown",
        )

        assert "❌ **Invalid Breakdown**" in result
        assert "invalid_breakdown" in result

    async def test_invalid_interface_type(self):
        """Test invalid interface type handling."""
        result = await usage_analytics_tool(
            time_period="week",
            breakdown_by="model",
            interface_type="invalid_interface",
        )

        assert "❌ **Invalid Interface Type**" in result
        assert "invalid_interface" in result

    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_budget_error(self, mock_cost_guard):
        """Test budget error handling."""
        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance
        mock_cost_guard_instance.check_and_reserve_budget.side_effect = Exception(
            "Budget exceeded"
        )

        result = await usage_analytics_tool(
            time_period="week", breakdown_by="model", cost_limit=0.01
        )

        assert "❌ **Budget Error**" in result
        assert "Budget exceeded" in result

    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_store_error_handling(self, mock_cost_guard, mock_store):
        """Test store error handling."""
        # Setup mocks
        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.side_effect = Exception(
            "Database error"
        )

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        result = await usage_analytics_tool(time_period="week", breakdown_by="model")

        assert "❌ **Analytics Error**" in result
        assert "Database error" in result

    def test_calculate_start_date(self):
        """Test start date calculation logic."""
        from second_opinion.mcp.tools.usage_analytics import _calculate_start_date

        end_date = datetime(2024, 1, 15, 12, 0, 0)

        # Test different periods
        assert _calculate_start_date("all", end_date) is None

        day_start = _calculate_start_date("day", end_date)
        assert day_start == datetime(2024, 1, 14, 12, 0, 0)

        week_start = _calculate_start_date("week", end_date)
        assert week_start == datetime(2024, 1, 8, 12, 0, 0)

        month_start = _calculate_start_date("month", end_date)
        assert month_start == datetime(2023, 12, 16, 12, 0, 0)

    def test_calculate_trend(self):
        """Test trend calculation logic."""
        from second_opinion.mcp.tools.usage_analytics import _calculate_trend

        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = _calculate_trend(increasing_values)
        assert trend["direction"] == "increasing"
        assert trend["change_percent"] > 0

        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = _calculate_trend(decreasing_values)
        assert trend["direction"] == "decreasing"
        assert trend["change_percent"] < 0

        # Test stable trend
        stable_values = [3.0, 3.1, 2.9, 3.0, 3.1]
        trend = _calculate_trend(stable_values)
        assert trend["direction"] == "stable"

        # Test empty/single values
        empty_trend = _calculate_trend([])
        assert empty_trend["direction"] == "stable"
        assert empty_trend["change_percent"] == 0

        single_trend = _calculate_trend([5.0])
        assert single_trend["direction"] == "stable"
        assert single_trend["change_percent"] == 0

    @pytest.mark.parametrize(
        "time_period,breakdown,interface",
        [
            ("day", "model", None),
            ("week", "interface", "cli"),
            ("month", "cost", "mcp"),
            ("quarter", "tool", None),
            ("year", "time", None),
            ("all", "model", None),
        ],
    )
    @patch("second_opinion.mcp.tools.usage_analytics.get_conversation_store")
    @patch("second_opinion.mcp.tools.usage_analytics.get_cost_guard")
    async def test_parameter_combinations(
        self,
        mock_cost_guard,
        mock_store,
        mock_analytics_data,
        time_period,
        breakdown,
        interface,
    ):
        """Test various parameter combinations."""
        # Setup mocks
        mock_store_instance = AsyncMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_usage_analytics.return_value = mock_analytics_data

        mock_cost_guard_instance = AsyncMock()
        mock_cost_guard.return_value = mock_cost_guard_instance

        budget_check = MagicMock()
        budget_check.reservation_id = "test-reservation"
        mock_cost_guard_instance.check_and_reserve_budget.return_value = budget_check

        mock_cost_guard_instance.get_usage_summary.return_value = MagicMock(
            used=Decimal("0.50"), limit=Decimal("5.00"), available=Decimal("4.50")
        )

        # Execute
        result = await usage_analytics_tool(
            time_period=time_period,
            breakdown_by=breakdown,
            interface_type=interface,
            include_trends=False,
            include_recommendations=False,
        )

        # Verify basic structure
        assert isinstance(result, str)
        assert "📊 Second Opinion Usage Analytics" in result
        assert f"**Primary Breakdown**: {breakdown.title()}" in result

        if interface:
            assert f"**Interface Filter**: {interface.upper()}" in result

        # Verify store was called with correct parameters
        call_args = mock_store_instance.get_usage_analytics.call_args
        assert call_args.kwargs["interface_type"] == interface
