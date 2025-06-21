"""
Tests for cost tracking and budget management utilities.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.second_opinion.core.models import Message, ModelRequest
from src.second_opinion.utils.cost_tracking import (
    BudgetExceededError,
    BudgetPeriod,
    BudgetReservation,
    BudgetStatus,
    BudgetUsage,
    CostGuard,
    CostLimitError,
    check_budget,
    get_cost_guard,
    record_cost,
    set_cost_guard,
)
from src.second_opinion.utils.sanitization import ValidationError


class TestBudgetReservation:
    """Test BudgetReservation class."""

    def test_reservation_creation(self):
        """Test creating a budget reservation."""
        reservation = BudgetReservation(
            reservation_id="test-123",
            estimated_cost=Decimal("0.05"),
            operation_type="second_opinion",
            timestamp=datetime.now(UTC),
            model="gpt-4",
        )

        assert reservation.reservation_id == "test-123"
        assert reservation.estimated_cost == Decimal("0.05")
        assert reservation.operation_type == "second_opinion"
        assert reservation.model == "gpt-4"
        assert reservation.user_id is None

    def test_reservation_expiry(self):
        """Test reservation expiry logic."""
        # Recent reservation
        recent = BudgetReservation(
            reservation_id="recent",
            estimated_cost=Decimal("0.01"),
            operation_type="test",
            timestamp=datetime.now(UTC) - timedelta(seconds=60),
            model="gpt-4",
        )
        assert not recent.is_expired(300)  # 5 minutes timeout

        # Expired reservation
        expired = BudgetReservation(
            reservation_id="expired",
            estimated_cost=Decimal("0.01"),
            operation_type="test",
            timestamp=datetime.now(UTC) - timedelta(seconds=600),
            model="gpt-4",
        )
        assert expired.is_expired(300)


class TestBudgetUsage:
    """Test BudgetUsage class."""

    def test_budget_usage_properties(self):
        """Test budget usage calculated properties."""
        usage = BudgetUsage(
            period=BudgetPeriod.DAILY,
            current_usage=Decimal("3.00"),
            limit=Decimal("5.00"),
            reserved=Decimal("1.00"),
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC) + timedelta(days=1),
        )

        assert usage.available == Decimal("1.00")  # 5 - 3 - 1
        assert usage.utilization == 0.8  # (3 + 1) / 5
        assert usage.status == BudgetStatus.WARNING  # >= 80%

    def test_budget_status_ok(self):
        """Test budget status OK."""
        usage = BudgetUsage(
            period=BudgetPeriod.DAILY,
            current_usage=Decimal("1.00"),
            limit=Decimal("5.00"),
            reserved=Decimal("0.00"),
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC) + timedelta(days=1),
        )

        assert usage.status == BudgetStatus.OK
        assert usage.utilization == 0.2

    def test_budget_status_exceeded(self):
        """Test budget status exceeded."""
        usage = BudgetUsage(
            period=BudgetPeriod.DAILY,
            current_usage=Decimal("6.00"),
            limit=Decimal("5.00"),
            reserved=Decimal("0.00"),
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC) + timedelta(days=1),
        )

        assert usage.status == BudgetStatus.EXCEEDED
        assert usage.available == Decimal("0")  # Never negative

    def test_budget_status_reserved(self):
        """Test budget status with reservations."""
        usage = BudgetUsage(
            period=BudgetPeriod.DAILY,
            current_usage=Decimal("1.00"),
            limit=Decimal("5.00"),
            reserved=Decimal("2.00"),
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC) + timedelta(days=1),
        )

        assert usage.status == BudgetStatus.RESERVED
        assert usage.utilization == 0.6


class TestCostGuard:
    """Test CostGuard class."""

    def setup_method(self):
        """Set up test instance."""
        self.cost_guard = CostGuard(
            per_request_limit=Decimal("0.10"),
            daily_limit=Decimal("5.00"),
            weekly_limit=Decimal("25.00"),
            monthly_limit=Decimal("100.00"),
            warning_threshold=0.8,
        )

    def test_cost_guard_initialization(self):
        """Test CostGuard initialization."""
        assert self.cost_guard.per_request_limit == Decimal("0.10")
        assert self.cost_guard.daily_limit == Decimal("5.00")
        assert self.cost_guard.warning_threshold == 0.8
        assert len(self.cost_guard._usage_history) == 0
        assert len(self.cost_guard._reservations) == 0

    def test_cost_guard_with_invalid_limits(self):
        """Test CostGuard with invalid cost limits."""
        with pytest.raises(ValidationError):
            CostGuard(per_request_limit=Decimal("-1.00"))

        with pytest.raises(ValidationError):
            CostGuard(daily_limit="invalid")

    @pytest.mark.asyncio
    async def test_check_and_reserve_budget_success(self):
        """Test successful budget check and reservation."""
        estimated_cost = Decimal("0.05")

        result = await self.cost_guard.check_and_reserve_budget(
            estimated_cost, "second_opinion", "gpt-4", "user123"
        )

        assert result.estimated_cost == estimated_cost
        assert result.approved is True
        assert result.reservation_id is not None
        assert result.budget_remaining > Decimal("0")
        assert result.daily_budget_remaining >= Decimal("0")
        assert result.monthly_budget_remaining >= Decimal("0")

        # Check that reservation was created
        assert len(self.cost_guard._reservations) == 1
        reservation = list(self.cost_guard._reservations.values())[0]
        assert reservation.estimated_cost == estimated_cost
        assert reservation.operation_type == "second_opinion"
        assert reservation.model == "gpt-4"
        assert reservation.user_id == "user123"

    @pytest.mark.asyncio
    async def test_check_budget_per_request_limit_exceeded(self):
        """Test per-request limit exceeded."""
        excessive_cost = Decimal("0.50")  # Exceeds 0.10 limit

        with pytest.raises(CostLimitError) as exc_info:
            await self.cost_guard.check_and_reserve_budget(
                excessive_cost, "test", "gpt-4"
            )

        assert "per-request limit" in str(exc_info.value)
        assert exc_info.value.estimated_cost == excessive_cost
        assert exc_info.value.limit == Decimal("0.10")

    @pytest.mark.asyncio
    async def test_check_budget_daily_limit_exceeded(self):
        """Test daily budget limit exceeded."""
        # Add usage that approaches daily limit
        for i in range(10):
            await self._add_mock_usage(Decimal("0.50"), hours_ago=i)

        # This should exceed the daily limit of 5.00
        with pytest.raises(BudgetExceededError) as exc_info:
            await self.cost_guard.check_and_reserve_budget(
                Decimal("0.05"), "test", "gpt-4"
            )

        assert "daily" in str(exc_info.value)
        assert exc_info.value.period == "daily"

    @pytest.mark.asyncio
    async def test_record_actual_cost(self):
        """Test recording actual cost."""
        # First, make a reservation
        estimated_cost = Decimal("0.05")
        budget_check = await self.cost_guard.check_and_reserve_budget(
            estimated_cost, "second_opinion", "gpt-4"
        )

        # Record actual cost
        actual_cost = Decimal("0.03")
        cost_analysis = await self.cost_guard.record_actual_cost(
            budget_check.reservation_id,
            actual_cost,
            "gpt-4",
            "second_opinion",
            {"tokens": 150},
        )

        assert cost_analysis.estimated_cost == estimated_cost
        assert cost_analysis.actual_cost == actual_cost
        assert cost_analysis.budget_remaining >= Decimal("0")

        # Check that reservation was removed and usage recorded
        assert len(self.cost_guard._reservations) == 0
        assert len(self.cost_guard._usage_history) == 1

        usage_record = self.cost_guard._usage_history[0]
        assert usage_record["actual_cost"] == actual_cost
        assert usage_record["model"] == "gpt-4"
        assert usage_record["metadata"]["tokens"] == 150

    @pytest.mark.asyncio
    async def test_record_cost_invalid_reservation(self):
        """Test recording cost with invalid reservation ID."""
        with pytest.raises(ValueError, match="Reservation .* not found"):
            await self.cost_guard.record_actual_cost(
                "invalid-id", Decimal("0.05"), "gpt-4", "test"
            )

    @pytest.mark.asyncio
    async def test_get_usage_summary(self):
        """Test getting usage summary."""
        # Add some usage
        await self._add_mock_usage(Decimal("2.00"), hours_ago=2)
        await self._add_mock_usage(Decimal("1.50"), hours_ago=1)

        usage = await self.cost_guard.get_usage_summary(BudgetPeriod.DAILY)

        assert usage.period == BudgetPeriod.DAILY
        assert usage.current_usage == Decimal("3.50")
        assert usage.limit == Decimal("5.00")
        assert usage.available == Decimal("1.50")
        assert usage.status == BudgetStatus.OK  # 70% usage, below 80% threshold

    @pytest.mark.asyncio
    async def test_get_usage_summary_with_user_filter(self):
        """Test usage summary with user filtering."""
        await self._add_mock_usage(Decimal("1.00"), hours_ago=1, user_id="user1")
        await self._add_mock_usage(Decimal("2.00"), hours_ago=1, user_id="user2")

        usage_user1 = await self.cost_guard.get_usage_summary(
            BudgetPeriod.DAILY, "user1"
        )
        usage_user2 = await self.cost_guard.get_usage_summary(
            BudgetPeriod.DAILY, "user2"
        )
        usage_all = await self.cost_guard.get_usage_summary(BudgetPeriod.DAILY)

        assert usage_user1.current_usage == Decimal("1.00")
        assert usage_user2.current_usage == Decimal("2.00")
        assert usage_all.current_usage == Decimal("3.00")

    @pytest.mark.asyncio
    async def test_get_detailed_analytics(self):
        """Test detailed analytics generation."""
        # Add varied usage across multiple days
        # Use direct timestamp manipulation to ensure we get multiple days
        now = datetime.now(UTC)
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        # Add today's usage
        await self._add_mock_usage(
            Decimal("1.00"), hours_ago=1, model="gpt-4", operation="second_opinion"
        )
        await self._add_mock_usage(
            Decimal("0.50"), hours_ago=2, model="claude-3", operation="compare"
        )

        # Add yesterday's usage manually
        usage_record = {
            "timestamp": yesterday,
            "actual_cost": Decimal("2.00"),
            "estimated_cost": Decimal("2.00"),
            "model": "gpt-4",
            "operation_type": "second_opinion",
            "user_id": None,
            "metadata": {},
        }
        self.cost_guard._usage_history.append(usage_record)

        analytics = await self.cost_guard.get_detailed_analytics(days=30)

        assert analytics["total_cost"] == Decimal("3.50")
        assert analytics["total_requests"] == 3
        assert analytics["average_cost_per_request"] == Decimal("3.50") / 3
        assert "gpt-4" in analytics["models_used"]
        assert "claude-3" in analytics["models_used"]
        assert "second_opinion" in analytics["operations_breakdown"]
        assert "compare" in analytics["operations_breakdown"]
        assert len(analytics["daily_spending"]) >= 2

    @pytest.mark.asyncio
    async def test_estimate_request_cost(self):
        """Test request cost estimation."""
        request = ModelRequest(
            model="gpt-4",
            messages=[Message(role="user", content="What is AI?")],
            max_tokens=100,
        )

        model_pricing = {
            "gpt-4": (Decimal("0.03"), Decimal("0.06"))  # input, output per 1k tokens
        }

        cost = await self.cost_guard.estimate_request_cost(request, model_pricing)

        assert cost > Decimal("0")
        assert cost < Decimal("1.00")  # Reasonable range

    @pytest.mark.asyncio
    async def test_estimate_request_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        request = ModelRequest(
            model="unknown-model", messages=[Message(role="user", content="Test")]
        )

        cost = await self.cost_guard.estimate_request_cost(request, {})
        assert cost == Decimal("0.10")  # Conservative fallback

    @pytest.mark.asyncio
    async def test_reservation_cleanup(self):
        """Test cleanup of expired reservations."""
        # Create reservation and manually set timestamp to expired
        budget_check = await self.cost_guard.check_and_reserve_budget(
            Decimal("0.05"), "test", "gpt-4"
        )

        # Manually expire the reservation
        reservation = self.cost_guard._reservations[budget_check.reservation_id]
        reservation.timestamp = datetime.now(UTC) - timedelta(
            seconds=600
        )  # 10 minutes ago

        # Trigger cleanup
        await self.cost_guard._cleanup_expired_reservations()

        assert len(self.cost_guard._reservations) == 0

    @pytest.mark.asyncio
    async def test_weekly_and_monthly_budgets(self):
        """Test weekly and monthly budget periods."""
        weekly_usage = await self.cost_guard.get_usage_summary(BudgetPeriod.WEEKLY)
        monthly_usage = await self.cost_guard.get_usage_summary(BudgetPeriod.MONTHLY)

        assert weekly_usage.period == BudgetPeriod.WEEKLY
        assert weekly_usage.limit == Decimal("25.00")
        assert monthly_usage.period == BudgetPeriod.MONTHLY
        assert monthly_usage.limit == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_budget_warning_generation(self):
        """Test budget warning message generation."""
        # Fill budget to warning level
        await self._add_mock_usage(Decimal("4.50"), hours_ago=1)  # 90% of daily limit

        budget_check = await self.cost_guard.check_and_reserve_budget(
            Decimal("0.01"), "test", "gpt-4"
        )

        assert budget_check.warning_message is not None
        assert "Budget warning" in budget_check.warning_message
        assert "daily" in budget_check.warning_message

    async def _add_mock_usage(
        self,
        cost: Decimal,
        hours_ago: int = 0,
        model: str = "gpt-4",
        operation: str = "test",
        user_id: str | None = None,
    ):
        """Helper to add mock usage records."""
        now = datetime.now(UTC)
        # Ensure timestamp stays within the current day for daily budget tests
        # Start from noon today and subtract hours to avoid crossing day boundaries
        base_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
        timestamp = base_time - timedelta(hours=hours_ago)

        # If we go before today, just use early morning today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if timestamp < today_start:
            timestamp = today_start + timedelta(
                minutes=hours_ago * 10
            )  # Use minutes instead

        usage_record = {
            "timestamp": timestamp,
            "actual_cost": cost,
            "estimated_cost": cost,
            "model": model,
            "operation_type": operation,
            "user_id": user_id,
            "metadata": {},
        }
        self.cost_guard._usage_history.append(usage_record)


class TestGlobalFunctions:
    """Test global convenience functions."""

    def setup_method(self):
        """Reset global state."""
        set_cost_guard(
            CostGuard(per_request_limit=Decimal("0.05"), daily_limit=Decimal("2.00"))
        )

    def test_get_cost_guard(self):
        """Test getting global cost guard."""
        guard = get_cost_guard()
        assert isinstance(guard, CostGuard)

        # Should return same instance
        guard2 = get_cost_guard()
        assert guard is guard2

    def test_set_cost_guard(self):
        """Test setting global cost guard."""
        new_guard = CostGuard(per_request_limit=Decimal("0.20"))
        set_cost_guard(new_guard)

        retrieved_guard = get_cost_guard()
        assert retrieved_guard is new_guard
        assert retrieved_guard.per_request_limit == Decimal("0.20")

    @pytest.mark.asyncio
    async def test_global_check_budget(self):
        """Test global budget check function."""
        result = await check_budget(Decimal("0.03"), "test", "gpt-4", "user123")

        assert result.estimated_cost == Decimal("0.03")
        assert result.approved is True
        assert result.reservation_id is not None

    @pytest.mark.asyncio
    async def test_global_record_cost(self):
        """Test global cost recording function."""
        # First check budget
        budget_result = await check_budget(Decimal("0.03"), "test", "gpt-4")

        # Then record actual cost
        cost_analysis = await record_cost(
            budget_result.reservation_id,
            Decimal("0.025"),
            "gpt-4",
            "test",
            {"success": True},
        )

        assert cost_analysis.estimated_cost == Decimal("0.03")
        assert cost_analysis.actual_cost == Decimal("0.025")

    def test_get_cost_guard_creates_default(self):
        """Test that get_cost_guard creates default instance."""
        # Reset global state
        set_cost_guard(None)

        # Getting cost guard should create default using configuration
        guard = get_cost_guard()
        assert isinstance(guard, CostGuard)
        assert guard.per_request_limit > Decimal(
            "0"
        )  # Should have a positive limit from config

    def test_set_cost_guard_none(self):
        """Test setting cost guard to None."""
        # Set to None
        set_cost_guard(None)

        # Getting should create new default
        guard = get_cost_guard()
        assert isinstance(guard, CostGuard)

    def test_global_state_persistence(self):
        """Test that global cost guard state persists."""
        # Create custom guard
        custom_guard = CostGuard(per_request_limit=Decimal("0.50"))
        set_cost_guard(custom_guard)

        # Verify it persists across multiple calls
        guard1 = get_cost_guard()
        guard2 = get_cost_guard()

        assert guard1 is custom_guard
        assert guard2 is custom_guard
        assert guard1 is guard2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        self.cost_guard = CostGuard()

    @pytest.mark.asyncio
    async def test_zero_cost_request(self):
        """Test handling of zero-cost requests."""
        result = await self.cost_guard.check_and_reserve_budget(
            Decimal("0"), "test", "local-model"
        )

        assert result.estimated_cost == Decimal("0")
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_very_small_cost(self):
        """Test handling of very small costs."""
        tiny_cost = Decimal("0.0001")
        result = await self.cost_guard.check_and_reserve_budget(
            tiny_cost, "test", "gpt-4"
        )

        assert result.estimated_cost == tiny_cost
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_concurrent_reservations(self):
        """Test concurrent budget reservations."""
        import asyncio

        # Create multiple concurrent reservations
        tasks = [
            self.cost_guard.check_and_reserve_budget(
                Decimal("0.05"), f"test-{i}", "gpt-4"
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert len(self.cost_guard._reservations) == 5
        assert all(r.approved for r in results)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_cost_limit_validation_security(self):
        """Test cost limit validation security."""
        # Test with potentially malicious inputs
        with pytest.raises(ValidationError):
            CostGuard(per_request_limit="<script>alert('xss')</script>")

        with pytest.raises(ValidationError):
            CostGuard(daily_limit=None)

    @pytest.mark.asyncio
    async def test_budget_period_edge_cases(self):
        """Test budget period calculations at boundaries."""
        # Test at month boundary (December -> January)
        with patch("src.second_opinion.utils.cost_tracking.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)
            mock_dt.min = datetime.min
            mock_dt.max = datetime.max

            usage = await self.cost_guard.get_usage_summary(BudgetPeriod.MONTHLY)
            assert usage.period == BudgetPeriod.MONTHLY

    @pytest.mark.asyncio
    async def test_analytics_with_no_data(self):
        """Test analytics with no usage data."""
        analytics = await self.cost_guard.get_detailed_analytics(days=30)

        assert analytics["total_cost"] == Decimal("0")
        assert analytics["total_requests"] == 0
        assert analytics["average_cost_per_request"] == Decimal("0")
        assert analytics["models_used"] == []
        assert analytics["operations_breakdown"] == {}
