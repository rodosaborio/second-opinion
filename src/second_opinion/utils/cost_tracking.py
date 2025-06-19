"""
Cost tracking and budget management utilities.

This module provides comprehensive cost tracking, budget protection, and
cost analysis functionality to prevent runaway spending.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from ..core.models import ModelRequest, BudgetCheck, CostAnalysis
from ..utils.pricing import get_pricing_manager
from ..utils.sanitization import validate_cost_limit


logger = logging.getLogger(__name__)


class BudgetPeriod(str, Enum):
    """Budget tracking periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"


class BudgetStatus(str, Enum):
    """Budget status indicators."""
    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"
    RESERVED = "reserved"


@dataclass
class BudgetReservation:
    """Represents a budget reservation for a pending operation."""
    reservation_id: str
    estimated_cost: Decimal
    operation_type: str
    timestamp: datetime
    model: str
    user_id: Optional[str] = None
    
    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if reservation has expired (default 5 minutes)."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() > timeout_seconds


@dataclass
class BudgetUsage:
    """Current budget usage for a specific period."""
    period: BudgetPeriod
    current_usage: Decimal
    limit: Decimal
    reserved: Decimal
    period_start: datetime
    period_end: datetime
    
    @property
    def available(self) -> Decimal:
        """Available budget (limit - usage - reserved)."""
        return max(Decimal('0'), self.limit - self.current_usage - self.reserved)
    
    @property
    def utilization(self) -> float:
        """Budget utilization as percentage (0.0 to 1.0+)."""
        if self.limit <= 0:
            return 0.0
        return float((self.current_usage + self.reserved) / self.limit)
    
    @property
    def status(self) -> BudgetStatus:
        """Current budget status."""
        if self.current_usage >= self.limit:
            return BudgetStatus.EXCEEDED
        elif self.utilization >= 0.8:  # 80% threshold
            return BudgetStatus.WARNING
        elif self.reserved > 0:
            return BudgetStatus.RESERVED
        else:
            return BudgetStatus.OK


class CostLimitError(Exception):
    """Cost limit exceeded error."""
    
    def __init__(self, message: str, estimated_cost: Decimal, limit: Decimal, period: str = "request"):
        self.message = message
        self.estimated_cost = estimated_cost
        self.limit = limit
        self.period = period
        super().__init__(message)


class BudgetExceededError(CostLimitError):
    """Budget exceeded error."""
    pass


class CostGuard:
    """
    Multi-layer cost protection and budget management.
    
    Provides:
    - Pre-request cost estimation and validation
    - Budget reservation system
    - Multiple budget periods (daily, weekly, monthly)
    - Cost limit enforcement
    - Usage analytics and warnings
    """
    
    def __init__(
        self,
        per_request_limit: Decimal = Decimal('0.10'),
        daily_limit: Decimal = Decimal('5.00'),
        weekly_limit: Decimal = Decimal('25.00'),
        monthly_limit: Decimal = Decimal('100.00'),
        warning_threshold: float = 0.8,
        reservation_timeout: int = 300
    ):
        """
        Initialize cost guard with budget limits.
        
        Args:
            per_request_limit: Maximum cost per individual request
            daily_limit: Daily spending limit
            weekly_limit: Weekly spending limit 
            monthly_limit: Monthly spending limit
            warning_threshold: Warning threshold as fraction (0.0-1.0)
            reservation_timeout: Reservation timeout in seconds
        """
        self.per_request_limit = validate_cost_limit(per_request_limit)
        self.daily_limit = validate_cost_limit(daily_limit)
        self.weekly_limit = validate_cost_limit(weekly_limit)
        self.monthly_limit = validate_cost_limit(monthly_limit)
        self.warning_threshold = max(0.0, min(1.0, warning_threshold))
        self.reservation_timeout = reservation_timeout
        
        # In-memory storage (in production, this would be backed by a database)
        self._usage_history: List[Dict[str, Any]] = []
        self._reservations: Dict[str, BudgetReservation] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"CostGuard initialized with per_request=${per_request_limit}, daily=${daily_limit}")
    
    async def check_and_reserve_budget(
        self,
        estimated_cost: Decimal,
        operation_type: str,
        model: str,
        user_id: Optional[str] = None
    ) -> BudgetCheck:
        """
        Check budget limits and reserve budget for operation.
        
        Args:
            estimated_cost: Estimated cost for the operation
            operation_type: Type of operation (e.g., 'second_opinion', 'compare')
            model: Model being used
            user_id: Optional user identifier
            
        Returns:
            BudgetCheck with reservation details
            
        Raises:
            CostLimitError: If cost limits would be exceeded
        """
        async with self._lock:
            # Clean up expired reservations
            await self._cleanup_expired_reservations()
            
            # Validate per-request limit
            if estimated_cost > self.per_request_limit:
                raise CostLimitError(
                    f"Estimated cost ${estimated_cost} exceeds per-request limit ${self.per_request_limit}",
                    estimated_cost,
                    self.per_request_limit,
                    "request"
                )
            
            # Check budget periods
            budget_checks = await self._check_all_budget_periods(estimated_cost)
            
            # Find any exceeded budgets
            exceeded_periods = [usage for usage in budget_checks if usage.status == BudgetStatus.EXCEEDED]
            if exceeded_periods:
                period = exceeded_periods[0].period.value
                raise BudgetExceededError(
                    f"Budget exceeded for {period} period: ${exceeded_periods[0].current_usage + estimated_cost} "
                    f"would exceed limit ${exceeded_periods[0].limit}",
                    estimated_cost,
                    exceeded_periods[0].limit,
                    period
                )
            
            # Create reservation
            reservation = BudgetReservation(
                reservation_id=str(uuid.uuid4()),
                estimated_cost=estimated_cost,
                operation_type=operation_type,
                timestamp=datetime.now(timezone.utc),
                model=model,
                user_id=user_id
            )
            
            self._reservations[reservation.reservation_id] = reservation
            
            # Calculate warnings
            warning_periods = [usage for usage in budget_checks if usage.status == BudgetStatus.WARNING]
            
            # Get current budget usage for the response
            daily_usage = next((u for u in budget_checks if u.period == BudgetPeriod.DAILY), None)
            monthly_usage = next((u for u in budget_checks if u.period == BudgetPeriod.MONTHLY), None)
            
            return BudgetCheck(
                approved=True,
                reservation_id=reservation.reservation_id,
                estimated_cost=estimated_cost,
                budget_remaining=min(usage.available for usage in budget_checks),
                daily_budget_remaining=daily_usage.available if daily_usage else Decimal('0'),
                monthly_budget_remaining=monthly_usage.available if monthly_usage else Decimal('0'),
                warning_message=self._generate_warning_message(warning_periods) if warning_periods else None
            )
    
    async def record_actual_cost(
        self,
        reservation_id: str,
        actual_cost: Decimal,
        model: str,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostAnalysis:
        """
        Record actual cost and release reservation.
        
        Args:
            reservation_id: Reservation ID from check_and_reserve_budget
            actual_cost: Actual cost incurred
            model: Model used
            operation_type: Type of operation
            metadata: Optional metadata
            
        Returns:
            Cost analysis with budget impact
            
        Raises:
            ValueError: If reservation not found
        """
        async with self._lock:
            reservation = self._reservations.get(reservation_id)
            if not reservation:
                raise ValueError(f"Reservation {reservation_id} not found")
            
            # Record usage
            usage_record = {
                "timestamp": datetime.now(timezone.utc),
                "actual_cost": actual_cost,
                "estimated_cost": reservation.estimated_cost,
                "model": model,
                "operation_type": operation_type,
                "user_id": reservation.user_id,
                "metadata": metadata or {}
            }
            self._usage_history.append(usage_record)
            
            # Remove reservation
            del self._reservations[reservation_id]
            
            # Calculate budget remaining after this transaction
            budget_usage = await self._get_budget_usage(BudgetPeriod.DAILY)
            
            return CostAnalysis(
                estimated_cost=reservation.estimated_cost,
                actual_cost=actual_cost,
                cost_per_token=self._calculate_cost_per_token(usage_record),
                budget_remaining=budget_usage.available
            )
    
    async def get_usage_summary(
        self,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        user_id: Optional[str] = None
    ) -> BudgetUsage:
        """
        Get usage summary for a specific period.
        
        Args:
            period: Budget period to analyze
            user_id: Optional user filter
            
        Returns:
            Budget usage summary
        """
        async with self._lock:
            await self._cleanup_expired_reservations()
            return await self._get_budget_usage(period, user_id)
    
    async def get_detailed_analytics(
        self,
        days: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed cost analytics and trends.
        
        Args:
            days: Number of days to analyze
            user_id: Optional user filter
            
        Returns:
            Detailed analytics dictionary
        """
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Filter relevant usage records
            records = [
                r for r in self._usage_history
                if r["timestamp"] >= cutoff and (not user_id or r.get("user_id") == user_id)
            ]
            
            if not records:
                return {
                    "total_cost": Decimal('0'),
                    "total_requests": 0,
                    "average_cost_per_request": Decimal('0'),
                    "models_used": [],
                    "operations_breakdown": {},
                    "daily_spending": [],
                    "cost_trends": {}
                }
            
            total_cost = sum(r["actual_cost"] for r in records)
            total_requests = len(records)
            
            # Model usage breakdown
            model_costs = {}
            for record in records:
                model = record["model"]
                model_costs[model] = model_costs.get(model, Decimal('0')) + record["actual_cost"]
            
            # Operations breakdown
            operation_costs = {}
            for record in records:
                op = record["operation_type"]
                operation_costs[op] = operation_costs.get(op, Decimal('0')) + record["actual_cost"]
            
            # Daily spending trend
            daily_spending = {}
            for record in records:
                day = record["timestamp"].date()
                daily_spending[day] = daily_spending.get(day, Decimal('0')) + record["actual_cost"]
            
            return {
                "total_cost": total_cost,
                "total_requests": total_requests,
                "average_cost_per_request": total_cost / total_requests if total_requests > 0 else Decimal('0'),
                "models_used": list(model_costs.keys()),
                "model_costs": {k: float(v) for k, v in model_costs.items()},
                "operations_breakdown": {k: float(v) for k, v in operation_costs.items()},
                "daily_spending": {str(k): float(v) for k, v in daily_spending.items()},
                "period_days": days,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def estimate_request_cost(
        self,
        request: ModelRequest,
        model_pricing: Optional[Dict[str, Tuple[Decimal, Decimal]]] = None
    ) -> Decimal:
        """
        Estimate cost for a model request using dynamic pricing.
        
        Args:
            request: Model request to estimate
            model_pricing: Optional legacy pricing dict (deprecated, uses pricing manager by default)
            
        Returns:
            Estimated cost
        """
        # Estimate input tokens from messages
        input_tokens = 0
        for message in request.messages:
            input_tokens += self._estimate_tokens(message.content)
        
        if request.system_prompt:
            input_tokens += self._estimate_tokens(request.system_prompt)
        
        # Estimate output tokens (conservative estimate)
        max_tokens = request.max_tokens or 1000
        estimated_output_tokens = min(max_tokens, max(100, input_tokens // 2))
        
        # Use pricing manager for dynamic cost calculation
        if model_pricing is None:
            pricing_manager = get_pricing_manager()
            cost, source = pricing_manager.estimate_cost(request.model, input_tokens, estimated_output_tokens)
            logger.debug(f"Estimated cost for {request.model}: ${cost} (source: {source})")
            return cost
        else:
            # Legacy pricing mode (for backward compatibility)
            logger.warning("Using legacy pricing mode, consider migrating to pricing manager")
            
            if request.model not in model_pricing:
                # Conservative fallback estimate
                return Decimal('0.10')
            
            input_cost_per_1k, output_cost_per_1k = model_pricing[request.model]
            
            # Calculate costs
            input_cost = Decimal(input_tokens) * input_cost_per_1k / 1000
            output_cost = Decimal(estimated_output_tokens) * output_cost_per_1k / 1000
            
            return input_cost + output_cost
    
    async def _check_all_budget_periods(self, additional_cost: Decimal) -> List[BudgetUsage]:
        """Check all budget periods with additional cost."""
        periods = [BudgetPeriod.DAILY, BudgetPeriod.WEEKLY, BudgetPeriod.MONTHLY]
        results = []
        
        for period in periods:
            usage = await self._get_budget_usage(period)
            # Simulate adding the cost
            usage.current_usage += additional_cost
            results.append(usage)
        
        return results
    
    async def _get_budget_usage(
        self,
        period: BudgetPeriod,
        user_id: Optional[str] = None
    ) -> BudgetUsage:
        """Get current budget usage for a period."""
        now = datetime.now(timezone.utc)
        
        # Calculate period boundaries
        if period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            limit = self.daily_limit
        elif period == BudgetPeriod.WEEKLY:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
            limit = self.weekly_limit
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
            limit = self.monthly_limit
        else:
            # Total usage
            start = datetime.min.replace(tzinfo=timezone.utc)
            end = datetime.max.replace(tzinfo=timezone.utc)
            limit = self.monthly_limit * 12  # Arbitrary large limit
        
        # Calculate current usage
        current_usage = sum(
            r["actual_cost"] for r in self._usage_history
            if start <= r["timestamp"] < end and (not user_id or r.get("user_id") == user_id)
        )
        
        # Calculate reserved amount
        reserved = sum(
            r.estimated_cost for r in self._reservations.values()
            if not r.is_expired(self.reservation_timeout) and (not user_id or r.user_id == user_id)
        )
        
        return BudgetUsage(
            period=period,
            current_usage=Decimal(str(current_usage)),
            limit=limit,
            reserved=Decimal(str(reserved)),
            period_start=start,
            period_end=end
        )
    
    async def _cleanup_expired_reservations(self):
        """Remove expired reservations."""
        expired_ids = [
            rid for rid, reservation in self._reservations.items()
            if reservation.is_expired(self.reservation_timeout)
        ]
        
        for rid in expired_ids:
            del self._reservations[rid]
            logger.debug(f"Cleaned up expired reservation {rid}")
    
    def _generate_warning_message(self, warning_periods: List[BudgetUsage]) -> str:
        """Generate warning message for budget periods approaching limits."""
        warnings = []
        for usage in warning_periods:
            percentage = int(usage.utilization * 100)
            warnings.append(f"{usage.period.value} budget at {percentage}% (${usage.current_usage}/${usage.limit})")
        
        return "Budget warning: " + "; ".join(warnings)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)."""
        return max(1, len(text) // 4)
    
    def _calculate_cost_per_token(self, usage_record: Dict[str, Any]) -> Decimal:
        """Calculate cost per token from usage record."""
        # This would be more accurate with actual token counts
        estimated_tokens = self._estimate_tokens(str(usage_record.get("metadata", {})))
        if estimated_tokens > 0:
            return usage_record["actual_cost"] / estimated_tokens
        return Decimal('0')


# Global cost guard instance
_cost_guard: Optional[CostGuard] = None


def get_cost_guard() -> CostGuard:
    """Get global cost guard instance."""
    global _cost_guard
    if _cost_guard is None:
        _cost_guard = CostGuard()
    return _cost_guard


def set_cost_guard(cost_guard: Optional[CostGuard]) -> None:
    """Set global cost guard instance."""
    global _cost_guard
    _cost_guard = cost_guard


async def check_budget(
    estimated_cost: Decimal,
    operation_type: str,
    model: str,
    user_id: Optional[str] = None
) -> BudgetCheck:
    """Global function to check and reserve budget."""
    guard = get_cost_guard()
    return await guard.check_and_reserve_budget(estimated_cost, operation_type, model, user_id)


async def record_cost(
    reservation_id: str,
    actual_cost: Decimal,
    model: str,
    operation_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> CostAnalysis:
    """Global function to record actual cost."""
    guard = get_cost_guard()
    return await guard.record_actual_cost(reservation_id, actual_cost, model, operation_type, metadata)