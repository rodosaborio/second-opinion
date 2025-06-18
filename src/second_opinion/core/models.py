"""
Core Pydantic models for Second Opinion.

This module contains all the data models used throughout the application,
providing type safety, validation, and serialization capabilities.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class SecurityContext(str, Enum):
    """Security context for input validation."""
    USER_PROMPT = "user_prompt"
    SYSTEM_PROMPT = "system_prompt"
    API_REQUEST = "api_request"
    CONFIGURATION = "configuration"


class ModelTier(str, Enum):
    """Model capability tiers for routing and recommendations."""
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    PREMIUM = "premium"
    REASONING = "reasoning"


class RecommendationType(str, Enum):
    """Types of model recommendations."""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    MAINTAIN = "maintain"
    ALTERNATIVE = "alternative"


class TaskComplexity(str, Enum):
    """Task complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class Message(BaseModel):
    """Standardized message format for model interactions."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        valid_roles = {'user', 'assistant', 'system'}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > 100000:  # 100KB limit
            raise ValueError("Message content exceeds maximum length")
        return v.strip()


class TokenUsage(BaseModel):
    """Token usage information from model APIs."""
    input_tokens: int = Field(..., ge=0, description="Number of input tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens") 
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    
    @model_validator(mode='after')
    def validate_total(self):
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Total tokens must equal input_tokens + output_tokens")
        return self


class PricingInfo(BaseModel):
    """Pricing information for a specific model."""
    model: str = Field(..., description="Model identifier")
    input_cost_per_1k: Decimal = Field(..., ge=0, description="Cost per 1K input tokens")
    output_cost_per_1k: Decimal = Field(..., ge=0, description="Cost per 1K output tokens")
    currency: str = Field(default="USD", description="Currency for pricing")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When pricing was last updated")


class ModelInfo(BaseModel):
    """Information about a model's capabilities."""
    model: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    tier: ModelTier = Field(..., description="Model capability tier")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens supported")
    supports_system_messages: bool = Field(default=True, description="Whether model supports system messages")
    supports_temperature: bool = Field(default=True, description="Whether model supports temperature control")
    context_window: Optional[int] = Field(None, ge=1, description="Context window size")
    pricing: Optional[PricingInfo] = Field(None, description="Pricing information")


class ModelRequest(BaseModel):
    """Standardized request format for all model providers."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., min_length=1, description="Conversation messages")
    max_tokens: Optional[int] = Field(None, ge=1, le=32000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    system_prompt: Optional[str] = Field(None, description="System prompt for models that support it")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and v > 32000:
            raise ValueError("max_tokens cannot exceed 32000")
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is not None and len(v) > 10000:  # 10KB limit for system prompts
            raise ValueError("System prompt exceeds maximum length")
        return v


class ModelResponse(BaseModel):
    """Standardized response format with cost tracking."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    usage: TokenUsage = Field(..., description="Token usage information")
    cost_estimate: Decimal = Field(..., ge=0, description="Estimated cost in USD")
    provider: str = Field(..., description="Model provider")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class CostAnalysis(BaseModel):
    """Cost analysis for model operations."""
    estimated_cost: Decimal = Field(..., ge=0, description="Estimated cost before request")
    actual_cost: Decimal = Field(..., ge=0, description="Actual cost after request")
    cost_per_token: Decimal = Field(..., ge=0, description="Cost per token for this operation")
    budget_remaining: Decimal = Field(..., ge=0, description="Remaining budget after operation")
    
    @property
    def cost_difference(self) -> Decimal:
        """Computed difference between estimated and actual cost."""
        return self.actual_cost - self.estimated_cost


class EvaluationCriteria(BaseModel):
    """Criteria for evaluating and comparing responses."""
    accuracy_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for accuracy assessment")
    completeness_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for completeness assessment")
    clarity_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for clarity assessment")
    usefulness_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for usefulness assessment")
    
    @model_validator(mode='after')
    def validate_weights_sum(self):
        total_weight = (
            self.accuracy_weight + 
            self.completeness_weight + 
            self.clarity_weight + 
            self.usefulness_weight
        )
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError("Evaluation criteria weights must sum to 1.0")
        return self


class ComparisonResult(BaseModel):
    """Results from comparing two model responses."""
    primary_response: str = Field(..., description="Primary model response")
    comparison_response: str = Field(..., description="Comparison model response")
    primary_model: str = Field(..., description="Primary model identifier")
    comparison_model: str = Field(..., description="Comparison model identifier")
    
    accuracy_score: float = Field(..., ge=0.0, le=10.0, description="Accuracy score (0-10)")
    completeness_score: float = Field(..., ge=0.0, le=10.0, description="Completeness score (0-10)")
    clarity_score: float = Field(..., ge=0.0, le=10.0, description="Clarity score (0-10)")
    usefulness_score: float = Field(..., ge=0.0, le=10.0, description="Usefulness score (0-10)")
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall comparison score")
    
    winner: str = Field(..., description="Which response is better (primary/comparison/tie)")
    reasoning: str = Field(..., description="Explanation of the comparison results")
    cost_analysis: CostAnalysis = Field(..., description="Cost analysis for the comparison")
    
    @field_validator('winner')
    @classmethod
    def validate_winner(cls, v):
        valid_winners = {'primary', 'comparison', 'tie'}
        if v not in valid_winners:
            raise ValueError(f"Winner must be one of {valid_winners}")
        return v


class RecommendationResult(BaseModel):
    """Results from model tier recommendation analysis."""
    current_model: str = Field(..., description="Current model being evaluated")
    recommended_action: RecommendationType = Field(..., description="Recommended action")
    recommended_model: Optional[str] = Field(None, description="Recommended model (if applicable)")
    task_complexity: TaskComplexity = Field(..., description="Assessed task complexity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    
    current_quality_score: float = Field(..., ge=0.0, le=10.0, description="Quality score of current response")
    expected_improvement: Optional[float] = Field(None, ge=0.0, le=10.0, description="Expected quality improvement")
    cost_impact: Decimal = Field(..., description="Cost impact of recommendation (positive = more expensive)")
    
    reasoning: str = Field(..., description="Explanation of the recommendation")
    cost_analysis: CostAnalysis = Field(..., description="Cost analysis for the recommendation")


class Conversation(BaseModel):
    """A conversation session with associated metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique conversation identifier")
    primary_model: str = Field(..., description="Primary model for this conversation")
    tool_used: str = Field(..., description="Tool used in this conversation")
    messages: List[Message] = Field(..., description="Conversation messages")
    
    cost_total: Decimal = Field(default=Decimal('0'), ge=0, description="Total cost for conversation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Conversation creation time")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update time")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional conversation metadata")


class UsageStats(BaseModel):
    """Usage statistics for analytics."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    model: str = Field(..., description="Model identifier") 
    tool: str = Field(..., description="Tool name")
    request_count: int = Field(..., ge=0, description="Number of requests")
    total_cost: Decimal = Field(..., ge=0, description="Total cost for the period")
    avg_cost_per_request: Decimal = Field(..., ge=0, description="Average cost per request")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")


class BudgetCheck(BaseModel):
    """Result of budget validation check."""
    approved: bool = Field(..., description="Whether the operation is approved")
    reservation_id: str = Field(default_factory=lambda: str(uuid4()), description="Budget reservation identifier")
    estimated_cost: Decimal = Field(..., ge=0, description="Estimated cost for operation")
    budget_remaining: Decimal = Field(..., ge=0, description="Budget remaining after operation")
    warning_message: Optional[str] = Field(None, description="Warning message if applicable")
    
    daily_budget_remaining: Decimal = Field(..., ge=0, description="Daily budget remaining")
    monthly_budget_remaining: Decimal = Field(..., ge=0, description="Monthly budget remaining")


class ToolResponse(BaseModel):
    """Standardized response format for MCP tools."""
    success: bool = Field(..., description="Whether the tool operation succeeded")
    result: Dict[str, Any] = Field(..., description="Tool operation results")
    cost: Decimal = Field(..., ge=0, description="Cost of the operation")
    model_used: str = Field(..., description="Primary model used")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RequestContext(BaseModel):
    """Context information for processing requests."""
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Session identifier")
    primary_model: Optional[str] = Field(None, description="Primary model for this session")
    user_id: Optional[str] = Field(None, description="User identifier (if available)")
    
    daily_budget_used: Decimal = Field(default=Decimal('0'), ge=0, description="Daily budget already used")
    monthly_budget_used: Decimal = Field(default=Decimal('0'), ge=0, description="Monthly budget already used")
    
    security_context: SecurityContext = Field(default=SecurityContext.USER_PROMPT, description="Security validation context")
    request_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Request timestamp")