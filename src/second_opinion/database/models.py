"""
SQLAlchemy models for conversation storage.
"""

from uuid import uuid4

from sqlalchemy import (
    DECIMAL,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Conversation(Base):
    """Main conversation record."""

    __tablename__ = "conversations"

    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, nullable=True, index=True)  # MCP session ID

    # Interface tracking
    interface_type = Column(String, nullable=False)  # 'cli' or 'mcp'
    tool_name = Column(String, nullable=True)  # For MCP tools

    # Content (encrypted)
    user_prompt_encrypted = Column(Text, nullable=False)
    context_encrypted = Column(Text, nullable=True)  # Additional context

    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Cost tracking
    total_cost = Column(DECIMAL(10, 6), nullable=False, default=0)
    estimated_cost = Column(DECIMAL(10, 6), nullable=True)

    # Task classification
    task_complexity = Column(String, nullable=True)
    domain_classification = Column(String, nullable=True)

    # Privacy and security
    encryption_key_id = Column(String, nullable=False)
    is_sensitive = Column(Boolean, default=False)

    # Relationships
    responses: Mapped[list["Response"]] = relationship(
        "Response", back_populates="conversation", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_conversations_created_at", "created_at"),
        Index("idx_conversations_interface_tool", "interface_type", "tool_name"),
        Index("idx_conversations_session", "session_id"),
        Index("idx_conversations_cost", "total_cost"),
        Index("idx_conversations_complexity", "task_complexity"),
    )


class Response(Base):
    """Model response record."""

    __tablename__ = "responses"

    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)

    # Response metadata
    model = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    response_type = Column(String, nullable=False)  # 'primary' or 'comparison'
    response_order = Column(Integer, nullable=False, default=0)

    # Content (encrypted)
    content_encrypted = Column(Text, nullable=False)
    metadata_encrypted = Column(Text, nullable=True)  # JSON metadata

    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Cost tracking
    cost = Column(DECIMAL(10, 6), nullable=False)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Quality metrics (if available)
    evaluation_score = Column(DECIMAL(4, 2), nullable=True)
    evaluation_reasoning_encrypted = Column(Text, nullable=True)

    # Privacy and security
    encryption_key_id = Column(String, nullable=False)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="responses"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_responses_conversation", "conversation_id"),
        Index("idx_responses_model", "model"),
        Index("idx_responses_provider", "provider"),
        Index("idx_responses_type", "response_type"),
        Index("idx_responses_created_at", "created_at"),
        Index("idx_responses_cost", "cost"),
    )


class ConversationAnalytics(Base):
    """Aggregated analytics for performance."""

    __tablename__ = "conversation_analytics"

    # Time-based aggregation
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    date = Column(DateTime, nullable=False)
    period_type = Column(String, nullable=False)  # 'daily', 'weekly', 'monthly'

    # Interface breakdown
    cli_conversations = Column(Integer, default=0)
    mcp_conversations = Column(Integer, default=0)
    total_conversations = Column(Integer, default=0)

    # Cost analytics
    total_cost = Column(DECIMAL(10, 6), default=0)
    average_cost_per_conversation = Column(DECIMAL(10, 6), default=0)
    cli_total_cost = Column(DECIMAL(10, 6), default=0)
    mcp_total_cost = Column(DECIMAL(10, 6), default=0)

    # Model usage analytics
    most_used_model = Column(String, nullable=True)
    most_expensive_model = Column(String, nullable=True)
    model_diversity_count = Column(Integer, default=0)

    # Performance metrics
    average_response_time_ms = Column(Integer, nullable=True)
    total_tokens = Column(Integer, default=0)
    average_tokens_per_conversation = Column(Integer, default=0)

    # Task complexity distribution
    simple_task_count = Column(Integer, default=0)
    moderate_task_count = Column(Integer, default=0)
    complex_task_count = Column(Integer, default=0)

    # Quality metrics
    average_evaluation_score = Column(DECIMAL(4, 2), nullable=True)
    conversations_with_evaluations = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Indexes for analytics queries
    __table_args__ = (
        Index("idx_analytics_date_period", "date", "period_type"),
        Index("idx_analytics_created_at", "created_at"),
    )
