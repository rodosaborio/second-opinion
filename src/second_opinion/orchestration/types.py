"""
Type definitions for conversation orchestration.

This module defines data classes and interfaces used by the ConversationOrchestrator
for managing conversation storage across different interfaces.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass
class ConversationResult:
    """
    Result of conversation storage operation.

    This data class provides confirmation and metadata about stored conversations,
    including storage status and cost tracking information.
    """

    conversation_id: str | None
    """Database ID of stored conversation, None if storage disabled/failed"""

    session_id: str
    """Session ID for the conversation (CLI or MCP format)"""

    total_cost: Decimal
    """Total cost of the conversation including all model calls"""

    total_tokens: int
    """Total tokens used across all responses"""

    responses_stored: int
    """Number of responses successfully stored"""

    storage_enabled: bool
    """Whether storage was enabled for this conversation"""

    storage_error: str | None = None
    """Error message if storage failed, None if successful"""


@dataclass
class StorageContext:
    """
    Context information for conversation storage.

    This provides metadata about the storage operation including
    interface type, tool name, and optional context.
    """

    interface_type: str  # "cli" or "mcp"
    """Interface that generated the conversation (cli or mcp)"""

    tool_name: str
    """Name of the tool that generated the conversation"""

    session_id: str | None = None
    """Session ID for conversation grouping"""

    context: str | None = None
    """Additional context about the conversation"""

    save_conversation: bool = True
    """Whether to save this conversation (feature flag)"""

    user_metadata: dict[str, Any] | None = None
    """Optional metadata provided by user/interface"""
