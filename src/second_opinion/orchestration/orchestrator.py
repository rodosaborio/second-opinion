"""
ConversationOrchestrator for unified conversation storage management.

This module provides the core ConversationOrchestrator class that coordinates
conversation storage across CLI and MCP interfaces while integrating with
existing cost tracking and session management systems.
"""

import logging
import threading
from decimal import Decimal
from typing import Any, cast

from ..config.settings import get_settings
from ..core.models import ModelResponse
from ..database.store import ConversationStore
from ..utils.cost_tracking import get_cost_guard
from .session_manager import generate_cli_session_id, generate_mcp_session_id
from .types import ConversationResult, StorageContext

logger = logging.getLogger(__name__)

# Global orchestrator instance and lock for thread-safe singleton
_global_orchestrator: "ConversationOrchestrator | None" = None
_orchestrator_lock = threading.Lock()


class ConversationOrchestrator:
    """
    Unified conversation storage orchestrator for CLI and MCP interfaces.

    This class coordinates between ConversationStore, CostGuard, and session management
    while maintaining clear separation of concerns and providing optional storage.
    """

    def __init__(self, conversation_store: ConversationStore | None = None):
        """
        Initialize conversation orchestrator.

        Args:
            conversation_store: Optional conversation store instance.
                              If None, will create default instance.
        """
        self.conversation_store = conversation_store or ConversationStore()
        self.cost_guard = get_cost_guard()
        self.settings = get_settings()

        logger.debug("ConversationOrchestrator initialized")

    async def handle_interaction(
        self,
        prompt: str,
        responses: list[ModelResponse],
        storage_context: StorageContext,
        evaluation_result: dict[str, Any] | None = None,
    ) -> ConversationResult:
        """
        Handle complete conversation storage workflow.

        This method coordinates conversation storage while providing detailed
        result information and handling errors gracefully.

        Args:
            prompt: Original user prompt
            responses: List of model responses (primary + comparisons)
            storage_context: Context about the storage operation
            evaluation_result: Optional evaluation results for storage

        Returns:
            ConversationResult with storage confirmation and metadata
        """
        logger.info(
            f"Handling conversation storage: tool={storage_context.tool_name}, "
            f"interface={storage_context.interface_type}, "
            f"responses={len(responses)}, "
            f"enabled={storage_context.save_conversation}"
        )

        # Generate session ID if not provided
        session_id = storage_context.session_id
        if not session_id:
            if storage_context.interface_type == "cli":
                session_id = generate_cli_session_id()
            else:  # mcp
                session_id = generate_mcp_session_id()

        # Calculate total cost and tokens
        total_cost = sum((r.cost_estimate for r in responses), Decimal("0"))
        total_tokens_input = sum(r.usage.input_tokens for r in responses)
        total_tokens_output = sum(r.usage.output_tokens for r in responses)
        total_tokens = total_tokens_input + total_tokens_output

        # Create result object
        conversation_result = ConversationResult(
            conversation_id=None,
            session_id=session_id,
            total_cost=total_cost,
            total_tokens=total_tokens,
            responses_stored=len(responses),
            storage_enabled=storage_context.save_conversation,
        )

        # Early return if storage disabled
        if not storage_context.save_conversation:
            logger.debug("Conversation storage disabled, skipping storage")
            return conversation_result

        # Attempt to store conversation
        try:
            # Separate primary and comparison responses
            primary_response = responses[0] if responses else None
            comparison_responses = responses[1:] if len(responses) > 1 else []

            if not primary_response:
                logger.warning("No primary response found, skipping storage")
                conversation_result.storage_error = "No primary response to store"
                return conversation_result

            # Store conversation using existing ConversationStore interface
            conversation_id = await self.conversation_store.store_conversation(
                user_prompt=prompt,
                primary_response=primary_response,
                comparison_responses=comparison_responses,
                evaluation_result=evaluation_result,
                interface_type=storage_context.interface_type,
                session_id=session_id,
                tool_name=storage_context.tool_name,
                context=storage_context.context,
                # Additional metadata can be added here as needed
            )

            conversation_result.conversation_id = conversation_id
            logger.info(
                f"Successfully stored conversation {conversation_id} "
                f"(session: {session_id}, cost: ${total_cost:.4f})"
            )

        except Exception as e:
            # Storage failure is non-fatal - log warning and continue
            error_msg = f"Conversation storage failed: {str(e)}"
            logger.warning(error_msg)
            conversation_result.storage_error = error_msg

        return conversation_result

    async def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics for monitoring and debugging.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # This could be extended to call ConversationStore analytics
            return {
                "storage_enabled": True,
                "orchestrator_active": True,
                "store_type": type(self.conversation_store).__name__,
            }
        except Exception as e:
            logger.warning(f"Failed to get storage stats: {e}")
            return {
                "storage_enabled": False,
                "error": str(e),
            }


def get_conversation_orchestrator() -> ConversationOrchestrator:
    """
    Get the global ConversationOrchestrator instance.

    Uses a thread-safe, double-checked locking pattern for a consistent
    and performant singleton instance across the application.

    Returns:
        Global ConversationOrchestrator instance
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        with _orchestrator_lock:
            # Second check ensures that another thread didn't initialize
            # the instance while the current thread was waiting for the lock.
            if _global_orchestrator is None:
                _global_orchestrator = ConversationOrchestrator()
                logger.debug("Created global ConversationOrchestrator instance")
    return cast(ConversationOrchestrator, _global_orchestrator)


def reset_conversation_orchestrator() -> None:
    """
    Reset the global orchestrator instance.

    This is primarily used for testing to ensure clean state
    between test runs.
    """
    global _global_orchestrator
    with _orchestrator_lock:
        _global_orchestrator = None
        logger.debug("Reset global ConversationOrchestrator instance")
