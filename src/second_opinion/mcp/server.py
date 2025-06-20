"""
FastMCP server implementation for Second Opinion.

This module provides the main MCP server using FastMCP framework,
integrating with the existing evaluation engine, cost tracking, and
OpenRouter client infrastructure.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastmcp import FastMCP

from ..config.settings import get_settings
from ..utils.cost_tracking import get_cost_guard
from ..utils.pricing import get_pricing_manager
from .session import MCPSession

# Configure logging
logger = logging.getLogger(__name__)

# Global session storage (in production, this would be more sophisticated)
_sessions: Dict[str, MCPSession] = {}


@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """
    Manage MCP server lifecycle with resource initialization and cleanup.
    
    This context manager handles:
    - Configuration loading and validation
    - Cost tracking system initialization
    - Pricing manager setup
    - Session cleanup on shutdown
    """
    logger.info("Starting Second Opinion MCP server...")
    
    try:
        # Initialize core systems
        settings = get_settings()
        cost_guard = get_cost_guard()
        pricing_manager = get_pricing_manager()
        
        # Validate essential configuration
        if not settings.openrouter_api_key:
            logger.warning("No OpenRouter API key configured - tool functionality may be limited")
        
        # Initialize pricing data
        try:
            await pricing_manager.load_pricing_data()
            logger.info(f"Pricing manager initialized with {len(pricing_manager._pricing_data)} models")
        except Exception as e:
            logger.warning(f"Failed to load pricing data: {e}")
        
        # Create server context
        context = {
            "settings": settings,
            "cost_guard": cost_guard,
            "pricing_manager": pricing_manager,
            "sessions": _sessions,
        }
        
        logger.info("Second Opinion MCP server started successfully")
        yield context
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise
    finally:
        # Cleanup sessions on shutdown
        logger.info("Shutting down Second Opinion MCP server...")
        _sessions.clear()
        logger.info("Server shutdown complete")


# Create FastMCP server instance with metadata
mcp = FastMCP(
    name="Second Opinion",
    lifespan=mcp_lifespan,
)

# Import and register tools after server creation
from .tools.second_opinion import second_opinion_tool


# Register the core second_opinion tool
@mcp.tool(
    name="second_opinion",
    description="Get a second opinion on an AI response by comparing it against alternative models for quality assessment and cost optimization"
)
async def second_opinion(
    prompt: str,
    primary_model: str | None = None,
    primary_response: str | None = None,
    context: str | None = None,
    comparison_models: list[str] | None = None,
    cost_limit: float | None = None,
) -> str:
    """
    Get a second opinion on an AI response by comparing it against alternative models.
    
    This tool is designed for natural conversation flow where an AI client has already
    provided a response and wants to evaluate it against alternatives. It helps optimize
    AI model usage by providing quality assessments, cost optimization recommendations,
    and suggestions for when to use local vs cloud models.
    
    NATURAL USAGE PATTERN:
    1. User asks: "Write a Python function to calculate fibonacci"
    2. AI responds: <provides code>
    3. User asks: "Can you get a second opinion on that?"
    4. AI calls this tool with its response for comparison
    
    Args:
        prompt: The original question or task that was asked
        primary_model: The model that provided the original response. Use OpenRouter format:
                      - Claude Desktop: "anthropic/claude-3-5-sonnet"
                      - ChatGPT: "openai/gpt-4o" or "openai/gpt-4o-mini"
                      - Gemini: "google/gemini-pro-1.5"
                      - Local models: "qwen3-4b-mlx", "codestral-22b-v0.1", etc.
        primary_response: The response to evaluate (RECOMMENDED). When provided, saves
                         costs and evaluates the actual response the user saw.
        context: Additional context about the task domain for better comparison quality.
                For example: "coding task", "academic research", "creative writing".
        comparison_models: Specific models to compare against. If not provided, will
                          auto-select alternatives including cost-effective local options.
        cost_limit: Maximum cost limit for this operation in USD (default: $0.25).
    
    Returns:
        A second opinion report with quality assessment, cost optimization recommendations,
        and decision guidance for model selection.
        
    RECOMMENDED USAGE (Natural Conversation Flow):
        # After providing a response to user, get second opinion
        result = await second_opinion(
            prompt="Write a Python function to calculate fibonacci",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="def fibonacci(n):\n    if n <= 1:\n        return n...",
            context="coding task"
        )
    """
    # Get or create session for this request
    session = get_mcp_session()
    
    # Update session activity
    session.update_activity()
    
    try:
        # Call the tool implementation
        result = await second_opinion_tool(
            prompt=prompt,
            primary_model=primary_model,
            primary_response=primary_response,
            context=context,
            comparison_models=comparison_models,
            cost_limit=cost_limit,
        )
        
        # Add to conversation context
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt=prompt,
            primary_model=primary_model or "unknown",
            comparison_models=comparison_models or [],
            result_summary="Comparison completed successfully"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        # Still add to context for debugging
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt=prompt,
            primary_model=primary_model or "unknown",
            comparison_models=comparison_models or [],
            result_summary=f"Error: {str(e)}"
        )
        raise


def get_mcp_session(session_id: str | None = None) -> MCPSession:
    """
    Get or create an MCP session for the current request.
    
    Args:
        session_id: Optional session identifier. If None, creates a new session.
        
    Returns:
        MCPSession instance for tracking costs and context
    """
    if session_id is None:
        # Create new session
        session = MCPSession()
        _sessions[session.session_id] = session
        logger.debug(f"Created new MCP session: {session.session_id}")
        return session
    
    # Get existing session or create new one
    if session_id in _sessions:
        logger.debug(f"Retrieved existing MCP session: {session_id}")
        return _sessions[session_id]
    
    # Session not found, create new one
    session = MCPSession(session_id=session_id)
    _sessions[session_id] = session
    logger.debug(f"Created MCP session with provided ID: {session_id}")
    return session


def cleanup_sessions(max_sessions: int = 100) -> None:
    """
    Clean up old sessions to prevent memory leaks.
    
    Args:
        max_sessions: Maximum number of sessions to keep
    """
    if len(_sessions) > max_sessions:
        # Remove oldest sessions (simple FIFO cleanup)
        sessions_to_remove = len(_sessions) - max_sessions
        session_ids = list(_sessions.keys())
        
        for session_id in session_ids[:sessions_to_remove]:
            del _sessions[session_id]
            logger.debug(f"Cleaned up old session: {session_id}")


if __name__ == "__main__":
    """
    Run the MCP server in standalone mode.
    
    This supports the Claude Desktop configuration:
    {
        "command": "uv",
        "args": [
            "--directory", "/Users/rsc/second-opinion",
            "run", "python", "-m", "second_opinion.mcp.server"
        ]
    }
    """
    # Configure logging for standalone mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Second Opinion MCP server in standalone mode")
    mcp.run()