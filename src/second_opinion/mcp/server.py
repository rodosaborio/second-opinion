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
    description="Compare AI responses across models for alternative perspectives and quality assessment"
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
    Compare AI responses across models for alternative perspectives and quality assessment.
    
    This tool helps optimize AI model usage by comparing responses from different models,
    providing quality assessments, and suggesting cost-effective alternatives. It supports
    response reuse to minimize API costs when you already have a primary model response.
    
    Args:
        prompt: The question or task to analyze and compare across models
        primary_model: The model name that generated the original response 
                      (e.g., "anthropic/claude-3-5-sonnet"). If not provided, will use
                      the most frequently used model from session history or default.
        primary_response: The original response to compare against. When provided,
                         saves API costs by skipping the primary model call. This is
                         especially useful when you already have a response from another client.
        context: Additional context about the task or domain to improve comparison quality.
                For example: "This is for academic research" or "Technical documentation".
        comparison_models: Specific models to compare against the primary model.
                          Can be a list like ["openai/gpt-4o", "google/gemini-pro"].
                          If not provided, models will be auto-selected based on the
                          primary model tier and task complexity.
        cost_limit: Maximum cost limit for this operation in USD (e.g., 0.25).
                   If not provided, uses the configured default limit.
    
    Returns:
        A formatted comparison report showing response quality assessment, cost analysis,
        task complexity evaluation, and actionable insights for model selection.
        
    Example Usage:
        # Basic comparison with auto-selected models
        result = await second_opinion(
            prompt="What's the capital of France?",
            primary_model="anthropic/claude-3-5-sonnet"
        )
        
        # Cost-efficient comparison with existing response
        result = await second_opinion(
            prompt="Explain quantum computing",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Quantum computing is...",  # Saves API call
            comparison_models=["openai/gpt-4o", "google/gemini-pro"],
            context="For technical documentation"
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