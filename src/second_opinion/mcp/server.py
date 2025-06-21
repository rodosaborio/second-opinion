"""
FastMCP server implementation for Second Opinion.

This module provides the main MCP server using FastMCP framework,
integrating with the existing evaluation engine, cost tracking, and
OpenRouter client infrastructure.
"""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from pydantic import Field

from ..config.settings import get_settings
from ..utils.cost_tracking import get_cost_guard
from ..utils.pricing import get_pricing_manager
from .session import MCPSession


# Configure comprehensive logging for MCP server debugging
def setup_mcp_logging(debug: bool = False) -> logging.Logger:
    """Set up comprehensive logging for MCP server debugging."""
    logger = logging.getLogger(__name__)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set logging level
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Create formatter with more detail
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [MCP] %(message)s'
    )

    # Add stderr handler (MCP servers should log to stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

# Set up logger
logger = setup_mcp_logging(debug=True)  # Enable debug for troubleshooting

def setup_python_path():
    """Set up Python path for MCP server context."""
    import pathlib

    # Get the project root (should be /Users/rsc/second-opinion)
    current_file = pathlib.Path(__file__)
    project_root = current_file.parent.parent.parent.parent  # Go up to project root
    src_path = project_root / "src"

    logger.info(f"Current file: {current_file}")
    logger.info(f"Calculated project root: {project_root}")
    logger.info(f"Calculated src path: {src_path}")

    # Add src directory to Python path if not already there
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
        logger.info(f"Added to sys.path: {src_path_str}")
    else:
        logger.info(f"Already in sys.path: {src_path_str}")

    # Also add project root
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        logger.info(f"Added to sys.path: {project_root_str}")
    else:
        logger.info(f"Already in sys.path: {project_root_str}")

def log_environment_info():
    """Log comprehensive environment information for debugging."""
    logger.info("=== MCP Server Environment Information ===")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"sys.path entries: {len(sys.path)}")
    for i, path in enumerate(sys.path):
        logger.info(f"  [{i}] {path}")

    # Log key environment variables
    env_vars = ['PATH', 'UV_PROJECT_ENVIRONMENT', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        logger.info(f"{var}: {value}")

    logger.info("=== End Environment Information ===")

def test_critical_imports():
    """Test importing critical modules and log results."""
    logger.info("=== Testing Critical Imports ===")

    critical_modules = [
        'second_opinion',
        'second_opinion.mcp',
        'second_opinion.mcp.tools',
        'second_opinion.mcp.tools.second_opinion',
        'second_opinion.mcp.tools.should_downgrade',
        'second_opinion.mcp.tools.should_upgrade',
        'second_opinion.core.models',
        'second_opinion.clients',
        'second_opinion.utils.cost_tracking',
    ]

    successful_imports = []
    failed_imports = []

    for module_name in critical_modules:
        try:
            __import__(module_name)
            logger.info(f"✓ Successfully imported: {module_name}")
            successful_imports.append(module_name)
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            failed_imports.append((module_name, str(e)))

    logger.info(f"Import summary: {len(successful_imports)} successful, {len(failed_imports)} failed")
    logger.info("=== End Import Tests ===")

    return successful_imports, failed_imports

# Global session storage (in production, this would be more sophisticated)
_sessions: dict[str, MCPSession] = {}


@asynccontextmanager
async def mcp_lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Manage MCP server lifecycle with resource initialization and cleanup.

    This context manager handles:
    - Configuration loading and validation
    - Cost tracking system initialization
    - Pricing manager setup
    - Session cleanup on shutdown
    """
    logger.info("Starting Second Opinion MCP server...")

    # Set up Python path for proper module resolution
    setup_python_path()

    # Log comprehensive environment information for debugging
    log_environment_info()

    # Test critical imports before proceeding
    successful_imports, failed_imports = test_critical_imports()

    if failed_imports:
        logger.warning(f"Some imports failed, but continuing startup. Failed: {[name for name, _ in failed_imports]}")

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
# Move this import to inside the function to debug import issues
# from .tools.second_opinion import second_opinion_tool
# from .tools.should_downgrade import should_downgrade_tool


# Register the core second_opinion tool
@mcp.tool(
    name="second_opinion",
    description="Get a second opinion on an AI response by comparing it against alternative models for quality assessment and cost optimization"
)
async def second_opinion(
    prompt: str = Field(
        ...,
        description="The original question or task that was asked. This helps provide context for comparison quality.",
        examples=["Write a Python function to calculate fibonacci", "Explain quantum computing", "Debug this code snippet"]
    ),
    primary_model: str | None = Field(
        None,
        description="The model that provided the original response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o', 'google/gemini-pro-1.5') or model name for local models (e.g. 'qwen3-4b-mlx', 'codestral-22b-v0.1').",
        examples=["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "google/gemini-pro-1.5", "qwen3-4b-mlx"]
    ),
    primary_response: str | None = Field(
        None,
        description="The response to evaluate (RECOMMENDED). When provided, saves costs and evaluates the actual response the user saw. If not provided, will generate a new response from the primary model.",
        examples=["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"]
    ),
    context: str | None = Field(
        None,
        description="Additional context about the task domain for better comparison quality. Helps select appropriate comparison models and evaluation criteria.",
        examples=["coding task", "academic research", "creative writing", "technical documentation", "educational content"]
    ),
    comparison_models: list[str] | None = Field(
        None,
        description="Specific models to compare against. If not provided, will auto-select alternatives including cost-effective local options. Use OpenRouter format for cloud models.",
        examples=[["openai/gpt-4o", "google/gemini-pro-1.5"], ["qwen3-4b-mlx", "anthropic/claude-3-haiku"]]
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for this operation in USD. Defaults to $0.25. Set higher for complex tasks requiring multiple model calls.",
        examples=[0.25, 0.50, 1.00],
        ge=0.01,
        le=10.00
    ),
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

    logger.info("=== MCP Tool Call: second_opinion ===")
    logger.info(f"Prompt length: {len(prompt) if prompt else 0}")
    logger.info(f"Primary model: {primary_model}")
    logger.info(f"Has primary response: {primary_response is not None}")
    logger.info(f"Context: {context}")
    logger.info(f"Comparison models: {comparison_models}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Test import of second_opinion_tool with multiple strategies
        logger.info("Attempting to import second_opinion_tool...")
        second_opinion_tool = None
        import_strategies = [
            # Strategy 1: Relative import (current approach)
            lambda: __import__('.tools.second_opinion', package=__package__, fromlist=['second_opinion_tool']).second_opinion_tool,
            # Strategy 2: Absolute import
            lambda: __import__('second_opinion.mcp.tools.second_opinion', fromlist=['second_opinion_tool']).second_opinion_tool,
            # Strategy 3: Direct module import
            lambda: __import__('second_opinion.mcp.tools.second_opinion').second_opinion_tool,
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                second_opinion_tool = strategy()
                logger.info(f"✓ Successfully imported second_opinion_tool using strategy {i}")
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if second_opinion_tool is None:
            # Final fallback: try to import manually step by step
            logger.info("All import strategies failed, trying manual step-by-step import...")
            try:
                import importlib
                module = importlib.import_module('second_opinion.mcp.tools.second_opinion')
                second_opinion_tool = module.second_opinion_tool
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback
                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(f"Unable to import second_opinion_tool after trying multiple strategies. Last error: {final_error}")

        if second_opinion_tool is None:
            raise ImportError("second_opinion_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling second_opinion_tool implementation...")
        logger.info(f"Tool function type: {type(second_opinion_tool)}")
        logger.info(f"Tool function module: {getattr(second_opinion_tool, '__module__', 'unknown')}")

        result = await second_opinion_tool(
            prompt=prompt,
            primary_model=primary_model,
            primary_response=primary_response,
            context=context,
            comparison_models=comparison_models,
            cost_limit=cost_limit,
        )
        logger.info("✓ second_opinion_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

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
        # Log comprehensive error information
        import traceback
        error_details = traceback.format_exc()
        logger.error("=== Error in second_opinion MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Still add to context for debugging
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt=prompt,
            primary_model=primary_model or "unknown",
            comparison_models=comparison_models or [],
            result_summary=f"Error: {str(e)}"
        )

        # Return user-friendly error message instead of raising
        return f"❌ **Second Opinion Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the should_downgrade tool
@mcp.tool(
    name="should_downgrade",
    description="Analyze whether cheaper model alternatives could achieve similar quality for cost optimization"
)
async def should_downgrade(
    current_response: str = Field(
        ...,
        description="The response to analyze for potential cost savings. This is the output from your current (expensive) model that you want to test for downgrade opportunities.",
        examples=["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"]
    ),
    task: str = Field(
        ...,
        description="The original task/question that generated the current response. This provides context for comparing quality across models.",
        examples=["Write a Python function to calculate fibonacci", "Explain quantum computing", "Debug this code snippet"]
    ),
    current_model: str | None = Field(
        None,
        description="The model that generated the current response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "google/gemini-pro-1.5", "qwen3-4b-mlx"]
    ),
    downgrade_candidates: list[str] | None = Field(
        None,
        description="Specific cheaper models to test instead of auto-selection. Use OpenRouter format for cloud models or local model names. If not provided, will auto-select based on current model and task complexity.",
        examples=[["anthropic/claude-3-haiku", "openai/gpt-4o-mini"], ["qwen3-4b-mlx", "codestral-22b-v0.1"], ["google/gemini-flash-1.5"]]
    ),
    test_local: bool = Field(
        True,
        description="Whether to include local models in the comparison for maximum cost savings. Local models have zero marginal cost but may have quality trade-offs."
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for testing downgrade options in USD. Defaults to $0.15 for cost-focused testing. Set higher for more comprehensive analysis.",
        examples=[0.15, 0.25, 0.50],
        ge=0.01,
        le=5.00
    ),
) -> str:
    """
    Analyze whether cheaper model alternatives could achieve similar quality.

    This tool helps optimize AI costs by testing if cheaper models (especially local ones)
    can provide similar quality responses to more expensive cloud models. It focuses on
    cost reduction while maintaining acceptable quality standards.

    COST OPTIMIZATION FOCUS:
    - Tests local models for maximum savings (100% cost reduction)
    - Evaluates budget cloud alternatives (50-80% cost reduction)
    - Provides quality vs cost trade-off analysis
    - Gives specific downgrade recommendations with savings projections

    Args:
        current_response: The response to analyze for cost savings potential
        task: The original task that generated the response (for context)
        current_model: Model that generated the current response (for cost comparison)
        downgrade_candidates: Specific cheaper models to test (None = auto-select)
        test_local: Include local models for maximum cost savings (recommended: True)
        cost_limit: Maximum cost for testing alternatives (default: $0.15)

    Returns:
        A cost optimization report with specific downgrade recommendations,
        quality assessments, cost savings projections, and actionable next steps.

    USAGE PATTERN:
        # Test if expensive model can be downgraded (auto-selection)
        result = await should_downgrade(
            current_response="<response from expensive model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True
        )

        # Test specific models (custom selection)
        result = await should_downgrade(
            current_response="<response from expensive model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-5-sonnet",
            downgrade_candidates=["anthropic/claude-3-haiku", "openai/gpt-4o-mini"]
        )
    """
    # Get or create session for this request
    session = get_mcp_session()
    session.update_activity()

    logger.info("=== MCP Tool Call: should_downgrade ===")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Current model: {current_model}")
    logger.info(f"Response length: {len(current_response) if current_response else 0}")
    logger.info(f"Test local: {test_local}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import should_downgrade_tool with multiple strategies (same as second_opinion)
        logger.info("Attempting to import should_downgrade_tool...")
        should_downgrade_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: __import__('.tools.should_downgrade', package=__package__, fromlist=['should_downgrade_tool']).should_downgrade_tool,
            # Strategy 2: Absolute import
            lambda: __import__('second_opinion.mcp.tools.should_downgrade', fromlist=['should_downgrade_tool']).should_downgrade_tool,
            # Strategy 3: Direct module import
            lambda: __import__('second_opinion.mcp.tools.should_downgrade').should_downgrade_tool,
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                should_downgrade_tool = strategy()
                logger.info(f"✓ Successfully imported should_downgrade_tool using strategy {i}")
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if should_downgrade_tool is None:
            # Final fallback: manual importlib approach
            logger.info("All import strategies failed, trying manual step-by-step import...")
            try:
                import importlib
                module = importlib.import_module('second_opinion.mcp.tools.should_downgrade')
                should_downgrade_tool = module.should_downgrade_tool
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback
                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(f"Unable to import should_downgrade_tool after trying multiple strategies. Last error: {final_error}")

        if should_downgrade_tool is None:
            raise ImportError("should_downgrade_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling should_downgrade_tool implementation...")
        logger.info(f"Tool function type: {type(should_downgrade_tool)}")
        logger.info(f"Tool function module: {getattr(should_downgrade_tool, '__module__', 'unknown')}")

        result = await should_downgrade_tool(
            current_response=current_response,
            task=task,
            current_model=current_model,
            downgrade_candidates=downgrade_candidates,
            test_local=test_local,
            cost_limit=cost_limit,
        )
        logger.info("✓ should_downgrade_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="should_downgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],  # Downgrade candidates are selected internally
            result_summary="Downgrade analysis completed successfully"
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback
        error_details = traceback.format_exc()
        logger.error("=== Error in should_downgrade MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="should_downgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],
            result_summary=f"Error: {str(e)}"
        )

        # Return user-friendly error message
        return f"❌ **Should Downgrade Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the should_upgrade tool
@mcp.tool(
    name="should_upgrade",
    description="Analyze whether premium model alternatives could provide quality improvements that justify additional cost"
)
async def should_upgrade(
    current_response: str = Field(
        ...,
        description="The response to analyze for potential quality improvements. This is the output from your current (budget/mid-tier) model that you want to test for upgrade opportunities.",
        examples=["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"]
    ),
    task: str = Field(
        ...,
        description="The original task/question that generated the current response. This provides context for comparing quality across models.",
        examples=["Write a Python function to calculate fibonacci", "Explain quantum computing", "Debug this code snippet"]
    ),
    current_model: str | None = Field(
        None,
        description="The model that generated the current response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-haiku', 'openai/gpt-4o-mini') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=["anthropic/claude-3-haiku", "openai/gpt-4o-mini", "google/gemini-flash-1.5", "qwen3-4b-mlx"]
    ),
    upgrade_candidates: list[str] | None = Field(
        None,
        description="Specific premium models to test instead of auto-selection. Use OpenRouter format for cloud models. If not provided, will auto-select based on current model and task complexity.",
        examples=[["anthropic/claude-3-opus", "openai/gpt-4o"], ["anthropic/claude-3-5-sonnet", "google/gemini-pro-1.5"]]
    ),
    include_premium: bool = Field(
        True,
        description="Whether to include premium models in the comparison for maximum quality improvement. Premium models provide the best quality but at higher cost."
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for testing upgrade options in USD. Defaults to $0.50 for comprehensive quality testing. Set higher for complex tasks requiring premium models.",
        examples=[0.50, 1.00, 2.00],
        ge=0.01,
        le=10.00
    ),
) -> str:
    """
    Analyze whether premium model alternatives could provide quality improvements.

    This tool helps optimize AI quality by testing if premium models can provide
    significant quality improvements that justify additional cost. It focuses on
    quality enhancement while providing transparent cost analysis.

    QUALITY OPTIMIZATION FOCUS:
    - Tests premium models for maximum quality improvement
    - Evaluates cross-provider alternatives for diverse perspectives
    - Provides quality vs cost trade-off analysis
    - Gives specific upgrade recommendations with ROI analysis

    Args:
        current_response: The response to analyze for quality improvement potential
        task: The original task that generated the response (for context)
        current_model: Model that generated the current response (for comparison)
        upgrade_candidates: Specific premium models to test (None = auto-select)
        include_premium: Include premium models for maximum quality (recommended: True)
        cost_limit: Maximum cost for testing alternatives (default: $0.50)

    Returns:
        A quality enhancement report with specific upgrade recommendations,
        quality assessments, cost vs benefit analysis, and actionable next steps.

    USAGE PATTERN:
        # Test if budget model can be upgraded (auto-selection)
        result = await should_upgrade(
            current_response="<response from budget model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-haiku",
            include_premium=True
        )

        # Test specific premium models (custom selection)
        result = await should_upgrade(
            current_response="<response from budget model>",
            task="Write a Python function to calculate fibonacci",
            current_model="anthropic/claude-3-haiku",
            upgrade_candidates=["anthropic/claude-3-opus", "openai/gpt-4o"]
        )
    """
    # Get or create session for this request
    session = get_mcp_session()
    session.update_activity()

    logger.info("=== MCP Tool Call: should_upgrade ===")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Current model: {current_model}")
    logger.info(f"Response length: {len(current_response) if current_response else 0}")
    logger.info(f"Include premium: {include_premium}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import should_upgrade_tool with multiple strategies (same as other tools)
        logger.info("Attempting to import should_upgrade_tool...")
        should_upgrade_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: __import__('.tools.should_upgrade', package=__package__, fromlist=['should_upgrade_tool']).should_upgrade_tool,
            # Strategy 2: Absolute import
            lambda: __import__('second_opinion.mcp.tools.should_upgrade', fromlist=['should_upgrade_tool']).should_upgrade_tool,
            # Strategy 3: Direct module import
            lambda: __import__('second_opinion.mcp.tools.should_upgrade').should_upgrade_tool,
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                should_upgrade_tool = strategy()
                logger.info(f"✓ Successfully imported should_upgrade_tool using strategy {i}")
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if should_upgrade_tool is None:
            # Final fallback: manual importlib approach
            logger.info("All import strategies failed, trying manual step-by-step import...")
            try:
                import importlib
                module = importlib.import_module('second_opinion.mcp.tools.should_upgrade')
                should_upgrade_tool = module.should_upgrade_tool
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback
                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(f"Unable to import should_upgrade_tool after trying multiple strategies. Last error: {final_error}")

        if should_upgrade_tool is None:
            raise ImportError("should_upgrade_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling should_upgrade_tool implementation...")
        logger.info(f"Tool function type: {type(should_upgrade_tool)}")
        logger.info(f"Tool function module: {getattr(should_upgrade_tool, '__module__', 'unknown')}")

        result = await should_upgrade_tool(
            current_response=current_response,
            task=task,
            current_model=current_model,
            upgrade_candidates=upgrade_candidates,
            include_premium=include_premium,
            cost_limit=cost_limit,
        )
        logger.info("✓ should_upgrade_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="should_upgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],  # Upgrade candidates are selected internally
            result_summary="Upgrade analysis completed successfully"
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback
        error_details = traceback.format_exc()
        logger.error("=== Error in should_upgrade MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="should_upgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],
            result_summary=f"Error: {str(e)}"
        )

        # Return user-friendly error message
        return f"❌ **Should Upgrade Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


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
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Second Opinion MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging for standalone mode
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Update our MCP logger setup
    logger = setup_mcp_logging(debug=args.debug)

    logger.info(f"Starting Second Opinion MCP server in standalone mode (debug={'on' if args.debug else 'off'})")
    mcp.run()
