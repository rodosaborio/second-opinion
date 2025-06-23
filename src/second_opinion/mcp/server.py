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
from ..utils.template_loader import load_mcp_tool_description
from .session import MCPSession


def get_tool_description(tool_name: str) -> str:
    """
    Get MCP tool description from externalized templates.

    Args:
        tool_name: Name of the tool (matches template filename)

    Returns:
        Tool description string, with fallback if template loading fails
    """
    try:
        return load_mcp_tool_description(tool_name)
    except Exception as e:
        logger.error(f"Failed to load description for tool '{tool_name}': {e}")
        # Fallback to minimal description
        return f"MCP tool: {tool_name} (description unavailable)"


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
        "%(asctime)s - %(name)s - %(levelname)s - [MCP] %(message)s"
    )

    # Add stderr handler (MCP servers should log to stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Set up logger
logger = setup_mcp_logging(debug=True)  # Enable debug for troubleshooting


def setup_python_path() -> None:
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


def log_environment_info() -> None:
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
    env_vars = ["PATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        logger.info(f"{var}: {value}")

    logger.info("=== End Environment Information ===")


def test_critical_imports() -> tuple[list[str], list[str]]:
    """Test importing critical modules and log results."""
    logger.info("=== Testing Critical Imports ===")

    critical_modules = [
        "second_opinion",
        "second_opinion.mcp",
        "second_opinion.mcp.tools",
        "second_opinion.mcp.tools.second_opinion",
        "second_opinion.mcp.tools.should_downgrade",
        "second_opinion.mcp.tools.should_upgrade",
        "second_opinion.core.models",
        "second_opinion.clients",
        "second_opinion.utils.cost_tracking",
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

    logger.info(
        f"Import summary: {len(successful_imports)} successful, {len(failed_imports)} failed"
    )
    logger.info("=== End Import Tests ===")

    return successful_imports, failed_imports


# Global session storage (in production, this would be more sophisticated)
_sessions: dict[str, MCPSession] = {}


@asynccontextmanager
async def mcp_lifespan(_: FastMCP[Any]) -> AsyncIterator[dict[str, Any]]:
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
        logger.warning(
            f"Some imports failed, but continuing startup. Failed: {[name for name, _ in failed_imports]}"
        )

    try:
        # Initialize core systems
        settings = get_settings()
        cost_guard = get_cost_guard()
        pricing_manager = get_pricing_manager()

        # Validate essential configuration
        if not settings.openrouter_api_key:
            logger.warning(
                "No OpenRouter API key configured - tool functionality may be limited"
            )

        # Initialize pricing data
        try:
            await pricing_manager.fetch_latest_pricing()
            logger.info(
                f"Pricing manager initialized with {pricing_manager.get_model_count()} models"
            )
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
    description=get_tool_description("second_opinion"),
)
async def second_opinion(
    prompt: str = Field(
        ...,
        description="The original question or task that was asked. This helps provide context for comparison quality.",
        examples=[
            "Write a Python function to calculate fibonacci",
            "Explain quantum computing",
            "Debug this code snippet",
        ],
    ),
    primary_model: str | None = Field(
        None,
        description="The model that provided the original response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o', 'google/gemini-pro-1.5') or model name for local models (e.g. 'qwen3-4b-mlx', 'codestral-22b-v0.1').",
        examples=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "qwen3-4b-mlx",
        ],
    ),
    primary_response: str | None = Field(
        None,
        description="The response to evaluate (RECOMMENDED). When provided, saves costs and evaluates the actual response the user saw. If not provided, will generate a new response from the primary model.",
        examples=[
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ],
    ),
    context: str | None = Field(
        None,
        description="Additional context about the task domain for better comparison quality. Helps select appropriate comparison models and evaluation criteria.",
        examples=[
            "coding task",
            "academic research",
            "creative writing",
            "technical documentation",
            "educational content",
        ],
    ),
    comparison_models: list[str] | None = Field(
        None,
        description="Specific models to compare against. If not provided, will auto-select alternatives including cost-effective local options. Use OpenRouter format for cloud models.",
        examples=[
            ["openai/gpt-4o", "google/gemini-pro-1.5"],
            ["qwen3-4b-mlx", "anthropic/claude-3-haiku"],
        ],
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for this operation in USD. Defaults to $0.25. Set higher for complex tasks requiring multiple model calls.",
        examples=[0.25, 0.50, 1.00],
        ge=0.01,
        le=10.00,
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
            lambda: importlib.import_module(
                ".tools.second_opinion", package=__package__
            ).second_opinion_tool,  # type: ignore[attr-defined]
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.second_opinion",
                fromlist=["second_opinion_tool"],
            ).second_opinion_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.second_opinion"
            ).second_opinion_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                second_opinion_tool = strategy()
                logger.info(
                    f"✓ Successfully imported second_opinion_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if second_opinion_tool is None:
            # Final fallback: try to import manually step by step
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.second_opinion"
                )
                second_opinion_tool = module.second_opinion_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import second_opinion_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if second_opinion_tool is None:
            raise ImportError("second_opinion_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling second_opinion_tool implementation...")
        logger.info(f"Tool function type: {type(second_opinion_tool)}")
        logger.info(
            f"Tool function module: {getattr(second_opinion_tool, '__module__', 'unknown')}"
        )

        result = await second_opinion_tool(
            prompt=prompt,
            primary_model=primary_model,
            primary_response=primary_response,
            context=context,
            comparison_models=comparison_models,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ second_opinion_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt=prompt,
            primary_model=primary_model or "unknown",
            comparison_models=comparison_models or [],
            result_summary="Comparison completed successfully",
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
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message instead of raising
        return f"❌ **Second Opinion Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the should_downgrade tool
@mcp.tool(
    name="should_downgrade",
    description=get_tool_description("should_downgrade"),
)
async def should_downgrade(
    current_response: str = Field(
        ...,
        description="The response to analyze for potential cost savings. This is the output from your current (expensive) model that you want to test for downgrade opportunities.",
        examples=[
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ],
    ),
    task: str = Field(
        ...,
        description="The original task/question that generated the current response. This provides context for comparing quality across models.",
        examples=[
            "Write a Python function to calculate fibonacci",
            "Explain quantum computing",
            "Debug this code snippet",
        ],
    ),
    current_model: str | None = Field(
        None,
        description="The model that generated the current response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "qwen3-4b-mlx",
        ],
    ),
    downgrade_candidates: list[str] | None = Field(
        None,
        description="Specific cheaper models to test instead of auto-selection. Use OpenRouter format for cloud models or local model names. If not provided, will auto-select based on current model and task complexity.",
        examples=[
            ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
            ["qwen3-4b-mlx", "codestral-22b-v0.1"],
            ["google/gemini-flash-1.5"],
        ],
    ),
    test_local: bool = Field(
        True,
        description="Whether to include local models in the comparison for maximum cost savings. Local models have zero marginal cost but may have quality trade-offs.",
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for testing downgrade options in USD. Defaults to $0.15 for cost-focused testing. Set higher for more comprehensive analysis.",
        examples=[0.15, 0.25, 0.50],
        ge=0.01,
        le=5.00,
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
            lambda: importlib.import_module(
                ".tools.should_downgrade", package=__package__
            ).should_downgrade_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.should_downgrade",
                fromlist=["should_downgrade_tool"],
            ).should_downgrade_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.should_downgrade"
            ).should_downgrade_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                should_downgrade_tool = strategy()
                logger.info(
                    f"✓ Successfully imported should_downgrade_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if should_downgrade_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.should_downgrade"
                )
                should_downgrade_tool = module.should_downgrade_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import should_downgrade_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if should_downgrade_tool is None:
            raise ImportError("should_downgrade_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling should_downgrade_tool implementation...")
        logger.info(f"Tool function type: {type(should_downgrade_tool)}")
        logger.info(
            f"Tool function module: {getattr(should_downgrade_tool, '__module__', 'unknown')}"
        )

        result = await should_downgrade_tool(
            current_response=current_response,
            task=task,
            current_model=current_model,
            downgrade_candidates=downgrade_candidates,
            test_local=test_local,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ should_downgrade_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="should_downgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],  # Downgrade candidates are selected internally
            result_summary="Downgrade analysis completed successfully",
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
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Should Downgrade Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the should_upgrade tool
@mcp.tool(
    name="should_upgrade",
    description=get_tool_description("should_upgrade"),
)
async def should_upgrade(
    current_response: str = Field(
        ...,
        description="The response to analyze for potential quality improvements. This is the output from your current (budget/mid-tier) model that you want to test for upgrade opportunities.",
        examples=[
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ],
    ),
    task: str = Field(
        ...,
        description="The original task/question that generated the current response. This provides context for comparing quality across models.",
        examples=[
            "Write a Python function to calculate fibonacci",
            "Explain quantum computing",
            "Debug this code snippet",
        ],
    ),
    current_model: str | None = Field(
        None,
        description="The model that generated the current response. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-haiku', 'openai/gpt-4o-mini') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=[
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
            "qwen3-4b-mlx",
        ],
    ),
    upgrade_candidates: list[str] | None = Field(
        None,
        description="Specific premium models to test instead of auto-selection. Use OpenRouter format for cloud models. If not provided, will auto-select based on current model and task complexity.",
        examples=[
            ["anthropic/claude-3-opus", "openai/gpt-4o"],
            ["anthropic/claude-3-5-sonnet", "google/gemini-pro-1.5"],
        ],
    ),
    include_premium: bool = Field(
        True,
        description="Whether to include premium models in the comparison for maximum quality improvement. Premium models provide the best quality but at higher cost.",
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for testing upgrade options in USD. Defaults to $0.50 for comprehensive quality testing. Set higher for complex tasks requiring premium models.",
        examples=[0.50, 1.00, 2.00],
        ge=0.01,
        le=10.00,
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
            lambda: importlib.import_module(
                ".tools.should_upgrade", package=__package__
            ).should_upgrade_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.should_upgrade",
                fromlist=["should_upgrade_tool"],
            ).should_upgrade_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.should_upgrade"
            ).should_upgrade_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                should_upgrade_tool = strategy()
                logger.info(
                    f"✓ Successfully imported should_upgrade_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if should_upgrade_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.should_upgrade"
                )
                should_upgrade_tool = module.should_upgrade_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import should_upgrade_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if should_upgrade_tool is None:
            raise ImportError("should_upgrade_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling should_upgrade_tool implementation...")
        logger.info(f"Tool function type: {type(should_upgrade_tool)}")
        logger.info(
            f"Tool function module: {getattr(should_upgrade_tool, '__module__', 'unknown')}"
        )

        result = await should_upgrade_tool(
            current_response=current_response,
            task=task,
            current_model=current_model,
            upgrade_candidates=upgrade_candidates,
            include_premium=include_premium,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ should_upgrade_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="should_upgrade",
            prompt=task,
            primary_model=current_model or "unknown",
            comparison_models=[],  # Upgrade candidates are selected internally
            result_summary="Upgrade analysis completed successfully",
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
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Should Upgrade Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the compare_responses tool
@mcp.tool(
    name="compare_responses",
    description=get_tool_description("compare_responses"),
)
async def compare_responses(
    response_a: str = Field(
        ...,
        description="The first response to compare. This is one of the responses you want to analyze for quality, accuracy, and usefulness.",
        examples=[
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ],
    ),
    response_b: str = Field(
        ...,
        description="The second response to compare. This is the other response you want to analyze for quality, accuracy, and usefulness.",
        examples=[
            "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        ],
    ),
    task: str = Field(
        ...,
        description="The original task/question that generated both responses. This provides context for comparing quality and relevance.",
        examples=[
            "Write a Python function to calculate fibonacci",
            "Explain quantum computing",
            "Debug this code snippet",
        ],
    ),
    model_a: str | None = Field(
        None,
        description="The model that generated response_a. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "qwen3-4b-mlx",
        ],
    ),
    model_b: str | None = Field(
        None,
        description="The model that generated response_b. Use OpenRouter format for cloud models or model name for local models (same format as model_a).",
        examples=[
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
            "codestral-22b-v0.1",
        ],
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for evaluation in USD. Defaults to $0.25. Note: No additional API costs when comparing existing responses.",
        examples=[0.25, 0.50, 1.00],
        ge=0.01,
        le=10.00,
    ),
) -> str:
    """
    Compare two AI responses with detailed side-by-side analysis.

    This tool provides comprehensive comparison of two responses across multiple
    quality criteria, enabling informed decisions about model selection, response
    quality, and cost optimization. Perfect for A/B testing different models
    or comparing responses from different AI systems.

    KEY BENEFITS:
    - Zero additional API costs when comparing existing responses
    - Detailed quality scoring across accuracy, completeness, clarity, usefulness
    - Side-by-side analysis with winner determination
    - Cost analysis for both models with tier comparison
    - Actionable recommendations for future model selection

    Args:
        response_a: The first response to compare
        response_b: The second response to compare
        task: The original task that generated both responses (for context)
        model_a: Model that generated response_a (optional, for cost analysis)
        model_b: Model that generated response_b (optional, for cost analysis)
        cost_limit: Maximum cost for evaluation (default: $0.25)

    Returns:
        A detailed comparison report with quality analysis, winner determination,
        cost comparison, and actionable recommendations for model selection.

    USAGE PATTERN:
        # Compare responses from two different models
        result = await compare_responses(
            response_a="Response from Model A...",
            response_b="Response from Model B...",
            task="Write a Python function to calculate fibonacci",
            model_a="anthropic/claude-3-5-sonnet",
            model_b="openai/gpt-4o"
        )
    """
    # Get or create session for this request
    session = get_mcp_session()
    session.update_activity()

    logger.info("=== MCP Tool Call: compare_responses ===")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Response A length: {len(response_a) if response_a else 0}")
    logger.info(f"Response B length: {len(response_b) if response_b else 0}")
    logger.info(f"Model A: {model_a}")
    logger.info(f"Model B: {model_b}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import compare_responses_tool with multiple strategies (same as other tools)
        logger.info("Attempting to import compare_responses_tool...")
        compare_responses_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: importlib.import_module(
                ".tools.compare_responses", package=__package__
            ).compare_responses_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.compare_responses",
                fromlist=["compare_responses_tool"],
            ).compare_responses_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.compare_responses"
            ).compare_responses_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                compare_responses_tool = strategy()
                logger.info(
                    f"✓ Successfully imported compare_responses_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if compare_responses_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.compare_responses"
                )
                compare_responses_tool = module.compare_responses_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import compare_responses_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if compare_responses_tool is None:
            raise ImportError(
                "compare_responses_tool is None after all import attempts"
            )

        # Call the tool implementation
        logger.info("Calling compare_responses_tool implementation...")
        logger.info(f"Tool function type: {type(compare_responses_tool)}")
        logger.info(
            f"Tool function module: {getattr(compare_responses_tool, '__module__', 'unknown')}"
        )

        result = await compare_responses_tool(
            response_a=response_a,
            response_b=response_b,
            task=task,
            model_a=model_a,
            model_b=model_b,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ compare_responses_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="compare_responses",
            prompt=task,
            primary_model=model_a or "unknown",
            comparison_models=[model_b] if model_b else [],
            result_summary="Response comparison completed successfully",
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback

        error_details = traceback.format_exc()
        logger.error("=== Error in compare_responses MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="compare_responses",
            prompt=task,
            primary_model=model_a or "unknown",
            comparison_models=[model_b] if model_b else [],
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Compare Responses Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the usage_analytics tool
@mcp.tool(
    name="usage_analytics",
    description=get_tool_description("usage_analytics"),
)
async def usage_analytics(
    time_period: str = Field(
        "week",
        description="Analysis period - 'day', 'week', 'month', 'quarter', 'year', or 'all'",
        examples=["day", "week", "month", "quarter", "year", "all"],
    ),
    breakdown_by: str = Field(
        "model",
        description="Primary breakdown dimension - 'model', 'interface', 'tool', 'cost', or 'time'",
        examples=["model", "interface", "tool", "cost", "time"],
    ),
    interface_type: str | None = Field(
        None,
        description="Filter by interface - 'cli', 'mcp', or None for all",
        examples=["cli", "mcp"],
    ),
    include_trends: bool = Field(
        True,
        description="Whether to include trend analysis over time",
    ),
    include_recommendations: bool = Field(
        True,
        description="Whether to include optimization recommendations",
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost for this analytics operation (default: $0.10)",
        examples=[0.10, 0.25, 0.50],
        ge=0.01,
        le=5.00,
    ),
) -> str:
    """
    Analyze model usage patterns and provide cost optimization insights.

    This tool provides comprehensive analytics on your AI model usage, including:
    - Cost breakdown by model, time period, and interface
    - Usage patterns and trends over time
    - Cost optimization recommendations
    - Model performance insights
    - Budget analysis and projections

    Args:
        time_period: Analysis period for data aggregation
        breakdown_by: Primary dimension for organizing results
        interface_type: Filter results by CLI or MCP usage
        include_trends: Add trend analysis over time periods
        include_recommendations: Include cost optimization suggestions
        cost_limit: Maximum cost for analytics processing

    Returns:
        Comprehensive usage analytics report with insights and actionable recommendations

    Usage Examples:
        # Weekly cost breakdown by model
        usage_analytics(time_period="week", breakdown_by="model")

        # Monthly MCP tool usage analysis with trends
        usage_analytics(time_period="month", interface_type="mcp", include_trends=True)

        # Cost optimization analysis for all time
        usage_analytics(time_period="all", breakdown_by="cost", include_recommendations=True)
    """
    # Get or create session for this request
    session = get_mcp_session()
    session.update_activity()

    logger.info("=== MCP Tool Call: usage_analytics ===")
    logger.info(f"Time period: {time_period}")
    logger.info(f"Breakdown by: {breakdown_by}")
    logger.info(f"Interface type: {interface_type}")
    logger.info(f"Include trends: {include_trends}")
    logger.info(f"Include recommendations: {include_recommendations}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import usage_analytics_tool with multiple strategies (same as other tools)
        logger.info("Attempting to import usage_analytics_tool...")
        usage_analytics_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: importlib.import_module(
                ".tools.usage_analytics", package=__package__
            ).usage_analytics_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.usage_analytics",
                fromlist=["usage_analytics_tool"],
            ).usage_analytics_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.usage_analytics"
            ).usage_analytics_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                usage_analytics_tool = strategy()
                logger.info(
                    f"✓ Successfully imported usage_analytics_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if usage_analytics_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.usage_analytics"
                )
                usage_analytics_tool = module.usage_analytics_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import usage_analytics_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if usage_analytics_tool is None:
            raise ImportError("usage_analytics_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling usage_analytics_tool implementation...")
        logger.info(f"Tool function type: {type(usage_analytics_tool)}")
        logger.info(
            f"Tool function module: {getattr(usage_analytics_tool, '__module__', 'unknown')}"
        )

        result = await usage_analytics_tool(
            time_period=time_period,
            breakdown_by=breakdown_by,
            interface_type=interface_type,
            include_trends=include_trends,
            include_recommendations=include_recommendations,
            cost_limit=cost_limit,
        )
        logger.info("✓ usage_analytics_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="usage_analytics",
            prompt=f"Analytics: {time_period} period, breakdown by {breakdown_by}",
            primary_model="local-analytics",
            comparison_models=[],
            result_summary=f"Analytics ({time_period}) completed successfully",
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback

        error_details = traceback.format_exc()
        logger.error("=== Error in usage_analytics MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="usage_analytics",
            prompt=f"Analytics: {time_period} period, breakdown by {breakdown_by}",
            primary_model="local-analytics",
            comparison_models=[],
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Usage Analytics Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the consult tool
@mcp.tool(
    name="consult",
    description=get_tool_description("consult"),
)
async def consult(
    query: str = Field(
        ...,
        description="The question or task to consult about. This can be a request for expert opinion, task delegation, or complex problem-solving.",
        examples=[
            "Should I use async/await or threading for this I/O operation?",
            "Write unit tests for this function",
            "Help me design a scalable authentication system",
        ],
    ),
    consultation_type: str = Field(
        "quick",
        description="Type of consultation: 'quick' (single-turn expert opinion), 'deep' (multi-turn analysis), 'delegate' (task completion), 'brainstorm' (creative exploration)",
        examples=["quick", "deep", "delegate", "brainstorm"],
    ),
    target_model: str | None = Field(
        None,
        description="Specific model to consult with. Auto-selected if not provided. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o') or model name for local models (e.g. 'qwen3-4b-mlx').",
        examples=[
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o-mini",
            "google/gemini-pro-1.5",
            "qwen3-4b-mlx",
        ],
    ),
    session_id: str | None = Field(
        None,
        description="Continue existing consultation session (for multi-turn). Use the session ID from a previous consultation to continue the conversation.",
        examples=["abc123-session-id"],
    ),
    max_turns: int = Field(
        3,
        description="Maximum conversation turns for multi-turn consultations (1-5). Only applies to 'deep' and 'brainstorm' types.",
        ge=1,
        le=5,
        examples=[1, 3, 5],
    ),
    context: str | None = Field(
        None,
        description="Additional context about the task domain for better model routing and consultation quality.",
        examples=[
            "coding task",
            "system architecture",
            "performance optimization",
            "creative writing",
        ],
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for this consultation in USD. Defaults vary by consultation type: delegate ($0.10), deep ($0.50), others ($0.25).",
        examples=[0.10, 0.25, 0.50],
        ge=0.01,
        le=10.00,
    ),
) -> str:
    """
    Consult with AI models for expert opinions, task delegation, and problem solving.

    This tool enables AI-to-AI consultation across different specialized models,
    supporting both single-turn expert opinions and multi-turn collaborative
    problem-solving sessions.

    CONSULTATION TYPES:
    - **quick**: Single-turn expert opinion with focused advice
    - **deep**: Multi-turn comprehensive analysis and problem exploration
    - **delegate**: Task completion using cost-effective models (60-80% savings)
    - **brainstorm**: Creative collaborative exploration with multiple perspectives

    SMART MODEL ROUTING:
    - Automatically selects optimal models based on consultation type and task complexity
    - delegate + simple → GPT-4o-mini (cost optimization)
    - expert + complex → Claude Opus (premium quality)
    - brainstorm → GPT-4o (creative balance)
    - quick → Claude 3.5 Sonnet (reliable default)

    COST OPTIMIZATION:
    - Task delegation: 60-80% cost savings vs premium models
    - Multi-turn conversations: Intelligent context management
    - Transparent cost tracking with session management
    - Per-consultation cost limits with auto-stop protection

    Args:
        query: The question or task to consult about
        consultation_type: Type of consultation (quick/deep/delegate/brainstorm)
        target_model: Specific model to consult (auto-selected if None)
        session_id: Continue existing session for multi-turn conversations
        max_turns: Maximum turns for multi-turn consultations (1-5)
        context: Additional context for better routing and quality
        cost_limit: Maximum cost for this consultation (type-specific defaults)

    Returns:
        Consultation results with expert insights, cost analysis, session management,
        and actionable recommendations for follow-up.

    USAGE PATTERNS:
        # Quick expert opinion
        result = await consult(
            query="Should I use async/await or threading for this I/O operation?",
            consultation_type="quick",
            context="performance optimization"
        )

        # Task delegation for cost savings
        result = await consult(
            query="Write unit tests for this function: def fibonacci(n): ...",
            consultation_type="delegate",
            target_model="openai/gpt-4o-mini"
        )

        # Deep problem solving (multi-turn)
        result = await consult(
            query="Help me design a scalable authentication system",
            consultation_type="deep",
            max_turns=3,
            context="system architecture"
        )

        # Continue existing conversation
        result = await consult(
            query="Now help me implement the OAuth2 flow",
            consultation_type="deep",
            session_id="abc123-session-id"
        )
    """
    # Get or create session for this request
    session = get_mcp_session()
    session.update_activity()

    logger.info("=== MCP Tool Call: consult ===")
    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Consultation type: {consultation_type}")
    logger.info(f"Target model: {target_model}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Max turns: {max_turns}")
    logger.info(f"Context: {context}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import consult_tool with multiple strategies (same as other tools)
        logger.info("Attempting to import consult_tool...")
        consult_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: importlib.import_module(
                ".tools.consult", package=__package__
            ).consult_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.consult", fromlist=["consult_tool"]
            ).consult_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__("second_opinion.mcp.tools.consult").consult_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                consult_tool = strategy()
                logger.info(f"✓ Successfully imported consult_tool using strategy {i}")
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if consult_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module("second_opinion.mcp.tools.consult")
                consult_tool = module.consult_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import consult_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if consult_tool is None:
            raise ImportError("consult_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling consult_tool implementation...")
        logger.info(f"Tool function type: {type(consult_tool)}")
        logger.info(
            f"Tool function module: {getattr(consult_tool, '__module__', 'unknown')}"
        )

        result = await consult_tool(
            query=query,
            consultation_type=consultation_type,
            target_model=target_model,
            session_id=session.session_id,  # Use MCP session ID for consistency
            max_turns=max_turns,
            context=context,
            cost_limit=cost_limit,
        )
        logger.info("✓ consult_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="consult",
            prompt=query,
            primary_model=target_model or "auto-selected",
            comparison_models=[],  # Consultation uses single model
            result_summary=f"Consultation ({consultation_type}) completed successfully",
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback

        error_details = traceback.format_exc()
        logger.error("=== Error in consult MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="consult",
            prompt=query,
            primary_model=target_model or "unknown",
            comparison_models=[],
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Consult Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the batch_comparison tool
@mcp.tool(
    name="batch_comparison",
    description=get_tool_description("batch_comparison"),
)
async def batch_comparison(
    task: str = Field(
        ...,
        description="The task or question to evaluate responses for. This provides context for comparison quality.",
        examples=[
            "Write a Python function to validate email addresses",
            "Explain quantum computing to a 10-year-old",
            "Debug this JavaScript code snippet",
        ],
    ),
    responses: list[str] | None = Field(
        None,
        description="List of existing responses to compare (optional). If provided, these will be evaluated without additional API calls. Must match the order of models if both are provided.",
        examples=[
            [
                "def validate_email(email): return '@' in email",
                "import re\ndef validate_email(email): return re.match(r'^[^@]+@[^@]+$', email)",
                "from email_validator import validate_email as ve\ndef validate_email(email): return ve(email)",
            ]
        ],
    ),
    models: list[str] | None = Field(
        None,
        description="List of models to use. Use OpenRouter format for cloud models (e.g. 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o') or model name for local models (e.g. 'qwen3-4b-mlx'). If responses are provided, this should match the response order. If no responses provided, will generate responses from these models.",
        examples=[
            ["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"],
            [
                "openai/gpt-4o-mini",
                "anthropic/claude-3-haiku",
                "google/gemini-flash-1.5",
            ],
        ],
    ),
    context: str | None = Field(
        None,
        description="Additional context about the task domain for better evaluation quality. For example: 'coding task', 'academic research', 'creative writing', 'educational content'.",
        examples=[
            "coding task",
            "academic research",
            "creative writing",
            "educational content",
        ],
    ),
    rank_by: str = Field(
        "quality",
        description="Ranking criteria for model comparison: 'quality' (response quality), 'cost' (cost efficiency), 'speed' (response speed estimation), or 'comprehensive' (balanced scoring).",
        examples=["quality", "cost", "speed", "comprehensive"],
    ),
    max_models: int = Field(
        5,
        description="Maximum number of models to compare (default: 5, max: 10). This limit prevents excessive costs and ensures meaningful comparisons.",
        examples=[3, 5, 8],
        ge=2,
        le=10,
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost limit for this operation in USD. Defaults to $0.50 for batch operations. Set higher for complex tasks requiring many model calls.",
        examples=[0.25, 0.5, 1.0],
        gt=0,
        le=10,
    ),
    session_id: str | None = Field(
        None,
        description="Session ID for conversation tracking and cost management. If not provided, a new session will be created.",
    ),
) -> str:
    """
    Batch model comparison MCP tool with comprehensive evaluation and ranking.

    This tool provides systematic model evaluation for the same task across multiple models,
    with detailed ranking and analysis. Perfect for model selection, benchmarking, and
    understanding model strengths for specific use cases.

    USAGE PATTERNS:
    1. **Evaluate Existing Responses**: Compare responses you already have
    2. **Model Benchmarking**: Generate fresh responses from multiple models
    3. **Mixed Evaluation**: Some existing responses + generate from additional models

    Cost-Saving Tips:
    - Use existing responses when possible to avoid API calls
    - Include local models (qwen3-4b-mlx, codestral-22b-v0.1) for cost-free alternatives
    - Set appropriate cost_limit to control expenses

    Returns:
        Comprehensive comparison report with:
        - Ranked model performance with detailed scoring
        - Cost-effectiveness analysis
        - Local vs cloud model comparison
        - Model selection recommendations for similar tasks
    """
    # Get session for this request
    session = get_mcp_session(session_id)

    logger.info(
        f"=== Starting batch_comparison MCP tool (Session: {session.session_id}) ==="
    )
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Responses provided: {responses is not None}")
    logger.info(f"Models: {models}")
    logger.info(f"Rank by: {rank_by}")
    logger.info(f"Max models: {max_models}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import the batch_comparison_tool implementation
        logger.info("Importing batch_comparison_tool...")

        batch_comparison_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: importlib.import_module(
                ".tools.batch_comparison", package=__package__
            ).batch_comparison_tool,  # type: ignore[attr-defined]
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.batch_comparison",
                fromlist=["batch_comparison_tool"],
            ).batch_comparison_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.batch_comparison"
            ).batch_comparison_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                batch_comparison_tool = strategy()
                logger.info(
                    f"✓ Successfully imported batch_comparison_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if batch_comparison_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.batch_comparison"
                )
                batch_comparison_tool = module.batch_comparison_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import batch_comparison_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if batch_comparison_tool is None:
            raise ImportError("batch_comparison_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling batch_comparison_tool implementation...")
        logger.info(f"Tool function type: {type(batch_comparison_tool)}")
        logger.info(
            f"Tool function module: {getattr(batch_comparison_tool, '__module__', 'unknown')}"
        )

        result = await batch_comparison_tool(
            task=task,
            responses=responses,
            models=models,
            context=context,
            rank_by=rank_by,
            max_models=max_models,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ batch_comparison_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="batch_comparison",
            prompt=task,
            primary_model="multiple",  # Multiple models compared
            comparison_models=models or [],
            result_summary="Batch comparison completed successfully",
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback

        error_details = traceback.format_exc()
        logger.error("=== Error in batch_comparison MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="batch_comparison",
            prompt=task,
            primary_model="multiple",
            comparison_models=models or [],
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Batch Comparison Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


# Register the model_benchmark tool
@mcp.tool(
    name="model_benchmark",
    description=get_tool_description("model_benchmark"),
)
async def model_benchmark(
    models: list[str] = Field(
        ...,
        description="List of models to benchmark. Format: Cloud models: 'anthropic/claude-3-5-sonnet', 'openai/gpt-4o', etc. Local models: 'qwen3-4b-mlx', 'codestral-22b-v0.1', etc.",
        examples=[
            ["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"],
            [
                "openai/gpt-4o-mini",
                "anthropic/claude-3-haiku",
                "google/gemini-flash-1.5",
            ],
        ],
        min_length=2,
        max_length=8,
    ),
    task_types: list[str] | None = Field(
        None,
        description="Task categories to test (default: ['coding', 'reasoning', 'creative']). Available: 'coding', 'reasoning', 'creative', 'analysis', 'explanation'",
        examples=[
            ["coding", "reasoning"],
            ["creative", "analysis", "explanation"],
            ["coding", "reasoning", "creative", "analysis", "explanation"],
        ],
    ),
    sample_size: int = Field(
        3,
        description="Number of tasks per category to test (default: 3, max: 5)",
        ge=1,
        le=5,
    ),
    evaluation_criteria: str = Field(
        "comprehensive",
        description="How to evaluate responses: 'comprehensive' (balanced), 'accuracy' (correctness focus), 'creativity' (originality focus), 'speed' (efficiency focus)",
        examples=["comprehensive", "accuracy", "creativity", "speed"],
    ),
    cost_limit: float | None = Field(
        None,
        description="Maximum cost for this operation in USD (default: $2.00). Benchmarking can be expensive due to multiple model calls and evaluations.",
        examples=[1.0, 2.0, 5.0],
        gt=0,
        le=10,
    ),
    session_id: str | None = Field(
        None,
        description="Session ID for conversation tracking (optional)",
    ),
) -> str:
    """
    Benchmark multiple models across different task types for comprehensive performance analysis.

    This tool systematically tests models across various task categories to provide insights
    into their strengths, weaknesses, and optimal use cases. Perfect for model selection,
    performance analysis, and understanding model capabilities across different domains.

    COMPREHENSIVE BENCHMARKING PROCESS:
    1. Tests each model on multiple tasks across selected categories
    2. Performs pairwise comparisons using consistent evaluation criteria
    3. Analyzes performance patterns and cost efficiency
    4. Provides statistical confidence indicators
    5. Generates actionable recommendations for model selection

    Args:
        models: List of models to benchmark (2-8 models). Mix cloud and local models for best insights.
        task_types: Categories to test. Default covers core capabilities: coding, reasoning, creative.
        sample_size: Tasks per category (1-5). Higher = more reliable but more expensive.
        evaluation_criteria: Focus area for evaluation scoring.
        cost_limit: Budget protection (default $2.00 for comprehensive benchmarking).
        session_id: Optional session tracking for conversation context.

    Returns:
        Comprehensive benchmark report with model rankings, performance insights,
        cost analysis, and specific recommendations for optimal model usage.

    USAGE EXAMPLES:

        # Basic comparison across core capabilities
        result = await model_benchmark(
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"],
            task_types=["coding", "reasoning"],
            sample_size=3
        )

        # Comprehensive evaluation with all task types
        result = await model_benchmark(
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o-mini", "google/gemini-pro-1.5"],
            task_types=["coding", "reasoning", "creative", "analysis", "explanation"],
            evaluation_criteria="comprehensive"
        )

        # Cost-focused benchmark for budget optimization
        result = await model_benchmark(
            models=["qwen3-4b-mlx", "anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
            task_types=["coding", "analysis"],
            evaluation_criteria="speed",
            cost_limit=1.0
        )
    """
    # Get or create session for this request
    session = get_mcp_session(session_id)

    # Update session activity
    session.update_activity()

    logger.info("=== MCP Tool Call: model_benchmark ===")
    logger.info(f"Models: {models}")
    logger.info(f"Task types: {task_types}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Evaluation criteria: {evaluation_criteria}")
    logger.info(f"Cost limit: {cost_limit}")

    try:
        # Import model_benchmark_tool with multiple strategies
        logger.info("Attempting to import model_benchmark_tool...")
        model_benchmark_tool = None
        import_strategies = [
            # Strategy 1: Relative import
            lambda: importlib.import_module(
                ".tools.model_benchmark", package=__package__
            ).model_benchmark_tool,
            # Strategy 2: Absolute import
            lambda: __import__(
                "second_opinion.mcp.tools.model_benchmark",
                fromlist=["model_benchmark_tool"],
            ).model_benchmark_tool,  # type: ignore[attr-defined]
            # Strategy 3: Direct module import
            lambda: __import__(
                "second_opinion.mcp.tools.model_benchmark"
            ).model_benchmark_tool,  # type: ignore[attr-defined]
        ]

        for i, strategy in enumerate(import_strategies, 1):
            try:
                logger.info(f"Trying import strategy {i}...")
                model_benchmark_tool = strategy()
                logger.info(
                    f"✓ Successfully imported model_benchmark_tool using strategy {i}"
                )
                break
            except Exception as e:
                logger.warning(f"✗ Import strategy {i} failed: {e}")
                continue

        if model_benchmark_tool is None:
            # Final fallback: manual importlib approach
            logger.info(
                "All import strategies failed, trying manual step-by-step import..."
            )
            try:
                import importlib

                module = importlib.import_module(
                    "second_opinion.mcp.tools.model_benchmark"
                )
                model_benchmark_tool = module.model_benchmark_tool  # type: ignore[attr-defined]
                logger.info("✓ Successfully imported using manual importlib approach")
            except Exception as final_error:
                logger.error(f"✗ All import methods failed. Final error: {final_error}")
                import traceback

                logger.error(f"Final import traceback:\n{traceback.format_exc()}")
                raise ImportError(
                    f"Unable to import model_benchmark_tool after trying multiple strategies. Last error: {final_error}"
                ) from final_error

        if model_benchmark_tool is None:
            raise ImportError("model_benchmark_tool is None after all import attempts")

        # Call the tool implementation
        logger.info("Calling model_benchmark_tool implementation...")
        result = await model_benchmark_tool(
            models=models,
            task_types=task_types,
            sample_size=sample_size,
            evaluation_criteria=evaluation_criteria,
            cost_limit=cost_limit,
            session_id=session.session_id,
        )
        logger.info("✓ model_benchmark_tool completed successfully")
        logger.info(f"Result length: {len(result) if result else 0}")

        # Add to conversation context
        session.add_conversation_context(
            tool_name="model_benchmark",
            prompt=f"Benchmark {len(models)} models across {len(task_types) if task_types else 3} task types",
            primary_model="benchmark-system",
            comparison_models=models,
            result_summary=f"Benchmark ({len(models)} models) completed successfully",
        )

        return result

    except Exception as e:
        # Log comprehensive error information
        import traceback

        error_details = traceback.format_exc()
        logger.error("=== Error in model_benchmark MCP tool ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")
        logger.error("=== End Error Details ===")

        # Add to context for debugging
        session.add_conversation_context(
            tool_name="model_benchmark",
            prompt=f"Benchmark {len(models)} models across {len(task_types) if task_types else 3} task types",
            primary_model="benchmark-system",
            comparison_models=models,
            result_summary=f"Error: {str(e)}",
        )

        # Return user-friendly error message
        return f"❌ **Model Benchmark Error**: {str(e)}\n\n**Error Type**: {type(e).__name__}\n\nPlease check the server logs for detailed debugging information."


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
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Update our MCP logger setup
    logger = setup_mcp_logging(debug=args.debug)

    logger.info(
        f"Starting Second Opinion MCP server in standalone mode (debug={'on' if args.debug else 'off'})"
    )
    mcp.run()
