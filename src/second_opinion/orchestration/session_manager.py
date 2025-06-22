"""
Session management utilities for conversation orchestration.

This module provides session ID generation strategies that differ between
CLI and MCP interfaces to ensure proper conversation grouping.
"""

import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


def generate_cli_session_id() -> str:
    """
    Generate a unique session ID for CLI interactions.

    CLI sessions are unique per command execution to provide clear
    conversation boundaries in storage and analysis.

    Format: cli-{YYYYMMDD-HHMM}-{uuid8}
    Example: cli-20241222-1430-a1b2c3d4

    Returns:
        Unique CLI session identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    uuid_suffix = uuid4().hex[:8]
    session_id = f"cli-{timestamp}-{uuid_suffix}"

    logger.debug(f"Generated CLI session ID: {session_id}")
    return session_id


def generate_mcp_session_id() -> str:
    """
    Generate a unique session ID for MCP interactions.

    MCP sessions are designed to persist across multiple tool calls
    within the same conversation context.

    Format: mcp-{uuid}
    Example: mcp-a1b2c3d4-e5f6-7890-abcd-ef1234567890

    Returns:
        Unique MCP session identifier
    """
    session_id = f"mcp-{uuid4()}"

    logger.debug(f"Generated MCP session ID: {session_id}")
    return session_id


def detect_interface_from_session_id(session_id: str) -> str:
    """
    Detect interface type from session ID format.

    This utility function allows determining whether a session
    originated from CLI or MCP based on the ID format.

    Args:
        session_id: Session identifier to analyze

    Returns:
        Interface type: "cli", "mcp", or "unknown"
    """
    if session_id.startswith("cli-"):
        return "cli"
    elif session_id.startswith("mcp-"):
        return "mcp"
    else:
        logger.warning(f"Unknown session ID format: {session_id}")
        return "unknown"


def is_valid_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier to validate

    Returns:
        True if session ID follows expected format
    """
    if not session_id:
        return False

    interface_type = detect_interface_from_session_id(session_id)

    if interface_type == "cli":
        # CLI format: cli-YYYYMMDD-HHMM-{8 char hex}
        parts = session_id.split("-")
        return (
            len(parts) == 4
            and parts[0] == "cli"
            and len(parts[1]) == 8  # YYYYMMDD
            and len(parts[2]) == 4  # HHMM
            and len(parts[3]) == 8
        )  # uuid8

    elif interface_type == "mcp":
        # MCP format: mcp-{uuid4}
        parts = session_id.split("-", 1)
        return (
            len(parts) == 2 and parts[0] == "mcp" and len(parts[1]) == 36
        )  # UUID4 length

    return False
