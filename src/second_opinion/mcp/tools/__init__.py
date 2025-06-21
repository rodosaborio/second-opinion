"""
MCP tools for Second Opinion.

This package contains all MCP tool implementations using FastMCP decorators,
integrating with the existing evaluation engine, cost tracking, and client systems.
"""

from .second_opinion import second_opinion_tool
from .should_downgrade import should_downgrade_tool
from .should_upgrade import should_upgrade_tool
from .compare_responses import compare_responses_tool
from .consult import consult_tool

__all__ = [
    "second_opinion_tool",
    "should_downgrade_tool",
    "should_upgrade_tool",
    "compare_responses_tool",
    "consult_tool",
]
