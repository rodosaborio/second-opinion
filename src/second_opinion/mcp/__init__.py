"""
MCP (Model Context Protocol) integration for Second Opinion.

This package provides FastMCP-based server implementation with tools for
AI model comparison, cost optimization, and usage analytics.
"""

from .server import mcp
from .session import MCPSession

__all__ = ["mcp", "MCPSession"]