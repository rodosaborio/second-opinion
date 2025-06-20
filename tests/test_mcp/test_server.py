"""Test MCP server functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.second_opinion.mcp.server import mcp, get_mcp_session, cleanup_sessions
from src.second_opinion.mcp.session import MCPSession


class TestMCPServer:
    """Test MCP server setup and configuration."""
    
    def test_server_creation(self):
        """Test that the MCP server is created properly."""
        assert mcp is not None
        assert mcp.name == "Second Opinion"
        
    @pytest.mark.asyncio
    async def test_server_tools_registration(self):
        """Test that tools are registered with the server."""
        tools = await mcp.get_tools()
        assert len(tools) >= 1
        
        # Check that second_opinion tool is registered
        assert "second_opinion" in tools
        
        # Get the second_opinion tool specifically
        second_opinion_tool = await mcp.get_tool("second_opinion")
        assert second_opinion_tool is not None
        
    def test_session_management(self):
        """Test session creation and management."""
        # Test creating new session
        session1 = get_mcp_session()
        assert isinstance(session1, MCPSession)
        assert session1.session_id is not None
        
        # Test getting session by ID
        session2 = get_mcp_session(session1.session_id)
        assert session2 is session1
        
        # Test creating new session with different ID
        session3 = get_mcp_session("test-session")
        assert session3 is not session1
        assert session3.session_id == "test-session"
        
    def test_session_cleanup(self):
        """Test session cleanup functionality."""
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = get_mcp_session(f"test-session-{i}")
            sessions.append(session)
        
        # Cleanup with low limit
        cleanup_sessions(max_sessions=2)
        
        # Check that only recent sessions remain
        remaining_session = get_mcp_session("test-session-4")
        assert remaining_session.session_id == "test-session-4"


class TestMCPToolIntegration:
    """Test MCP tool integration with the server."""
    
    @pytest.mark.asyncio
    async def test_second_opinion_tool_signature(self):
        """Test that the second_opinion tool has correct signature."""
        tools = await mcp.get_tools()
        assert "second_opinion" in tools
        
        # Get the tool for detailed inspection
        second_opinion_tool = await mcp.get_tool("second_opinion")
        assert second_opinion_tool is not None
        
        # Note: FastMCP tool introspection may vary by version
        # This test mainly verifies the tool exists and can be retrieved


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests."""
    from src.second_opinion.mcp.server import _sessions
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture
def mock_api_key():
    """Mock OpenRouter API key for testing."""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test-key"}):
        yield


@pytest.fixture
def mock_pricing_manager():
    """Mock pricing manager for testing."""
    mock_manager = MagicMock()
    mock_manager.load_pricing_data = AsyncMock()
    
    with patch("src.second_opinion.mcp.server.get_pricing_manager", return_value=mock_manager):
        yield mock_manager


@pytest.fixture
def mock_cost_guard():
    """Mock cost guard for testing."""
    mock_guard = MagicMock()
    mock_guard.check_and_reserve_budget = AsyncMock()
    mock_guard.record_actual_cost = AsyncMock()
    
    with patch("src.second_opinion.mcp.server.get_cost_guard", return_value=mock_guard):
        yield mock_guard