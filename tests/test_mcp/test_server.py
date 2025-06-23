"""Test MCP server functionality."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.second_opinion.mcp.server import (
    cleanup_sessions,
    get_mcp_session,
    get_tool_description,
    log_environment_info,
    mcp,
    setup_mcp_logging,
    setup_python_path,
)
from src.second_opinion.mcp.server import (
    test_critical_imports as check_critical_imports,
)
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

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """Test that all expected tools are registered."""
        tools = await mcp.get_tools()
        expected_tools = [
            "second_opinion",
            "should_downgrade",
            "should_upgrade",
            "compare_responses",
            "consult",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"


class TestServerUtilities:
    """Test server utility functions."""

    def test_get_tool_description_success(self):
        """Test successful tool description loading."""
        with patch(
            "src.second_opinion.mcp.server.load_mcp_tool_description"
        ) as mock_load:
            mock_load.return_value = "Test description"

            result = get_tool_description("test_tool")
            assert result == "Test description"
            mock_load.assert_called_once_with("test_tool")

    def test_get_tool_description_failure(self):
        """Test tool description loading with fallback."""
        with patch(
            "src.second_opinion.mcp.server.load_mcp_tool_description"
        ) as mock_load:
            mock_load.side_effect = Exception("Template not found")

            result = get_tool_description("missing_tool")
            assert "missing_tool" in result
            assert "description unavailable" in result

    def test_setup_mcp_logging(self):
        """Test MCP logging setup."""
        logger = setup_mcp_logging(debug=True)
        assert logger is not None

        # Test without debug
        logger_no_debug = setup_mcp_logging(debug=False)
        assert logger_no_debug is not None

    def test_setup_python_path(self):
        """Test Python path setup."""
        original_path = sys.path.copy()

        try:
            setup_python_path()
            # Should have added paths without errors
            assert len(sys.path) >= len(original_path)
        finally:
            # Restore original path
            sys.path[:] = original_path

    def test_log_environment_info(self, caplog):
        """Test environment info logging."""
        log_environment_info()

        # Check that key information was logged
        assert "Python executable" in caplog.text
        assert "Working directory" in caplog.text
        assert "sys.path entries" in caplog.text

    def test_test_critical_imports(self):
        """Test critical imports function."""
        successful, failed = check_critical_imports()

        # Should be lists
        assert isinstance(successful, list)
        assert isinstance(failed, list)

        # At least some imports should succeed
        assert len(successful) > 0


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

    with patch(
        "src.second_opinion.mcp.server.get_pricing_manager", return_value=mock_manager
    ):
        yield mock_manager


@pytest.fixture
def mock_cost_guard():
    """Mock cost guard for testing."""
    mock_guard = MagicMock()
    mock_guard.check_and_reserve_budget = AsyncMock()
    mock_guard.record_actual_cost = AsyncMock()

    with patch("src.second_opinion.mcp.server.get_cost_guard", return_value=mock_guard):
        yield mock_guard
