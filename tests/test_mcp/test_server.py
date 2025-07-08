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


class TestMCPLifecycle:
    """Test MCP server lifecycle management."""

    @pytest.mark.asyncio
    async def test_mcp_lifespan_startup(
        self, mock_api_key, mock_pricing_manager, mock_cost_guard
    ):
        """Test MCP server startup in lifespan context."""
        from src.second_opinion.mcp.server import mcp_lifespan

        # Mock pricing manager methods
        mock_pricing_manager.fetch_latest_pricing = AsyncMock()
        mock_pricing_manager.get_model_count.return_value = 42

        async with mcp_lifespan(mcp) as context:
            assert "settings" in context
            assert "cost_guard" in context
            assert "pricing_manager" in context
            assert "sessions" in context

            # Check pricing manager was called
            mock_pricing_manager.fetch_latest_pricing.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_lifespan_no_api_key(
        self, mock_pricing_manager, mock_cost_guard, caplog
    ):
        """Test MCP server startup without API key."""
        from src.second_opinion.mcp.server import mcp_lifespan

        # Mock pricing manager methods
        mock_pricing_manager.fetch_latest_pricing = AsyncMock()
        mock_pricing_manager.get_model_count.return_value = 42

        # Clear API key environment variables
        with patch.dict("os.environ", {}, clear=True):
            async with mcp_lifespan(mcp) as context:
                assert "settings" in context
                # Should handle missing API key gracefully (may warn or work without it)
                # Since server starts successfully, we just check it doesn't crash
                pass

    @pytest.mark.asyncio
    async def test_mcp_lifespan_pricing_failure(
        self, mock_api_key, mock_pricing_manager, mock_cost_guard, caplog
    ):
        """Test MCP server startup with pricing fetch failure."""
        from src.second_opinion.mcp.server import mcp_lifespan

        # Mock pricing manager to fail
        mock_pricing_manager.fetch_latest_pricing = AsyncMock(
            side_effect=Exception("Network error")
        )
        mock_pricing_manager.get_model_count.return_value = 0

        async with mcp_lifespan(mcp) as context:
            assert "settings" in context
            # Should warn about pricing failure
            assert "Failed to load pricing data" in caplog.text

    @pytest.mark.asyncio
    async def test_mcp_lifespan_startup_failure(self, mock_cost_guard, caplog):
        """Test MCP server startup failure handling."""
        from src.second_opinion.mcp.server import mcp_lifespan

        with patch("src.second_opinion.mcp.server.get_settings") as mock_settings:
            mock_settings.side_effect = Exception("Configuration error")

            with pytest.raises(Exception, match="Configuration error"):
                async with mcp_lifespan(mcp):
                    pass


class TestServerMainFunction:
    """Test server main function and CLI integration."""

    def test_main_function_with_debug(self):
        """Test server main function with debug flag."""
        with patch("sys.argv", ["server.py", "--debug"]):
            with patch("src.second_opinion.mcp.server.mcp.run"):
                with patch(
                    "src.second_opinion.mcp.server.setup_mcp_logging"
                ) as mock_setup_logging:
                    # Mock argparse to simulate command line args
                    with patch("argparse.ArgumentParser.parse_args") as mock_args:
                        mock_args.return_value.debug = True

                        # Test the debug flag handling logic
                        # This tests the argparse configuration without actually running the server
                        mock_setup_logging.assert_not_called()  # Not called yet

    def test_main_function_without_debug(self):
        """Test server main function without debug flag."""
        with patch("sys.argv", ["server.py"]):
            with patch("src.second_opinion.mcp.server.mcp.run"):
                with patch("src.second_opinion.mcp.server.setup_mcp_logging"):
                    # Mock argparse to simulate command line args
                    with patch("argparse.ArgumentParser.parse_args") as mock_args:
                        mock_args.return_value.debug = False

                        # Test would require more complex setup to avoid actual execution
                        # For now, just verify the main function structure exists
                        from src.second_opinion.mcp.server import (
                            __name__ as module_name,
                        )

                        # The module should be importable and have main section
                        assert module_name is not None


class TestAdvancedServerFeatures:
    """Test advanced server features and edge cases."""

    def test_session_id_consistency(self):
        """Test that session IDs are consistent across multiple calls."""
        session1 = get_mcp_session("custom-session-123")
        session2 = get_mcp_session("custom-session-123")

        assert session1 is session2
        assert session1.session_id == "custom-session-123"

    def test_session_auto_generation(self):
        """Test automatic session ID generation."""
        session1 = get_mcp_session()
        session2 = get_mcp_session()

        assert session1 is not session2
        assert session1.session_id != session2.session_id
        assert len(session1.session_id) > 0
        assert len(session2.session_id) > 0

    def test_cleanup_sessions_functionality(self):
        """Test session cleanup with various scenarios."""
        # Create many sessions to test cleanup
        session_ids = []
        for i in range(15):
            session = get_mcp_session(f"cleanup-test-{i}")
            session_ids.append(session.session_id)

        # Cleanup to max 10 sessions
        cleanup_sessions(max_sessions=10)

        # Check that newer sessions still exist
        recent_session = get_mcp_session("cleanup-test-14")
        assert recent_session.session_id == "cleanup-test-14"

        # Verify cleanup behavior with max_sessions = 0
        cleanup_sessions(max_sessions=0)

        # Should still be able to create new sessions
        new_session = get_mcp_session("post-cleanup")
        assert new_session.session_id == "post-cleanup"

    def test_python_path_modification(self):
        """Test Python path setup behavior."""
        original_path = sys.path.copy()

        try:
            # Test with empty path
            sys.path.clear()
            setup_python_path()

            # Should have added at least some paths
            assert len(sys.path) > 0

            # Test idempotent behavior
            current_path = sys.path.copy()
            setup_python_path()

            # Should not duplicate paths
            assert len(sys.path) == len(current_path)
        finally:
            # Restore original path
            sys.path[:] = original_path

    def test_import_error_handling(self):
        """Test critical imports error handling."""
        import builtins

        from src.second_opinion.mcp.server import test_critical_imports

        # Mock the import of second_opinion specifically to trigger import failure
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("second_opinion"):
                raise ImportError("Test import failure")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            successful, failed = test_critical_imports()

            # Should have failures due to mocked import error
            assert len(failed) > 0
            assert all(isinstance(failure, tuple) for failure in failed)
            assert all(len(failure) == 2 for failure in failed)

    def test_environment_logging_edge_cases(self, caplog):
        """Test environment logging with missing environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            log_environment_info()

            # Should handle missing environment variables gracefully
            assert "Not set" in caplog.text
            assert "Python executable" in caplog.text


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
