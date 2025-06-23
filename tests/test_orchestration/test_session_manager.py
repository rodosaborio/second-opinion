"""
Tests for session manager functionality.
"""

import re
from unittest.mock import patch

from src.second_opinion.orchestration.session_manager import (
    generate_cli_session_id,
    generate_mcp_session_id,
)


class TestSessionManager:
    """Test session manager utility functions."""

    def test_generate_cli_session_id(self):
        """Test CLI session ID generation."""
        session_id = generate_cli_session_id()

        # Check format: cli-{YYYYMMDD-HHMM}-{uuid8}
        pattern = r"^cli-\d{8}-\d{4}-[a-f0-9]{8}$"
        assert re.match(pattern, session_id), f"Session ID format invalid: {session_id}"

        # Check uniqueness
        session_id2 = generate_cli_session_id()
        assert session_id != session_id2

    def test_generate_mcp_session_id(self):
        """Test MCP session ID generation."""
        session_id = generate_mcp_session_id()

        # Check format: mcp-{uuid}
        pattern = r"^mcp-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
        assert re.match(pattern, session_id), f"Session ID format invalid: {session_id}"

        # Check uniqueness
        session_id2 = generate_mcp_session_id()
        assert session_id != session_id2

    def test_cli_session_id_uniqueness(self):
        """Test that CLI session IDs are unique across multiple generations."""
        session_ids = set()

        for _ in range(10):
            session_id = generate_cli_session_id()
            assert session_id not in session_ids
            session_ids.add(session_id)

    def test_mcp_session_id_uniqueness(self):
        """Test that MCP session IDs are unique across multiple generations."""
        session_ids = set()

        for _ in range(10):
            session_id = generate_mcp_session_id()
            assert session_id not in session_ids
            session_ids.add(session_id)

    def test_session_id_prefixes(self):
        """Test that session IDs have correct prefixes."""
        cli_id = generate_cli_session_id()
        mcp_id = generate_mcp_session_id()

        assert cli_id.startswith("cli-")
        assert mcp_id.startswith("mcp-")

    def test_cli_session_id_contains_timestamp(self):
        """Test that CLI session ID contains current timestamp."""
        from datetime import datetime

        # Get the current timestamp in the expected format
        expected_timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        session_id = generate_cli_session_id()

        # The session ID should contain the timestamp
        assert expected_timestamp in session_id

    def test_session_id_logging(self, caplog):
        """Test that session ID generation includes debug logging."""
        with patch(
            "src.second_opinion.orchestration.session_manager.logger"
        ) as mock_logger:
            generate_cli_session_id()
            mock_logger.debug.assert_called()

    def test_mcp_session_id_format_consistency(self):
        """Test MCP session ID format consistency."""
        session_ids = [generate_mcp_session_id() for _ in range(5)]

        for session_id in session_ids:
            parts = session_id.split("-")
            assert len(parts) == 6  # mcp, plus 5 UUID parts
            assert parts[0] == "mcp"

            # Verify UUID format
            uuid_part = "-".join(parts[1:])
            assert len(uuid_part) == 36  # Standard UUID length

    def test_cli_session_id_format_consistency(self):
        """Test CLI session ID format consistency."""
        session_ids = [generate_cli_session_id() for _ in range(5)]

        for session_id in session_ids:
            parts = session_id.split("-")
            assert len(parts) == 4  # cli, YYYYMMDD, HHMM, uuid8
            assert parts[0] == "cli"
            assert len(parts[1]) == 8  # YYYYMMDD
            assert len(parts[2]) == 4  # HHMM
            assert len(parts[3]) == 8  # uuid8

    def test_session_id_generation_performance(self):
        """Test session ID generation performance."""
        import time

        # Generate many session IDs quickly
        start_time = time.time()

        for _ in range(100):
            generate_cli_session_id()
            generate_mcp_session_id()

        elapsed_time = time.time() - start_time

        # Should be very fast (under 1 second for 200 generations)
        assert elapsed_time < 1.0
