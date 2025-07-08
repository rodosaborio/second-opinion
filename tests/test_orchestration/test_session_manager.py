"""
Tests for session manager functionality.
"""

import re
from unittest.mock import patch

from src.second_opinion.orchestration.session_manager import (
    detect_interface_from_session_id,
    generate_cli_session_id,
    generate_mcp_session_id,
    is_valid_session_id,
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

    def test_detect_interface_from_session_id_cli(self):
        """Test interface detection for CLI session IDs."""
        cli_session_id = generate_cli_session_id()
        interface = detect_interface_from_session_id(cli_session_id)
        assert interface == "cli"

    def test_detect_interface_from_session_id_mcp(self):
        """Test interface detection for MCP session IDs."""
        mcp_session_id = generate_mcp_session_id()
        interface = detect_interface_from_session_id(mcp_session_id)
        assert interface == "mcp"

    def test_detect_interface_from_session_id_unknown(self):
        """Test interface detection for unknown session ID formats."""
        unknown_session_id = "unknown-format-12345"
        interface = detect_interface_from_session_id(unknown_session_id)
        assert interface == "unknown"

    def test_detect_interface_from_session_id_empty(self):
        """Test interface detection for empty session ID."""
        interface = detect_interface_from_session_id("")
        assert interface == "unknown"

    def test_detect_interface_from_session_id_malformed_cli(self):
        """Test interface detection for malformed CLI session ID."""
        malformed_cli_id = "cli-invalid"
        interface = detect_interface_from_session_id(malformed_cli_id)
        assert interface == "cli"  # Still detected as CLI due to prefix

    def test_detect_interface_from_session_id_malformed_mcp(self):
        """Test interface detection for malformed MCP session ID."""
        malformed_mcp_id = "mcp-invalid"
        interface = detect_interface_from_session_id(malformed_mcp_id)
        assert interface == "mcp"  # Still detected as MCP due to prefix

    def test_detect_interface_logging_unknown(self, caplog):
        """Test that unknown session ID formats trigger warning logs."""
        with patch(
            "src.second_opinion.orchestration.session_manager.logger"
        ) as mock_logger:
            detect_interface_from_session_id("unknown-format")
            mock_logger.warning.assert_called()

    def test_is_valid_session_id_cli_valid(self):
        """Test validation of valid CLI session IDs."""
        valid_cli_id = generate_cli_session_id()
        assert is_valid_session_id(valid_cli_id) is True

    def test_is_valid_session_id_mcp_valid(self):
        """Test validation of valid MCP session IDs."""
        valid_mcp_id = generate_mcp_session_id()
        assert is_valid_session_id(valid_mcp_id) is True

    def test_is_valid_session_id_empty(self):
        """Test validation of empty session ID."""
        assert is_valid_session_id("") is False
        # Test None handling (needs type ignore for testing edge case)
        assert is_valid_session_id(None) is False  # type: ignore[arg-type]

    def test_is_valid_session_id_cli_invalid_format(self):
        """Test validation of invalid CLI session ID formats."""
        # Missing parts
        assert is_valid_session_id("cli-20241222") is False

        # Wrong part lengths
        assert is_valid_session_id("cli-2024122-1430-a1b2c3d4") is False  # Short date
        assert is_valid_session_id("cli-20241222-143-a1b2c3d4") is False  # Short time
        assert is_valid_session_id("cli-20241222-1430-a1b2c3") is False  # Short UUID

        # Too many parts
        assert is_valid_session_id("cli-20241222-1430-a1b2c3d4-extra") is False

    def test_is_valid_session_id_mcp_invalid_format(self):
        """Test validation of invalid MCP session ID formats."""
        # Missing UUID part
        assert is_valid_session_id("mcp-") is False

        # Wrong UUID length
        assert is_valid_session_id("mcp-short-uuid") is False

        # Not a proper UUID format
        assert is_valid_session_id("mcp-not-a-valid-uuid-format-here") is False

    def test_is_valid_session_id_unknown_format(self):
        """Test validation of unknown session ID formats."""
        assert is_valid_session_id("unknown-format") is False
        assert is_valid_session_id("random-string-12345") is False
        assert is_valid_session_id("12345-67890") is False

    def test_is_valid_session_id_edge_cases(self):
        """Test validation of edge case session IDs."""
        # Just prefixes
        assert is_valid_session_id("cli-") is False
        assert is_valid_session_id("mcp-") is False

        # Prefixes without proper format
        assert is_valid_session_id("cli") is False
        assert is_valid_session_id("mcp") is False

    def test_session_id_validation_consistency(self):
        """Test that generated session IDs always pass validation."""
        # Generate multiple session IDs and ensure they all validate
        for _ in range(20):
            cli_id = generate_cli_session_id()
            mcp_id = generate_mcp_session_id()

            assert is_valid_session_id(cli_id) is True
            assert is_valid_session_id(mcp_id) is True

    def test_interface_detection_validation_consistency(self):
        """Test consistency between interface detection and validation."""
        # Generate session IDs and ensure interface detection matches validation
        cli_id = generate_cli_session_id()
        mcp_id = generate_mcp_session_id()

        # Valid IDs should have correct interface types
        assert detect_interface_from_session_id(cli_id) == "cli"
        assert detect_interface_from_session_id(mcp_id) == "mcp"
        assert is_valid_session_id(cli_id) is True
        assert is_valid_session_id(mcp_id) is True

        # Invalid IDs should be detected but not validate
        invalid_cli = "cli-invalid-format"
        invalid_mcp = "mcp-invalid-format"

        assert detect_interface_from_session_id(invalid_cli) == "cli"
        assert detect_interface_from_session_id(invalid_mcp) == "mcp"
        assert is_valid_session_id(invalid_cli) is False
        assert is_valid_session_id(invalid_mcp) is False

    def test_session_id_components_extraction(self):
        """Test extraction of components from session IDs."""
        cli_id = generate_cli_session_id()
        parts = cli_id.split("-")

        # Verify CLI components
        assert parts[0] == "cli"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 4  # HHMM
        assert len(parts[3]) == 8  # UUID8

        # Verify timestamp is numeric
        assert parts[1].isdigit()
        assert parts[2].isdigit()

        # Verify UUID part is hex
        assert all(c in "0123456789abcdef" for c in parts[3])

    def test_mcp_session_id_uuid_format(self):
        """Test that MCP session IDs contain valid UUID format."""
        mcp_id = generate_mcp_session_id()
        uuid_part = mcp_id[4:]  # Remove "mcp-" prefix

        # UUID4 format: 8-4-4-4-12 hex digits
        uuid_pattern = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
        assert re.match(uuid_pattern, uuid_part)

    def test_cli_session_id_timestamp_accuracy(self):
        """Test that CLI session ID timestamps are accurate within reasonable bounds."""
        import time
        from datetime import datetime

        before_time = datetime.now()
        time.sleep(0.001)  # Small delay to ensure timestamp precision

        cli_id = generate_cli_session_id()

        time.sleep(0.001)
        after_time = datetime.now()

        # Extract timestamp from session ID
        parts = cli_id.split("-")
        session_timestamp_str = f"{parts[1]}-{parts[2]}"
        session_timestamp = datetime.strptime(session_timestamp_str, "%Y%m%d-%H%M")

        # Should be within the time range (with minute precision)
        # Account for minute boundary crossings
        assert session_timestamp.replace(
            second=0, microsecond=0
        ) == before_time.replace(second=0, microsecond=0) or session_timestamp.replace(
            second=0, microsecond=0
        ) == after_time.replace(second=0, microsecond=0)

    def test_batch_session_id_generation(self):
        """Test generating many session IDs in batch."""
        cli_ids = [generate_cli_session_id() for _ in range(50)]
        mcp_ids = [generate_mcp_session_id() for _ in range(50)]

        # All should be unique
        assert len(set(cli_ids)) == len(cli_ids)
        assert len(set(mcp_ids)) == len(mcp_ids)

        # All should be valid
        assert all(is_valid_session_id(sid) for sid in cli_ids)
        assert all(is_valid_session_id(sid) for sid in mcp_ids)

        # All should have correct interface types
        assert all(detect_interface_from_session_id(sid) == "cli" for sid in cli_ids)
        assert all(detect_interface_from_session_id(sid) == "mcp" for sid in mcp_ids)
