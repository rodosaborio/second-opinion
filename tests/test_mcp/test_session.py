"""Test MCP session management."""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.second_opinion.mcp.session import MCPSession
from src.second_opinion.clients.base import ModelInfo


class TestMCPSession:
    """Test MCPSession functionality."""
    
    def test_session_creation(self):
        """Test session creation with and without ID."""
        # Test with auto-generated ID
        session1 = MCPSession()
        assert session1.session_id is not None
        assert len(session1.session_id) > 0
        assert session1.created_at is not None
        assert session1.last_activity == session1.created_at
        
        # Test with provided ID
        session2 = MCPSession("custom-id")
        assert session2.session_id == "custom-id"
        
    def test_cost_tracking(self):
        """Test cost tracking functionality."""
        session = MCPSession("test-session")
        
        # Initial state
        assert session.total_cost == Decimal("0.0")
        assert session.operation_count == 0
        assert len(session.tool_costs) == 0
        
        # Record some costs
        session.record_cost("second_opinion", Decimal("0.05"), "gpt-4")
        assert session.total_cost == Decimal("0.05")
        assert session.operation_count == 1
        assert session.tool_costs["second_opinion"] == Decimal("0.05")
        assert session.last_used_model == "gpt-4"
        
        # Record more costs
        session.record_cost("second_opinion", Decimal("0.03"), "claude-3")
        session.record_cost("compare_responses", Decimal("0.02"), "claude-3")
        
        assert session.total_cost == Decimal("0.10")
        assert session.operation_count == 3
        assert session.tool_costs["second_opinion"] == Decimal("0.08")
        assert session.tool_costs["compare_responses"] == Decimal("0.02")
        assert session.last_used_model == "claude-3"
        
    def test_model_info_caching(self):
        """Test model information caching."""
        session = MCPSession("test-session")
        
        # Create mock model info
        model_info = ModelInfo(
            name="gpt-4",
            provider="openai",
            input_cost_per_1k=Decimal("0.03"),
            output_cost_per_1k=Decimal("0.06"),
            max_tokens=8192,
            supports_system_messages=True
        )
        
        # Cache model info
        session.cache_model_info("gpt-4", model_info)
        
        # Test retrieval
        cached_info = session.get_cached_model_info("gpt-4")
        assert cached_info is not None
        assert cached_info.name == "gpt-4"
        assert cached_info.provider == "openai"
        
        # Test pricing cache
        pricing = session.get_cached_pricing("gpt-4")
        assert pricing is not None
        assert pricing == (Decimal("0.03"), Decimal("0.06"))
        
        # Test non-existent model
        assert session.get_cached_model_info("non-existent") is None
        assert session.get_cached_pricing("non-existent") is None
        
    def test_conversation_context(self):
        """Test conversation context management."""
        session = MCPSession("test-session")
        
        # Add conversation context
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt="What is the capital of France?",
            primary_model="gpt-4",
            comparison_models=["claude-3", "gemini-pro"],
            result_summary="Paris was identified as the correct answer"
        )
        
        assert len(session.conversation_history) == 1
        context = session.conversation_history[0]
        assert context["tool"] == "second_opinion"
        assert context["prompt"] == "What is the capital of France?"
        assert context["primary_model"] == "gpt-4"
        assert context["comparison_models"] == ["claude-3", "gemini-pro"]
        
        # Test truncation for long prompts
        long_prompt = "A" * 300
        session.add_conversation_context(
            tool_name="test_tool",
            prompt=long_prompt,
            primary_model="gpt-4",
            comparison_models=[],
            result_summary="Test"
        )
        
        assert len(session.conversation_history) == 2
        assert len(session.conversation_history[1]["prompt"]) == 200
        
        # Test history limit (max 10 entries)
        for i in range(15):
            session.add_conversation_context(
                tool_name=f"tool_{i}",
                prompt=f"prompt_{i}",
                primary_model="gpt-4",
                comparison_models=[],
                result_summary=f"result_{i}"
            )
        
        assert len(session.conversation_history) == 10
        # Should keep the most recent 10
        assert session.conversation_history[-1]["tool"] == "tool_14"
        
    def test_model_usage_patterns(self):
        """Test model usage pattern analysis."""
        session = MCPSession("test-session")
        
        # Add some conversation history
        models_used = [
            ("gpt-4", ["claude-3"]),
            ("gpt-4", ["gemini-pro"]),
            ("claude-3", ["gpt-4"]),
            ("gpt-4", ["claude-3", "gemini-pro"]),
        ]
        
        for primary, comparisons in models_used:
            session.add_conversation_context(
                tool_name="second_opinion",
                prompt="test prompt",
                primary_model=primary,
                comparison_models=comparisons,
                result_summary="test result"
            )
        
        usage_patterns = session.get_model_usage_patterns()
        
        # gpt-4 appears as primary 3 times + comparison 1 time = 4 total
        # claude-3 appears as primary 1 time + comparison 2 times = 3 total
        # gemini-pro appears as comparison 2 times = 2 total
        assert usage_patterns["gpt-4"] == 4
        assert usage_patterns["claude-3"] == 3
        assert usage_patterns["gemini-pro"] == 2
        
    def test_primary_model_suggestion(self):
        """Test primary model suggestion logic."""
        session = MCPSession("test-session")
        
        # Test with no history
        assert session.suggest_primary_model() is None
        
        # Test with last used model
        session.record_cost("test_tool", Decimal("0.01"), "gpt-4")
        assert session.suggest_primary_model() == "gpt-4"
        
    def test_session_summary(self):
        """Test session summary generation."""
        session = MCPSession("test-session")
        
        # Add some activity
        session.record_cost("second_opinion", Decimal("0.05"), "gpt-4")
        session.add_conversation_context(
            tool_name="second_opinion",
            prompt="test prompt",
            primary_model="gpt-4",
            comparison_models=["claude-3"],
            result_summary="test result"
        )
        
        summary = session.get_session_summary()
        
        assert summary["session_id"] == "test-session"
        assert summary["total_cost"] == 0.05
        assert summary["operation_count"] == 1
        assert summary["tool_costs"]["second_opinion"] == 0.05
        assert summary["conversation_entries"] == 1
        assert summary["last_used_model"] == "gpt-4"
        assert "gpt-4" in summary["model_usage_patterns"]
        
    def test_session_expiration(self):
        """Test session expiration logic."""
        session = MCPSession("test-session")
        
        # Fresh session should not be expired
        assert not session.is_expired(timeout_hours=24)
        
        # Manually set old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        session.last_activity = old_time
        
        assert session.is_expired(timeout_hours=24)
        assert not session.is_expired(timeout_hours=48)
        
    def test_session_reset_functions(self):
        """Test session reset functionality."""
        session = MCPSession("test-session")
        
        # Add some data
        session.record_cost("test_tool", Decimal("0.05"), "gpt-4")
        session.add_conversation_context(
            tool_name="test_tool",
            prompt="test",
            primary_model="gpt-4",
            comparison_models=[],
            result_summary="test"
        )
        
        # Test conversation history clearing
        session.clear_conversation_history()
        assert len(session.conversation_history) == 0
        assert session.total_cost == Decimal("0.05")  # Cost should remain
        
        # Test cost tracking reset
        session.reset_cost_tracking()
        assert session.total_cost == Decimal("0.0")
        assert session.operation_count == 0
        assert len(session.tool_costs) == 0
        assert session.last_used_model is None
        
    def test_activity_update(self):
        """Test activity timestamp update."""
        session = MCPSession("test-session")
        original_time = session.last_activity
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        session.update_activity()
        assert session.last_activity > original_time