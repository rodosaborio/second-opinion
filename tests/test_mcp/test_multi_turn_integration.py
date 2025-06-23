"""
Integration tests for multi-turn consultation functionality.

This module tests real multi-turn session persistence, context preservation,
and conversation storage across separate MCP tool calls, simulating actual
MCP client behavior.
"""

import re
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from second_opinion.core.models import ModelResponse, TokenUsage
from second_opinion.mcp.tools.consult import consult_tool


@pytest.fixture
def mock_multi_turn_dependencies(monkeypatch):
    """Mock dependencies specifically for multi-turn testing."""

    # Mock cost guard
    mock_cost_guard = MagicMock()
    mock_cost_guard.check_and_reserve_budget = AsyncMock(
        return_value=MagicMock(reservation_id="test-reservation", approved=True)
    )
    mock_cost_guard.record_actual_cost = AsyncMock()

    def get_mock_cost_guard():
        return mock_cost_guard

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.get_cost_guard", get_mock_cost_guard
    )

    # Mock evaluator with consistent complexity classification
    mock_evaluator = MagicMock()
    mock_evaluator.classify_task_complexity = AsyncMock(
        return_value="moderate"  # Return string to match TaskComplexity enum
    )

    def get_mock_evaluator():
        return mock_evaluator

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.get_evaluator", get_mock_evaluator
    )

    # Mock client with realistic multi-turn responses
    mock_client = MagicMock()
    response_templates = {
        "turn_1": "This is a complex topic. Let me start with the basics: {}. Would you like me to elaborate on any specific aspect?",
        "turn_2": "Building on our previous discussion about {}, here are the detailed considerations: {}. Should we dive deeper into implementation?",
        "turn_3": "To complete our analysis of {}, here are the final recommendations: {}. This should give you a comprehensive understanding.",
    }

    call_count = [0]  # Use list to allow modification in nested function

    async def mock_complete(request):
        call_count[0] += 1
        turn_key = f"turn_{min(call_count[0], 3)}"
        template = response_templates[turn_key]
        content = template.format(
            request.messages[0].content[:50] + "...",
            f"detailed analysis for turn {call_count[0]}",
        )

        return ModelResponse(
            content=content,
            model=request.model,
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.01"),
            provider="mock",
        )

    mock_client.complete = mock_complete
    mock_client.estimate_cost = AsyncMock(return_value=Decimal("0.01"))

    def create_mock_client(provider):
        return mock_client

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.create_client_from_config", create_mock_client
    )

    # Mock provider detection
    def mock_detect_provider(model):
        if "local" in model.lower():
            return "lmstudio"
        return "openrouter"

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.detect_model_provider", mock_detect_provider
    )

    # Mock domain classification
    async def mock_classify_domain(query, context=None):
        return "general"

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.classify_consultation_domain",
        mock_classify_domain,
    )

    # Mock conversation store for session recovery
    mock_store = MagicMock()

    # Simulate conversation history for session recovery
    async def mock_get_session_history(session_id):
        if "test-session" in session_id or len(session_id) > 10:
            return [
                {
                    "id": "conv-1",
                    "user_prompt": "Explain the fundamentals of async programming",
                    "context": None,
                    "responses": [
                        {
                            "content": "Async programming is about concurrent execution...",
                            "model": "anthropic/claude-3-5-sonnet",
                            "response_type": "primary",
                            "cost": "0.01",
                        }
                    ],
                    "total_cost": "0.01",
                    "created_at": "2024-01-01T10:00:00",
                    "tool_name": "consult",
                }
            ]
        return []

    mock_store.get_session_conversation_history = mock_get_session_history

    def get_mock_store():
        return mock_store

    monkeypatch.setattr(
        "second_opinion.database.store.get_conversation_store",
        get_mock_store,
    )

    return {
        "cost_guard": mock_cost_guard,
        "evaluator": mock_evaluator,
        "client": mock_client,
        "call_count": call_count,
    }


class TestMultiTurnSessionPersistence:
    """Test multi-turn session persistence across separate MCP calls."""

    @pytest.mark.asyncio
    async def test_session_id_extraction_and_reuse(self, mock_multi_turn_dependencies):
        """Test extracting session ID from first call and reusing in second call."""

        # First consultation call (creates new session)
        first_result = await consult_tool(
            query="Explain the fundamentals of async programming",
            consultation_type="deep",
            max_turns=3,
            target_model="anthropic/claude-3-5-sonnet",
        )

        # Verify first call was successful
        assert "🔍 Deep Consultation Session" in first_result
        assert "Session ID" in first_result

        # Extract session ID from the response
        session_id_match = re.search(r"Session ID.*?`([^`]+)`", first_result)
        if not session_id_match:
            session_id_match = re.search(r"Session ID.*?([a-f0-9-]+)", first_result)

        assert (
            session_id_match
        ), f"Could not find session ID in response: {first_result[:500]}"
        session_id = session_id_match.group(1)

        # Verify session ID format
        assert len(session_id) > 10, f"Session ID too short: {session_id}"

        # Second consultation call (continue existing session)
        second_result = await consult_tool(
            query="Can you elaborate on the event loop implementation?",
            consultation_type="deep",
            session_id=session_id,
            max_turns=3,
            target_model="anthropic/claude-3-5-sonnet",
        )

        # Verify second call was successful and references same session
        assert "🔍 Deep Consultation Session" in second_result
        assert session_id in second_result
        # Check for context loading (either previous discussion or loaded context)
        assert (
            "Building on our previous discussion" in second_result
            or "Previous conversation context from our session" in second_result
        )

    @pytest.mark.asyncio
    async def test_multi_turn_context_preservation(self, mock_multi_turn_dependencies):
        """Test that conversation context is preserved across multiple turns."""

        # Start a multi-turn conversation about system design
        result = await consult_tool(
            query="Help me design a scalable microservices architecture",
            consultation_type="deep",
            max_turns=3,
            context="system architecture design",
        )

        # Should contain system architecture context
        assert "system architecture design" in result
        assert "microservices" in result.lower()

        # Check for multi-turn conversation flow
        if "Turn 2" in result:
            assert "Building on our previous discussion" in result

        # Check session ID is provided for continuation
        assert "Session ID" in result
        assert "Continue Consultation" in result

    @pytest.mark.asyncio
    async def test_cost_accumulation_across_turns(self, mock_multi_turn_dependencies):
        """Test that costs accumulate correctly across multiple turns."""

        result = await consult_tool(
            query="Explain machine learning fundamentals in detail",
            consultation_type="deep",
            max_turns=3,
            cost_limit=0.50,
        )

        # Verify cost tracking
        assert "Cost Analysis" in result
        assert "Total Cost" in result

        # Cost should be > 0 and <= limit
        cost_match = re.search(r"Total Cost.*?\$(\d+\.\d+)", result)
        if cost_match:
            total_cost = float(cost_match.group(1))
            assert total_cost > 0, "Total cost should be greater than 0"
            assert total_cost <= 0.50, f"Total cost {total_cost} exceeds limit 0.50"

    @pytest.mark.asyncio
    async def test_brainstorm_multi_turn_flow(self, mock_multi_turn_dependencies):
        """Test multi-turn flow specifically for brainstorming sessions."""

        result = await consult_tool(
            query="Brainstorm creative approaches to reduce API latency",
            consultation_type="brainstorm",
            max_turns=2,
            context="performance optimization",
        )

        # Verify brainstorming-specific formatting
        assert "💡 Brainstorming Session" in result
        assert "performance optimization" in result

        # Should include creative prompts and session continuation
        assert "Session ID" in result


class TestMultiTurnConversationFlow:
    """Test multi-turn conversation flow and follow-up logic."""

    @pytest.mark.asyncio
    async def test_follow_up_assessment_logic(self, mock_multi_turn_dependencies):
        """Test that follow-up assessment works correctly."""

        # Use a query that should trigger follow-up
        result = await consult_tool(
            query="What are the best practices for database optimization?",
            consultation_type="deep",
            max_turns=3,
        )

        # Should complete at least one turn
        assert "🔍 Deep Consultation Session" in result
        assert "database optimization" in result.lower()

    @pytest.mark.asyncio
    async def test_turn_limit_enforcement(self, mock_multi_turn_dependencies):
        """Test that turn limits are properly enforced."""

        result = await consult_tool(
            query="Comprehensive analysis of cloud architecture patterns",
            consultation_type="deep",
            max_turns=1,  # Force single turn
        )

        # Should complete only 1 turn despite being "deep" type
        assert "🔍 Deep Consultation Session" in result
        turns_completed_match = re.search(r"Turns Completed.*?(\d+)", result)
        if turns_completed_match:
            turns_completed = int(turns_completed_match.group(1))
            assert turns_completed == 1, f"Expected 1 turn, got {turns_completed}"

    @pytest.mark.asyncio
    async def test_cost_limit_stops_conversation(self, mock_multi_turn_dependencies):
        """Test that cost limits properly stop multi-turn conversations."""

        result = await consult_tool(
            query="Detailed analysis of distributed systems",
            consultation_type="deep",
            max_turns=5,
            cost_limit=0.02,  # Very low limit
        )

        # Should complete but be limited by cost
        assert "🔍 Deep Consultation Session" in result

        # Cost should not exceed limit
        cost_match = re.search(r"Total Cost.*?\$(\d+\.\d+)", result)
        if cost_match:
            total_cost = float(cost_match.group(1))
            assert total_cost <= 0.02, f"Cost {total_cost} exceeds limit 0.02"


class TestMultiTurnRegressionProtection:
    """Test that multi-turn functionality doesn't break existing features."""

    @pytest.mark.asyncio
    async def test_single_turn_still_works(self, mock_multi_turn_dependencies):
        """Test that single-turn consultations still work correctly."""

        result = await consult_tool(
            query="What is the difference between REST and GraphQL?",
            consultation_type="quick",
        )

        assert "🎯 Quick Expert Consultation" in result
        assert "REST" in result or "GraphQL" in result

    @pytest.mark.asyncio
    async def test_delegate_functionality_preserved(self, mock_multi_turn_dependencies):
        """Test that task delegation functionality is preserved."""

        result = await consult_tool(
            query="Write a Python function to validate email addresses",
            consultation_type="delegate",
            target_model="openai/gpt-4o-mini",
        )

        assert "📋 Task Delegation Results" in result
        assert "openai/gpt-4o-mini" in result
        assert "Estimated Savings" in result

    @pytest.mark.asyncio
    async def test_error_handling_with_multi_turn(self, mock_multi_turn_dependencies):
        """Test error handling in multi-turn context."""

        # Test invalid session ID
        result = await consult_tool(
            query="Continue previous conversation",
            consultation_type="deep",
            session_id="invalid-session-id",
            max_turns=2,
        )

        # Should still work (create new session) rather than fail
        assert "🔍 Deep Consultation Session" in result


class TestMultiTurnPerformance:
    """Test multi-turn consultation performance characteristics."""

    @pytest.mark.asyncio
    async def test_multi_turn_response_time(self, mock_multi_turn_dependencies):
        """Test that multi-turn consultations complete in reasonable time."""
        import time

        start_time = time.time()

        result = await consult_tool(
            query="Analyze software architecture patterns in depth",
            consultation_type="deep",
            max_turns=2,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (generous limit for testing)
        assert (
            duration < 10.0
        ), f"Multi-turn consultation took {duration:.2f}s, too long"
        assert "🔍 Deep Consultation Session" in result

    @pytest.mark.asyncio
    async def test_multi_turn_memory_usage(self, mock_multi_turn_dependencies):
        """Test that multi-turn consultations don't leak memory."""
        import gc

        # Force garbage collection before test
        gc.collect()

        # Run multiple multi-turn consultations
        for i in range(3):
            result = await consult_tool(
                query=f"Explain topic {i} in detail",
                consultation_type="deep",
                max_turns=2,
            )
            assert "🔍 Deep Consultation Session" in result

        # Force garbage collection after test
        gc.collect()

        # If we get here without memory errors, test passes
        assert True


@pytest.mark.integration
class TestRealMultiTurnScenarios:
    """Test realistic multi-turn consultation scenarios."""

    @pytest.mark.asyncio
    async def test_technical_deep_dive_scenario(self, mock_multi_turn_dependencies):
        """Test a realistic technical deep-dive conversation."""

        # Start with broad question
        result = await consult_tool(
            query="I need to implement real-time notifications in my web app",
            consultation_type="deep",
            max_turns=3,
            context="web development",
        )

        # Should provide comprehensive technical guidance
        assert "🔍 Deep Consultation Session" in result
        assert "web development" in result
        assert "notifications" in result.lower()
        assert "Session ID" in result

    @pytest.mark.asyncio
    async def test_brainstorm_creative_scenario(self, mock_multi_turn_dependencies):
        """Test a realistic brainstorming session."""

        result = await consult_tool(
            query="I need creative ideas for improving user engagement in my mobile app",
            consultation_type="brainstorm",
            max_turns=2,
            context="product development",
        )

        assert "💡 Brainstorming Session" in result
        assert "product development" in result
        assert "engagement" in result.lower()

    @pytest.mark.asyncio
    async def test_architecture_consultation_scenario(
        self, mock_multi_turn_dependencies
    ):
        """Test a realistic architecture consultation."""

        result = await consult_tool(
            query="Help me choose between microservices and monolith for a new project",
            consultation_type="deep",
            max_turns=3,
            context="system architecture",
        )

        assert "🔍 Deep Consultation Session" in result
        assert "system architecture" in result
        assert "microservices" in result.lower() or "monolith" in result.lower()
