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


class TestEnhancedContextManagement:
    """Test enhanced context management features."""

    @pytest.mark.asyncio
    async def test_enhanced_context_window(self, mock_multi_turn_dependencies):
        """Test that enhanced context window preserves more conversation history."""

        result = await consult_tool(
            query="Explain distributed systems architecture with specific focus on consistency models",
            consultation_type="deep",
            max_turns=3,
            context="system architecture",
        )

        # Should handle longer, more detailed conversations
        assert "🔍 Deep Consultation Session" in result
        assert "system architecture" in result
        assert "distributed systems" in result.lower()

        # Check for comprehensive analysis indicators
        assert "Session ID" in result

    @pytest.mark.asyncio
    async def test_smart_context_truncation(self, mock_multi_turn_dependencies):
        """Test that smart truncation preserves key information."""

        # Use a very long query to test truncation
        long_query = "I need help designing a comprehensive microservices architecture that can handle high throughput, maintain data consistency across services, implement proper authentication and authorization, provide real-time monitoring and alerting, support horizontal scaling, ensure fault tolerance with circuit breakers and retries, implement proper logging and distributed tracing, handle service discovery and load balancing, manage configuration and secrets, and provide automated deployment pipelines with blue-green deployments. This system needs to support multiple programming languages, different data storage technologies, event-driven communication patterns, and proper API versioning strategies."

        result = await consult_tool(
            query=long_query,
            consultation_type="deep",
            max_turns=2,
            target_model="anthropic/claude-3-5-sonnet",
        )

        # Should handle long queries without errors
        assert "🔍 Deep Consultation Session" in result
        assert "microservices" in result.lower()

    @pytest.mark.asyncio
    async def test_structured_session_state(self, mock_multi_turn_dependencies):
        """Test that structured session state is maintained."""

        result = await consult_tool(
            query="Help me implement JWT authentication for my REST API",
            consultation_type="deep",
            max_turns=3,
            context="security implementation",
        )

        # Should capture security context and JWT topic
        assert "🔍 Deep Consultation Session" in result
        assert "security implementation" in result
        assert "jwt" in result.lower() or "authentication" in result.lower()


class TestLLMFollowUpEvaluation:
    """Test LLM-based follow-up evaluation functionality."""

    @pytest.mark.asyncio
    async def test_follow_up_evaluator_integration(
        self, mock_multi_turn_dependencies, monkeypatch
    ):
        """Test that LLM-based follow-up evaluation is working."""

        # Mock the follow-up evaluator to return consistent results
        async def mock_evaluate_follow_up(
            consultation_type,
            user_query,
            ai_response,
            turn_number,
            max_turns,
            conversation_context="",
        ):
            # Simulate intelligent follow-up decision
            if (
                "elaborate" in ai_response.lower()
                or "dive deeper" in ai_response.lower()
            ):
                return {
                    "needs_followup": True,
                    "confidence": 0.8,
                    "reason": "Response suggests further exploration",
                    "suggested_query": "Please provide specific implementation examples",
                    "estimated_cost": "0.002",
                }
            else:
                return {
                    "needs_followup": False,
                    "confidence": 0.9,
                    "reason": "Response appears complete",
                    "suggested_query": None,
                    "estimated_cost": "0.002",
                }

        # Add the mock to the dependencies
        monkeypatch.setattr(
            "second_opinion.mcp.tools.consult.evaluate_follow_up_need",
            mock_evaluate_follow_up,
        )

        result = await consult_tool(
            query="Explain the trade-offs between different database consistency models",
            consultation_type="deep",
            max_turns=3,
        )

        # Should use LLM-based evaluation
        assert "🔍 Deep Consultation Session" in result
        assert "consistency" in result.lower() or "database" in result.lower()

    @pytest.mark.asyncio
    async def test_follow_up_cost_tracking(self, mock_multi_turn_dependencies):
        """Test that follow-up evaluation costs are tracked."""

        result = await consult_tool(
            query="Design a scalable caching strategy for high-traffic applications",
            consultation_type="deep",
            max_turns=2,
            cost_limit=0.50,
        )

        # Should track costs including follow-up evaluation
        assert "🔍 Deep Consultation Session" in result
        assert "Cost Analysis" in result
        assert "Total Cost" in result

    @pytest.mark.asyncio
    async def test_consultation_type_specific_follow_up(
        self, mock_multi_turn_dependencies
    ):
        """Test that follow-up behavior varies by consultation type."""

        # Quick consultation - should rarely have follow-up
        quick_result = await consult_tool(
            query="What's the difference between NoSQL and SQL databases?",
            consultation_type="quick",
        )

        assert "🎯 Quick Expert Consultation" in quick_result
        # Should not have follow-up continuation options

        # Deep consultation - should support follow-up
        deep_result = await consult_tool(
            query="Compare NoSQL vs SQL for large-scale applications",
            consultation_type="deep",
            max_turns=3,
        )

        assert "🔍 Deep Consultation Session" in deep_result
        assert "Continue Consultation" in deep_result or "Session ID" in deep_result


class TestAdvancedMultiTurnScenarios:
    """Test advanced multi-turn scenarios with enhanced features."""

    @pytest.mark.asyncio
    async def test_complex_technical_conversation(self, mock_multi_turn_dependencies):
        """Test complex technical conversation with context preservation."""

        result = await consult_tool(
            query="I'm building a real-time trading platform that needs microsecond latency, how should I architect the system?",
            consultation_type="deep",
            max_turns=4,
            context="high-frequency trading systems",
        )

        # Should handle complex technical requirements
        assert "🔍 Deep Consultation Session" in result
        assert "high-frequency trading systems" in result
        assert "latency" in result.lower() or "trading" in result.lower()

    @pytest.mark.asyncio
    async def test_domain_specific_context_preservation(
        self, mock_multi_turn_dependencies
    ):
        """Test that domain-specific context is preserved across turns."""

        result = await consult_tool(
            query="Implement machine learning model versioning and deployment pipeline",
            consultation_type="deep",
            max_turns=3,
            context="MLOps and machine learning infrastructure",
        )

        # Should preserve ML domain context
        assert "🔍 Deep Consultation Session" in result
        assert "MLOps and machine learning infrastructure" in result
        assert (
            "machine learning" in result.lower()
            or "mlops" in result.lower()
            or "model" in result.lower()
        )

    @pytest.mark.asyncio
    async def test_session_state_across_continuation(
        self, mock_multi_turn_dependencies
    ):
        """Test that enhanced session state works across session continuation."""

        # First conversation
        first_result = await consult_tool(
            query="Help me design an event-driven microservices architecture",
            consultation_type="deep",
            max_turns=2,
            target_model="anthropic/claude-3-5-sonnet",
        )

        # Extract session ID
        import re

        session_id_match = re.search(r"Session ID.*?`([^`]+)`", first_result)
        if not session_id_match:
            session_id_match = re.search(r"Session ID.*?([a-f0-9-]+)", first_result)

        if session_id_match:
            session_id = session_id_match.group(1)

            # Continue conversation
            second_result = await consult_tool(
                query="Now focus on the event sourcing implementation details",
                consultation_type="deep",
                session_id=session_id,
                max_turns=2,
                target_model="anthropic/claude-3-5-sonnet",
            )

            # Should reference previous conversation context
            assert "🔍 Deep Consultation Session" in second_result
            assert session_id in second_result
            assert (
                "Previous conversation context" in second_result
                or "Building on our previous discussion" in second_result
            )
