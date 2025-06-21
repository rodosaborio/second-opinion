"""
Tests for the consult MCP tool.

This module tests the AI consultation functionality including session management,
model routing, multi-turn conversations, and cost optimization.
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from second_opinion.core.models import ModelResponse, TokenUsage, TaskComplexity
from second_opinion.mcp.tools.consult import (
    consult_tool,
    ConsultationSession,
    ConsultationModelRouter,
    TurnController,
    create_consultation_session,
    get_consultation_session,
    calculate_delegation_savings,
    _consultation_sessions,
)


@pytest.fixture(autouse=True)
def clear_consultation_sessions():
    """Clear consultation sessions between tests."""
    _consultation_sessions.clear()
    yield
    _consultation_sessions.clear()


@pytest.fixture
def mock_consultation_dependencies(monkeypatch):
    """Mock all external dependencies for consultation tests."""

    # Mock cost guard
    mock_cost_guard = MagicMock()
    mock_cost_guard.check_and_reserve_budget = AsyncMock(
        return_value=MagicMock(reservation_id="test-reservation", approved=True)
    )
    mock_cost_guard.record_actual_cost = AsyncMock()
    mock_cost_guard.release_reservation = AsyncMock()

    def get_mock_cost_guard():
        return mock_cost_guard

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.get_cost_guard", get_mock_cost_guard
    )

    # Mock evaluator
    mock_evaluator = MagicMock()
    mock_evaluator.classify_task_complexity = AsyncMock(
        return_value=TaskComplexity.MODERATE
    )

    def get_mock_evaluator():
        return mock_evaluator

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.get_evaluator", get_mock_evaluator
    )

    # Mock client factory
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=ModelResponse(
            content="Mock consultation response",
            model="test-model",
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            cost_estimate=Decimal("0.01"),
            provider="mock",
        )
    )
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
        """Mock domain classification based on keywords for predictable testing."""
        query_lower = query.lower()
        if any(
            term in query_lower
            for term in ["python", "function", "code", "debug", "programming"]
        ):
            return "coding"
        elif any(
            term in query_lower
            for term in ["performance", "optimize", "scalable", "infrastructure"]
        ):
            return "performance"
        elif any(
            term in query_lower
            for term in ["creative", "story", "marketing", "content"]
        ):
            return "creative"
        else:
            return "general"

    monkeypatch.setattr(
        "second_opinion.mcp.tools.consult.classify_consultation_domain",
        mock_classify_domain,
    )

    return {
        "cost_guard": mock_cost_guard,
        "evaluator": mock_evaluator,
        "client": mock_client,
    }


class TestConsultationSession:
    """Test ConsultationSession functionality."""

    def test_session_creation(self):
        """Test creating a new consultation session."""
        session = ConsultationSession("quick", "anthropic/claude-3-5-sonnet")

        assert session.consultation_type == "quick"
        assert session.target_model == "anthropic/claude-3-5-sonnet"
        assert session.turn_count == 0
        assert session.status == "active"
        assert len(session.messages) == 0
        assert session.total_cost == Decimal("0.0")

    async def test_add_turn(self):
        """Test adding conversation turns."""
        session = ConsultationSession("deep", "anthropic/claude-3-opus")

        await session.add_turn(
            "What is async programming?", "Async programming allows...", Decimal("0.05")
        )

        assert session.turn_count == 1
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "What is async programming?"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "Async programming allows..."
        assert session.total_cost == Decimal("0.05")

    def test_can_continue(self):
        """Test session continuation logic."""
        session = ConsultationSession("deep", "anthropic/claude-3-opus")

        # Should be able to continue initially
        assert session.can_continue(3, Decimal("0.50"))

        # Should not continue if max turns reached
        session.turn_count = 3
        assert not session.can_continue(3, Decimal("0.50"))

        # Should not continue if cost limit exceeded
        session.turn_count = 1
        session.total_cost = Decimal("0.60")
        assert not session.can_continue(3, Decimal("0.50"))

        # Should not continue if status is not active
        session.total_cost = Decimal("0.10")
        session.status = "completed"
        assert not session.can_continue(3, Decimal("0.50"))

    async def test_conversation_context(self):
        """Test conversation context generation."""
        session = ConsultationSession("deep", "anthropic/claude-3-opus")

        # No context initially
        assert session.get_conversation_context() == ""

        # Add turns and check context
        await session.add_turn("First question", "First answer", Decimal("0.01"))
        await session.add_turn("Second question", "Second answer", Decimal("0.01"))

        context = session.get_conversation_context()
        assert "Previous Q: First question" in context
        assert "Previous A: First answer" in context
        assert "Previous Q: Second question" in context
        assert "Previous A: Second answer" in context


class TestConsultationModelRouter:
    """Test ConsultationModelRouter functionality."""

    def test_model_recommendation_quick(self):
        """Test model recommendation for quick consultation."""
        router = ConsultationModelRouter()

        model = router.recommend_model("quick")
        assert model == "anthropic/claude-3-5-sonnet"

    def test_model_recommendation_delegate(self):
        """Test model recommendation for task delegation."""
        router = ConsultationModelRouter()

        # Simple task should use cheap model
        model = router.recommend_model(
            "delegate", task_complexity=TaskComplexity.SIMPLE
        )
        assert model == "openai/gpt-4o-mini"

        # Complex task should use better model
        model = router.recommend_model(
            "delegate", task_complexity=TaskComplexity.COMPLEX
        )
        assert model == "anthropic/claude-3-5-sonnet"

    def test_model_recommendation_expert(self):
        """Test model recommendation for expert consultation."""
        router = ConsultationModelRouter()

        model = router.recommend_model("expert")
        assert model == "anthropic/claude-3-opus"

    def test_model_recommendation_brainstorm(self):
        """Test model recommendation for brainstorming."""
        router = ConsultationModelRouter()

        model = router.recommend_model("brainstorm")
        assert model == "openai/gpt-4o"

    def test_user_specified_model_override(self):
        """Test that user-specified model overrides recommendations."""
        router = ConsultationModelRouter()

        model = router.recommend_model(
            "delegate", user_specified_model="anthropic/claude-3-opus"
        )
        assert model == "anthropic/claude-3-opus"

    @pytest.mark.asyncio
    async def test_domain_detection_coding(self, mock_consultation_dependencies):
        """Test domain detection for coding queries."""
        router = ConsultationModelRouter()

        domain = await router.detect_domain_specialization(
            "Write a Python function to sort a list"
        )
        assert domain == "coding"

        domain = await router.detect_domain_specialization(
            "Help me debug this JavaScript code"
        )
        assert domain == "coding"

    @pytest.mark.asyncio
    async def test_domain_detection_performance(self, mock_consultation_dependencies):
        """Test domain detection for performance queries."""
        router = ConsultationModelRouter()

        domain = await router.detect_domain_specialization(
            "How can I optimize database query performance?"
        )
        assert domain == "performance"

        domain = await router.detect_domain_specialization(
            "Design a scalable infrastructure"
        )
        assert domain == "performance"

    @pytest.mark.asyncio
    async def test_domain_detection_creative(self, mock_consultation_dependencies):
        """Test domain detection for creative queries."""
        router = ConsultationModelRouter()

        domain = await router.detect_domain_specialization(
            "Write a creative story about AI"
        )
        assert domain == "creative"

        domain = await router.detect_domain_specialization(
            "Help me with marketing content"
        )
        assert domain == "creative"

    @pytest.mark.asyncio
    async def test_domain_detection_general(self, mock_consultation_dependencies):
        """Test domain detection for general queries."""
        router = ConsultationModelRouter()

        domain = await router.detect_domain_specialization(
            "What is the capital of France?"
        )
        assert domain == "general"


class TestTurnController:
    """Test TurnController functionality."""

    @pytest.fixture
    def turn_controller(self):
        """Create a TurnController instance."""
        return TurnController()

    async def test_single_turn_consultation(
        self, turn_controller, mock_consultation_dependencies
    ):
        """Test single-turn consultation flow."""
        session = ConsultationSession("quick", "anthropic/claude-3-5-sonnet")

        results = await turn_controller.conduct_consultation(
            session=session,
            initial_query="What is machine learning?",
            max_turns=1,
            cost_limit=Decimal("0.25"),
        )

        assert len(results["conversation_turns"]) == 1
        assert results["conversation_turns"][0]["turn"] == 1
        assert results["conversation_turns"][0]["query"] == "What is machine learning?"
        assert (
            results["conversation_turns"][0]["response"] == "Mock consultation response"
        )
        assert session.turn_count == 1
        assert session.status == "completed"

    async def test_multi_turn_consultation(
        self, turn_controller, mock_consultation_dependencies
    ):
        """Test multi-turn consultation flow."""
        session = ConsultationSession("deep", "anthropic/claude-3-opus")

        # Mock follow-up assessment to continue conversation
        original_assess = turn_controller._assess_follow_up_need

        async def mock_assess_follow_up(session, response, turn, max_turns):
            if turn == 0:  # Continue after first turn
                return {"needed": True, "query": "Please elaborate further"}
            return {"needed": False, "query": None}

        turn_controller._assess_follow_up_need = mock_assess_follow_up

        results = await turn_controller.conduct_consultation(
            session=session,
            initial_query="Explain async programming",
            max_turns=3,
            cost_limit=Decimal("0.50"),
        )

        assert len(results["conversation_turns"]) == 2
        assert session.turn_count == 2
        assert session.status == "completed"

        # Restore original method
        turn_controller._assess_follow_up_need = original_assess

    async def test_cost_limit_stops_conversation(
        self, turn_controller, mock_consultation_dependencies
    ):
        """Test that cost limit stops conversation."""
        session = ConsultationSession("deep", "anthropic/claude-3-opus")
        session.total_cost = Decimal("0.40")  # Already near limit

        results = await turn_controller.conduct_consultation(
            session=session,
            initial_query="Complex question",
            max_turns=5,
            cost_limit=Decimal("0.45"),  # Low limit
        )

        # Should complete only 1 turn due to cost limit
        assert len(results["conversation_turns"]) == 1

    def test_system_prompt_generation(self, turn_controller):
        """Test system prompt generation for different consultation types."""

        # Quick consultation
        session = ConsultationSession("quick", "test-model")
        prompt = turn_controller._get_system_prompt(session)
        assert "quick consultation" in prompt.lower()
        assert "concise" in prompt.lower()

        # Delegate consultation
        session = ConsultationSession("delegate", "test-model")
        prompt = turn_controller._get_system_prompt(session)
        assert "delegate consultation" in prompt.lower()
        assert "complete the requested task" in prompt.lower()

        # Deep consultation
        session = ConsultationSession("deep", "test-model")
        prompt = turn_controller._get_system_prompt(session)
        assert "deep consultation" in prompt.lower()
        assert "comprehensive" in prompt.lower()

        # Brainstorm consultation
        session = ConsultationSession("brainstorm", "test-model")
        prompt = turn_controller._get_system_prompt(session)
        assert "brainstorm consultation" in prompt.lower()
        assert "creative" in prompt.lower()


class TestConsultTool:
    """Test the main consult_tool function."""

    @pytest.mark.asyncio
    async def test_quick_consultation(self, mock_consultation_dependencies):
        """Test quick expert consultation."""
        result = await consult_tool(
            query="Should I use async/await or threading?",
            consultation_type="quick",
            target_model="anthropic/claude-3-5-sonnet",
        )

        assert "🎯 Quick Expert Consultation" in result
        assert "Mock consultation response" in result
        assert "anthropic/claude-3-5-sonnet" in result
        assert "$0.01" in result  # Cost should be displayed

    @pytest.mark.asyncio
    async def test_delegate_consultation(self, mock_consultation_dependencies):
        """Test task delegation consultation."""
        result = await consult_tool(
            query="Write unit tests for this function",
            consultation_type="delegate",
            target_model="openai/gpt-4o-mini",
        )

        assert "📋 Task Delegation Results" in result
        assert "Mock consultation response" in result
        assert "openai/gpt-4o-mini" in result
        assert "Estimated Savings" in result  # Should show cost savings

    @pytest.mark.asyncio
    async def test_deep_consultation(self, mock_consultation_dependencies):
        """Test deep consultation with multi-turn capability."""
        result = await consult_tool(
            query="Help me design a scalable system",
            consultation_type="deep",
            max_turns=2,
        )

        assert "🔍 Deep Consultation Session" in result
        assert "Mock consultation response" in result
        assert "Session ID" in result  # Should provide session ID for continuation

    @pytest.mark.asyncio
    async def test_brainstorm_consultation(self, mock_consultation_dependencies):
        """Test brainstorming consultation."""
        result = await consult_tool(
            query="Explore approaches to this problem",
            consultation_type="brainstorm",
            context="creative problem solving",
        )

        assert "💡 Brainstorming Session" in result
        assert "Mock consultation response" in result
        assert "creative problem solving" in result

    @pytest.mark.asyncio
    async def test_invalid_consultation_type(self, mock_consultation_dependencies):
        """Test error handling for invalid consultation type."""
        result = await consult_tool(
            query="Test query", consultation_type="invalid_type"
        )

        assert "❌ **Invalid consultation type**" in result
        assert "invalid_type" in result
        assert "Valid types" in result

    @pytest.mark.asyncio
    async def test_invalid_max_turns(self, mock_consultation_dependencies):
        """Test error handling for invalid max_turns."""
        result = await consult_tool(query="Test query", max_turns=10)  # Too high

        assert "❌ **Invalid max_turns**" in result
        assert "between 1 and 5" in result

    @pytest.mark.asyncio
    async def test_invalid_target_model(self, mock_consultation_dependencies):
        """Test error handling for invalid target model."""
        result = await consult_tool(query="Test query", target_model="invalid@model")

        assert "❌ **Invalid target model**" in result
        assert "Suggested formats" in result

    @pytest.mark.asyncio
    async def test_session_continuation(self, mock_consultation_dependencies):
        """Test continuing an existing consultation session."""
        # Create initial session
        session = create_consultation_session("deep", "anthropic/claude-3-opus")
        session_id = session.session_id

        # Continue the session
        result = await consult_tool(
            query="Follow-up question", consultation_type="deep", session_id=session_id
        )

        assert session_id in result
        assert "Mock consultation response" in result

    @pytest.mark.asyncio
    async def test_session_not_found(self, mock_consultation_dependencies):
        """Test error handling for non-existent session."""
        result = await consult_tool(
            query="Test query", session_id="non-existent-session"
        )

        assert "❌ **Session not found**" in result
        assert "non-existent-session" in result

    @pytest.mark.asyncio
    async def test_cost_limit_override(self, mock_consultation_dependencies):
        """Test cost limit override functionality."""
        result = await consult_tool(
            query="Test query",
            consultation_type="quick",
            cost_limit=0.05,  # Custom cost limit
        )

        assert "Mock consultation response" in result
        # Cost guard should have been called with the override
        mock_cost_guard = mock_consultation_dependencies["cost_guard"]
        mock_cost_guard.check_and_reserve_budget.assert_called()
        call_args = mock_cost_guard.check_and_reserve_budget.call_args
        assert call_args[1]["per_request_override"] == Decimal("0.05")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_and_get_consultation_session(self):
        """Test session creation and retrieval."""
        session = create_consultation_session("quick", "test-model")

        assert session.consultation_type == "quick"
        assert session.target_model == "test-model"

        # Should be able to retrieve the session
        retrieved = get_consultation_session(session.session_id)
        assert retrieved is session

        # Non-existent session should return None
        assert get_consultation_session("non-existent") is None

    def test_calculate_delegation_savings(self):
        """Test delegation savings calculation."""
        # Mini model should have high savings
        savings = calculate_delegation_savings("openai/gpt-4o-mini")
        assert savings == Decimal("0.08")

        # Haiku model should have high savings
        savings = calculate_delegation_savings("anthropic/claude-3-haiku")
        assert savings == Decimal("0.08")

        # Sonnet should have moderate savings
        savings = calculate_delegation_savings("anthropic/claude-3-5-sonnet")
        assert savings == Decimal("0.03")

        # Unknown model should have minimal savings
        savings = calculate_delegation_savings("unknown/model")
        assert savings == Decimal("0.01")


@pytest.mark.asyncio
async def test_consultation_integration_flow(mock_consultation_dependencies):
    """
    Test complete consultation flow integration.

    This test verifies the entire consultation process from start to finish.
    """
    # Test quick consultation flow
    result = await consult_tool(
        query="What's the best way to handle errors in Python?",
        consultation_type="quick",
        context="programming best practices",
    )

    # Verify response structure
    assert "🎯 Quick Expert Consultation" in result
    assert "Consultation Summary" in result
    assert "Cost Analysis" in result
    assert "Next Steps" in result
    assert "programming best practices" in result

    # Verify cost tracking was called
    mock_cost_guard = mock_consultation_dependencies["cost_guard"]
    assert mock_cost_guard.check_and_reserve_budget.called
    assert mock_cost_guard.record_actual_cost.called

    # Verify model was called
    mock_client = mock_consultation_dependencies["client"]
    assert mock_client.complete.called
    assert mock_client.estimate_cost.called


@pytest.mark.asyncio
async def test_error_handling_robustness(mock_consultation_dependencies):
    """
    Test error handling robustness across different failure scenarios.
    """

    # Test client failure
    mock_client = mock_consultation_dependencies["client"]
    mock_client.complete.side_effect = Exception("API Error")

    result = await consult_tool(
        query="Test query with API failure", consultation_type="quick"
    )

    assert "❌ **Consultation Error**" in result
    assert "API Error" in result

    # Verify cost reservation was released
    mock_cost_guard = mock_consultation_dependencies["cost_guard"]
    assert mock_cost_guard.release_reservation.called


@pytest.mark.asyncio
async def test_consultation_type_specific_behavior(mock_consultation_dependencies):
    """
    Test that different consultation types produce appropriate behavior.
    """

    # Test delegate type uses cost-effective model
    result = await consult_tool(
        query="Simple task delegation", consultation_type="delegate"
    )

    assert "📋 Task Delegation Results" in result
    assert "Estimated Savings" in result

    # Test deep type allows multi-turn
    result = await consult_tool(
        query="Complex analysis needed", consultation_type="deep", max_turns=3
    )

    assert "🔍 Deep Consultation Session" in result
    assert "Session ID" in result  # Should provide continuation option

    # Test brainstorm type focuses on creativity
    result = await consult_tool(
        query="Need creative solutions", consultation_type="brainstorm"
    )

    assert "💡 Brainstorming Session" in result
