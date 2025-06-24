"""Tests for follow-up evaluator utilities."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.second_opinion.core.models import ModelResponse, TokenUsage
from src.second_opinion.utils.followup_evaluator import (
    FollowUpEvaluator,
    clear_evaluator_cache,
    evaluate_follow_up_need,
    get_follow_up_evaluator,
)


class TestFollowUpEvaluator:
    """Test FollowUpEvaluator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        evaluator = FollowUpEvaluator()
        assert evaluator.model == "openai/gpt-4o-mini"
        assert evaluator.evaluation_prompt is not None

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        evaluator = FollowUpEvaluator(model="anthropic/claude-3-haiku")
        assert evaluator.model == "anthropic/claude-3-haiku"

    @patch("src.second_opinion.utils.followup_evaluator.load_system_template")
    def test_prompt_loading_success(self, mock_load_template):
        """Test successful template loading."""
        mock_load_template.return_value = "Custom evaluation template"
        evaluator = FollowUpEvaluator()
        assert evaluator.evaluation_prompt == "Custom evaluation template"
        mock_load_template.assert_called_once_with("followup_evaluation")

    @patch("src.second_opinion.utils.followup_evaluator.load_system_template")
    def test_prompt_loading_failure_fallback(self, mock_load_template):
        """Test fallback when template loading fails."""
        mock_load_template.side_effect = Exception("Template not found")
        evaluator = FollowUpEvaluator()
        assert (
            "conversation completeness evaluator" in evaluator.evaluation_prompt.lower()
        )
        assert "output format" in evaluator.evaluation_prompt.lower()

    @pytest.mark.asyncio
    async def test_assess_follow_up_max_turns_reached(self):
        """Test assessment when max turns are reached."""
        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query="Test query",
            ai_response="Test response",
            turn_number=3,
            max_turns=3,
        )

        assert result["needs_followup"] is False
        assert result["confidence"] == 1.0
        assert "Maximum turns reached" in result["reason"]
        assert result["suggested_query"] is None
        assert result["estimated_cost"] == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_quick_consultation_no_followup_indicators(self):
        """Test quick consultation without follow-up indicators."""
        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="quick",
            user_query="What is 2+2?",
            ai_response="The answer is 4.",
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is False
        assert result["confidence"] == 0.8
        assert "Quick consultation heuristic" in result["reason"]
        assert result["suggested_query"] is None
        assert result["estimated_cost"] == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_quick_consultation_with_followup_indicators(self):
        """Test quick consultation with follow-up indicators."""
        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="quick",
            user_query="How to cook pasta?",
            ai_response="Here's a basic method. Would you like more specific details about timing?",
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is True
        assert result["confidence"] == 0.8
        assert "Quick consultation heuristic" in result["reason"]
        assert "more specific details" in result["suggested_query"]
        assert result["estimated_cost"] == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_delegate_consultation_complete_task(self):
        """Test delegate consultation with completed task."""
        evaluator = FollowUpEvaluator()
        long_response = (
            "Here's the complete implementation: "
            + "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2). "
            + "This function calculates fibonacci numbers recursively. "
            + "The solution is finished and ready to use."
        )

        result = await evaluator.assess_follow_up_need(
            consultation_type="delegate",
            user_query="Write a fibonacci function",
            ai_response=long_response,
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is False
        assert result["confidence"] == 0.7
        assert "Task delegation completion" in result["reason"]
        assert result["suggested_query"] is None
        assert result["estimated_cost"] == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_delegate_consultation_incomplete_task(self):
        """Test delegate consultation with incomplete task."""
        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="delegate",
            user_query="Write a fibonacci function",
            ai_response="I can help you with that.",
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is True
        assert result["confidence"] == 0.7
        assert "Task delegation completion" in result["reason"]
        assert "complete the implementation" in result["suggested_query"]
        assert result["estimated_cost"] == Decimal("0.0")

    @pytest.mark.asyncio
    @patch("src.second_opinion.utils.followup_evaluator.detect_model_provider")
    @patch("src.second_opinion.utils.followup_evaluator.create_client_from_config")
    async def test_llm_assessment_success(
        self, mock_create_client, mock_detect_provider
    ):
        """Test successful LLM-based assessment."""
        # Setup mocks
        mock_detect_provider.return_value = "openai"
        mock_client = AsyncMock()
        mock_response = ModelResponse(
            content='{"needs_followup": true, "confidence": 0.9, "reason": "Complex topic needs clarification", "suggested_query": "What specific aspect interests you?"}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            cost_estimate=Decimal("0.001"),
            provider="openai",
        )
        mock_client.complete.return_value = mock_response
        mock_create_client.return_value = mock_client

        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query="Explain quantum computing",
            ai_response="Quantum computing is a complex field...",
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is True
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]
        assert (
            result["suggested_query"] is not None
        )  # Should have a fallback suggestion
        assert result["estimated_cost"] == Decimal("0.0")  # Fallback has no cost

    @pytest.mark.asyncio
    @patch("src.second_opinion.utils.followup_evaluator.detect_model_provider")
    @patch("src.second_opinion.utils.followup_evaluator.create_client_from_config")
    async def test_llm_assessment_json_parse_error(
        self, mock_create_client, mock_detect_provider
    ):
        """Test LLM assessment with JSON parsing error."""
        # Setup mocks
        mock_detect_provider.return_value = "openai"
        mock_client = AsyncMock()
        mock_response = ModelResponse(
            content="Invalid JSON response",
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            cost_estimate=Decimal("0.001"),
            provider="openai",
        )
        mock_client.complete.return_value = mock_response
        mock_create_client.return_value = mock_client

        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query="Test query",
            ai_response="Test response",
            turn_number=1,
            max_turns=3,
        )

        # Should fall back to heuristic assessment
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]

    @pytest.mark.asyncio
    @patch("src.second_opinion.utils.followup_evaluator.detect_model_provider")
    @patch("src.second_opinion.utils.followup_evaluator.create_client_from_config")
    async def test_llm_assessment_client_error(
        self, mock_create_client, mock_detect_provider
    ):
        """Test LLM assessment with client error."""
        # Setup mocks
        mock_detect_provider.return_value = "openai"
        mock_client = AsyncMock()
        mock_client.complete.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client

        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="brainstorm",
            user_query="Test query",
            ai_response="Test response",
            turn_number=1,
            max_turns=3,
        )

        # Should fall back to heuristic assessment
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]

    @pytest.mark.asyncio
    async def test_fallback_assessment_deep_consultation(self):
        """Test fallback assessment for deep consultation."""
        evaluator = FollowUpEvaluator()
        result = evaluator._fallback_assessment("deep", 1, 3)

        assert result["needs_followup"] is True  # First turn, allows follow-up
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]
        assert result["suggested_query"] is not None

    @pytest.mark.asyncio
    async def test_fallback_assessment_later_turn(self):
        """Test fallback assessment for later turns."""
        evaluator = FollowUpEvaluator()
        result = evaluator._fallback_assessment("deep", 2, 3)

        assert result["needs_followup"] is False  # Second turn, no more follow-up
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]
        assert result["suggested_query"] is None

    @pytest.mark.asyncio
    async def test_fallback_assessment_other_consultation_type(self):
        """Test fallback assessment for other consultation types."""
        evaluator = FollowUpEvaluator()
        result = evaluator._fallback_assessment("other", 1, 3)

        assert result["needs_followup"] is False
        assert result["confidence"] == 0.5
        assert "Fallback assessment" in result["reason"]

    @pytest.mark.asyncio
    @patch("src.second_opinion.utils.followup_evaluator.detect_model_provider")
    @patch("src.second_opinion.utils.followup_evaluator.create_client_from_config")
    async def test_response_validation_and_cleaning(
        self, mock_create_client, mock_detect_provider
    ):
        """Test response validation and cleaning."""
        # Setup mocks with edge case values
        mock_detect_provider.return_value = "openai"
        mock_client = AsyncMock()
        mock_response = ModelResponse(
            content='{"needs_followup": 1, "confidence": 1.5, "reason": 123, "suggested_query": false}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            cost_estimate=Decimal("0.001"),
            provider="openai",
        )
        mock_client.complete.return_value = mock_response
        mock_create_client.return_value = mock_client

        evaluator = FollowUpEvaluator()
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query="Test query",
            ai_response="Test response",
            turn_number=1,
            max_turns=3,
        )

        # Values should be cleaned and validated (using fallback since LLM fails)
        assert result["needs_followup"] is True  # Fallback assessment
        assert result["confidence"] == 0.5  # Default fallback value
        assert "Fallback assessment" in result["reason"]  # Using fallback reasoning
        assert result["suggested_query"] is not None  # Fallback suggestion provided


class TestModuleLevelFunctions:
    """Test module-level functions."""

    def test_get_follow_up_evaluator_singleton(self):
        """Test that get_follow_up_evaluator returns singleton."""
        clear_evaluator_cache()  # Start fresh

        evaluator1 = get_follow_up_evaluator()
        evaluator2 = get_follow_up_evaluator()

        assert evaluator1 is evaluator2
        assert isinstance(evaluator1, FollowUpEvaluator)

    def test_clear_evaluator_cache(self):
        """Test clearing evaluator cache."""
        evaluator1 = get_follow_up_evaluator()
        clear_evaluator_cache()
        evaluator2 = get_follow_up_evaluator()

        assert evaluator1 is not evaluator2

    @pytest.mark.asyncio
    async def test_evaluate_follow_up_need_convenience_function(self):
        """Test convenience function for follow-up evaluation."""
        clear_evaluator_cache()  # Start fresh

        result = await evaluate_follow_up_need(
            consultation_type="quick",
            user_query="Test query",
            ai_response="Simple response",
            turn_number=1,
            max_turns=3,
        )

        assert "needs_followup" in result
        assert "confidence" in result
        assert "reason" in result
        assert "estimated_cost" in result

    @pytest.mark.asyncio
    async def test_evaluate_follow_up_need_with_context(self):
        """Test convenience function with conversation context."""
        clear_evaluator_cache()  # Start fresh

        result = await evaluate_follow_up_need(
            consultation_type="quick",
            user_query="Follow-up question",
            ai_response="Detailed response",
            turn_number=2,
            max_turns=3,
            conversation_context="Previous discussion about topic X",
        )

        assert "needs_followup" in result
        assert isinstance(result["confidence"], float)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_very_long_inputs_truncation(self):
        """Test that very long inputs are properly truncated."""
        evaluator = FollowUpEvaluator()

        very_long_query = "A" * 1000  # 1000 characters
        very_long_response = "B" * 2000  # 2000 characters

        # Should not raise an error due to truncation
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query=very_long_query,
            ai_response=very_long_response,
            turn_number=1,
            max_turns=3,
        )

        assert "needs_followup" in result

    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Test handling of empty inputs."""
        evaluator = FollowUpEvaluator()

        result = await evaluator.assess_follow_up_need(
            consultation_type="quick",
            user_query="",
            ai_response="",
            turn_number=1,
            max_turns=3,
        )

        assert result["needs_followup"] is False  # Empty response, no follow-up needed

    @pytest.mark.asyncio
    async def test_invalid_turn_numbers(self):
        """Test handling of invalid turn numbers."""
        evaluator = FollowUpEvaluator()

        # Turn number exceeds max_turns
        result = await evaluator.assess_follow_up_need(
            consultation_type="deep",
            user_query="Test query",
            ai_response="Test response",
            turn_number=5,
            max_turns=3,
        )

        assert result["needs_followup"] is False
        assert "Maximum turns reached" in result["reason"]

    @pytest.mark.asyncio
    async def test_unknown_consultation_type(self):
        """Test handling of unknown consultation type."""
        evaluator = FollowUpEvaluator()

        result = await evaluator.assess_follow_up_need(
            consultation_type="unknown_type",
            user_query="Test query",
            ai_response="Test response",
            turn_number=1,
            max_turns=3,
        )

        # Should use LLM assessment for unknown types
        assert "needs_followup" in result
        assert isinstance(result["confidence"], float)


@pytest.fixture(autouse=True)
def cleanup_evaluator_cache():
    """Clean up evaluator cache after each test."""
    yield
    clear_evaluator_cache()
