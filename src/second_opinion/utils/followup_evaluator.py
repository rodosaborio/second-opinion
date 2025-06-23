"""
Follow-up evaluator using LLM-based intelligent assessment.

This module provides smart follow-up assessment for multi-turn consultations,
replacing simple heuristic patterns with LLM-based evaluation of conversation
completeness and follow-up needs.
"""

import json
import logging
from decimal import Decimal
from typing import Any

from ..clients import detect_model_provider
from ..core.models import Message, ModelRequest
from ..utils.client_factory import create_client_from_config

logger = logging.getLogger(__name__)


class FollowUpEvaluator:
    """Intelligent follow-up assessment using small language models."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        """
        Initialize follow-up evaluator.

        Args:
            model: Evaluation model to use (default: cost-effective gpt-4o-mini)
        """
        self.model = model
        self.evaluation_prompt = self._get_evaluation_prompt()

    def _get_evaluation_prompt(self) -> str:
        """Get multi-shot evaluation prompt for follow-up assessment."""
        return """You are a conversation completeness evaluator for AI consultations. Assess whether a conversation needs follow-up based on the user's satisfaction and response completeness.

EVALUATION CRITERIA:

1. **Response Completeness**: Is the AI response comprehensive and actionable?
2. **User Satisfaction Indicators**: Does the response likely satisfy the user's need?
3. **Topic Complexity**: Is there remaining complexity that warrants further exploration?
4. **Consultation Type Alignment**: Does follow-up match the consultation type?

CONSULTATION TYPES:
- **quick**: Single-turn expert opinion (rarely needs follow-up)
- **delegate**: Task completion (follow-up only if incomplete)
- **deep**: Multi-turn analysis (often benefits from follow-up)
- **brainstorm**: Creative exploration (often benefits from follow-up)

FOLLOW-UP INDICATORS:

**Needs Follow-up**:
- Response asks clarifying questions
- Complex topic only partially addressed
- Multiple implementation approaches mentioned without details
- User's original goal partially met
- Response suggests "next steps" or "further discussion"
- Deep/brainstorm consultation with room for exploration

**No Follow-up Needed**:
- Complete, actionable response provided
- User's question fully answered
- Simple task successfully delegated
- Quick consultation with sufficient detail
- Response provides clear conclusion
- Maximum turns already reached

EXAMPLES:

**Example 1**:
Consultation Type: deep
User Query: "Help me design a scalable authentication system"
AI Response: "Here's a comprehensive authentication architecture: 1) JWT tokens for stateless auth, 2) OAuth2 for third-party integration, 3) Redis for session management, 4) bcrypt for password hashing. Implementation details: [500 words of specifics]. This approach handles 100k+ users with proper security."
Assessment: {"needs_followup": false, "confidence": 0.9, "reason": "Comprehensive response with implementation details", "suggested_query": null}

**Example 2**:
Consultation Type: deep
User Query: "What are the best practices for database optimization?"
AI Response: "Database optimization involves several key areas: indexing, query optimization, and hardware considerations. Would you like me to elaborate on any specific aspect? I can dive deeper into indexing strategies or query performance tuning."
Assessment: {"needs_followup": true, "confidence": 0.8, "reason": "Response asks for elaboration and offers specific directions", "suggested_query": "Please elaborate on indexing strategies and provide specific examples"}

**Example 3**:
Consultation Type: quick
User Query: "What's the difference between REST and GraphQL?"
AI Response: "REST uses multiple endpoints and HTTP methods, GraphQL uses a single endpoint with flexible queries. REST is simpler to implement, GraphQL reduces over-fetching. Choose REST for simple APIs, GraphQL for complex data requirements."
Assessment: {"needs_followup": false, "confidence": 0.9, "reason": "Clear comparison with decision guidance provided", "suggested_query": null}

**Example 4**:
Consultation Type: brainstorm
User Query: "Creative approaches to reduce API latency"
AI Response: "Here are 5 creative approaches: 1) Predictive prefetching using ML, 2) Edge computing with CDNs, 3) GraphQL query optimization, 4) Async processing patterns, 5) Smart caching strategies. Each has different trade-offs."
Assessment: {"needs_followup": true, "confidence": 0.7, "reason": "Multiple approaches mentioned but implementation details could be explored", "suggested_query": "Can you explore the trade-offs of predictive prefetching vs edge computing approaches?"}

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{
  "needs_followup": boolean,
  "confidence": float (0.0-1.0),
  "reason": "Brief explanation for the decision",
  "suggested_query": "Intelligent follow-up question" or null
}

CONVERSATION TO EVALUATE:

Consultation Type: {consultation_type}
Turn: {turn_number}/{max_turns}
User Query: {user_query}
AI Response: {ai_response}

Assessment:"""

    async def assess_follow_up_need(
        self,
        consultation_type: str,
        user_query: str,
        ai_response: str,
        turn_number: int,
        max_turns: int,
        conversation_context: str = "",
    ) -> dict[str, Any]:
        """
        Assess if follow-up is needed using LLM evaluation.

        Args:
            consultation_type: Type of consultation (quick, deep, delegate, brainstorm)
            user_query: User's query for this turn
            ai_response: AI's response for this turn
            turn_number: Current turn number (1-indexed)
            max_turns: Maximum allowed turns
            conversation_context: Optional previous conversation context

        Returns:
            Dictionary with follow-up assessment and suggested query
        """
        try:
            # Skip follow-up evaluation if we've reached max turns
            if turn_number >= max_turns:
                return {
                    "needs_followup": False,
                    "confidence": 1.0,
                    "reason": "Maximum turns reached",
                    "suggested_query": None,
                    "estimated_cost": Decimal("0.0"),
                }

            # Quick consultations rarely need follow-up
            if consultation_type == "quick":
                return await self._quick_heuristic_assessment(ai_response)

            # Delegate consultations need follow-up only if incomplete
            if consultation_type == "delegate":
                return await self._delegate_assessment(user_query, ai_response)

            # For deep/brainstorm, use full LLM evaluation
            return await self._llm_assessment(
                consultation_type, user_query, ai_response, turn_number, max_turns
            )

        except Exception as e:
            logger.warning(f"Follow-up evaluation failed, using fallback: {e}")
            return self._fallback_assessment(consultation_type, turn_number, max_turns)

    async def _llm_assessment(
        self,
        consultation_type: str,
        user_query: str,
        ai_response: str,
        turn_number: int,
        max_turns: int,
    ) -> dict[str, Any]:
        """Perform full LLM-based follow-up assessment."""
        # Build evaluation input
        evaluation_input = self.evaluation_prompt.format(
            consultation_type=consultation_type,
            user_query=user_query[:500],  # Truncate for cost efficiency
            ai_response=ai_response[:800],  # Truncate for cost efficiency
            turn_number=turn_number,
            max_turns=max_turns,
        )

        # Get provider and client
        provider = detect_model_provider(self.model)
        client = create_client_from_config(provider)

        # Create model request
        request = ModelRequest(
            model=self.model,
            messages=[Message(role="user", content=evaluation_input)],
            max_tokens=150,  # Small response needed
            temperature=0.1,  # Low temperature for consistency
            system_prompt="You are a precise conversation evaluator. Always respond with valid JSON.",
        )

        # Get evaluation
        response = await client.complete(request)

        # Parse JSON response
        try:
            result = json.loads(response.content.strip())

            # Validate and clean result
            needs_followup = bool(result.get("needs_followup", False))
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            reason = str(result.get("reason", "LLM evaluation completed"))
            suggested_query = result.get("suggested_query")

            if suggested_query and not isinstance(suggested_query, str):
                suggested_query = None

            return {
                "needs_followup": needs_followup,
                "confidence": confidence,
                "reason": reason,
                "suggested_query": suggested_query,
                "estimated_cost": response.cost_estimate,
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse follow-up evaluation response: {e}")
            return self._fallback_assessment(consultation_type, turn_number, max_turns)

    async def _quick_heuristic_assessment(self, ai_response: str) -> dict[str, Any]:
        """Quick heuristic assessment for 'quick' consultation type."""
        response_lower = ai_response.lower()

        # Quick consultations need follow-up only if response explicitly asks
        follow_up_indicators = [
            "would you like",
            "need more details",
            "want me to elaborate",
            "any specific",
            "further questions",
        ]

        needs_followup = any(
            indicator in response_lower for indicator in follow_up_indicators
        )

        return {
            "needs_followup": needs_followup,
            "confidence": 0.8,
            "reason": "Quick consultation heuristic assessment",
            "suggested_query": "Could you provide more specific details?"
            if needs_followup
            else None,
            "estimated_cost": Decimal("0.0"),
        }

    async def _delegate_assessment(
        self, user_query: str, ai_response: str
    ) -> dict[str, Any]:
        """Assessment for 'delegate' consultation type."""
        # Delegate consultations need follow-up if task appears incomplete
        completion_indicators = [
            "here's the",
            "implementation:",
            "complete",
            "finished",
            "solution:",
            "result:",
        ]

        response_lower = ai_response.lower()
        appears_complete = any(
            indicator in response_lower for indicator in completion_indicators
        )

        # Also check if response is substantial (likely a completed task)
        is_substantial = len(ai_response) > 200

        needs_followup = not (appears_complete and is_substantial)

        return {
            "needs_followup": needs_followup,
            "confidence": 0.7,
            "reason": "Task delegation completion assessment",
            "suggested_query": "Could you complete the implementation?"
            if needs_followup
            else None,
            "estimated_cost": Decimal("0.0"),
        }

    def _fallback_assessment(
        self, consultation_type: str, turn_number: int, max_turns: int
    ) -> dict[str, Any]:
        """Fallback assessment when LLM evaluation fails."""
        # Conservative fallback based on consultation type and turn number
        if consultation_type in ["deep", "brainstorm"] and turn_number < max_turns:
            needs_followup = turn_number < 2  # Allow 1 follow-up for deep/brainstorm
        else:
            needs_followup = False

        return {
            "needs_followup": needs_followup,
            "confidence": 0.5,
            "reason": "Fallback assessment (LLM evaluation failed)",
            "suggested_query": "Could you provide more details?"
            if needs_followup
            else None,
            "estimated_cost": Decimal("0.0"),
        }


# Global evaluator instance
_follow_up_evaluator: FollowUpEvaluator | None = None


def get_follow_up_evaluator() -> FollowUpEvaluator:
    """Get global follow-up evaluator instance."""
    global _follow_up_evaluator
    if _follow_up_evaluator is None:
        _follow_up_evaluator = FollowUpEvaluator()

    # Explicit type narrowing for ty
    evaluator = _follow_up_evaluator
    if evaluator is None:
        raise RuntimeError("Failed to initialize follow-up evaluator")
    return evaluator


async def evaluate_follow_up_need(
    consultation_type: str,
    user_query: str,
    ai_response: str,
    turn_number: int,
    max_turns: int,
    conversation_context: str = "",
) -> dict[str, Any]:
    """
    Convenience function for follow-up evaluation.

    Args:
        consultation_type: Type of consultation
        user_query: User's query
        ai_response: AI's response
        turn_number: Current turn number
        max_turns: Maximum turns allowed
        conversation_context: Optional conversation context

    Returns:
        Follow-up assessment dictionary
    """
    evaluator = get_follow_up_evaluator()
    return await evaluator.assess_follow_up_need(
        consultation_type,
        user_query,
        ai_response,
        turn_number,
        max_turns,
        conversation_context,
    )


def clear_evaluator_cache() -> None:
    """Clear the follow-up evaluator instance (useful for testing)."""
    global _follow_up_evaluator
    _follow_up_evaluator = None
    logger.debug("Follow-up evaluator cache cleared")
