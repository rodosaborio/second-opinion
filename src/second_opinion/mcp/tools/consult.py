"""
AI Consultation MCP tool implementation.

This module provides the `consult` tool for AI-to-AI consultation, enabling
task delegation, expert opinions, and multi-turn problem-solving conversations
with specialized models.
"""

import logging
from decimal import Decimal
from typing import Any

from ...cli.main import filter_think_tags
from ...clients import detect_model_provider
from ...core.evaluator import get_evaluator
from ...core.models import Message, ModelRequest, TaskComplexity
from ...orchestration import get_conversation_orchestrator
from ...orchestration.types import StorageContext
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import get_cost_guard
from ...utils.domain_classifier import classify_consultation_domain
from ...utils.followup_evaluator import evaluate_follow_up_need
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)
from ..session import MCPSession

logger = logging.getLogger(__name__)

# Note: Removed global session storage - sessions are now managed per tool call
# following the same pattern as other working MCP tools


class ConsultationSession(MCPSession):
    """Extended session management for multi-turn conversations."""

    def __init__(
        self, consultation_type: str, target_model: str, session_id: str | None = None
    ):
        """
        Initialize a consultation session.

        Args:
            consultation_type: Type of consultation (quick, deep, delegate, brainstorm)
            target_model: AI model to consult with
            session_id: Optional session ID, generates new if None
        """
        super().__init__(session_id)
        self.consultation_type = consultation_type
        self.target_model = target_model
        self.messages: list[Message] = []
        self.turn_count = 0
        self.status = "active"  # active, paused, completed
        self.conversation_summary = ""
        self.domain = "general"

        # Enhanced session state management
        self.key_topics: list[str] = []  # Track main topics discussed
        self.decisions_made: list[dict] = []  # Track key decisions and conclusions
        self.context_hierarchy: dict[str, Any] = {  # Structured context preservation
            "original_goal": "",
            "current_focus": "",
            "technical_details": [],
            "constraints": [],
            "progress_markers": [],
        }
        self.conversation_quality_metrics = {
            "completeness_score": 0.0,
            "user_satisfaction_indicators": [],
            "complexity_handled": "low",
        }

        logger.debug(
            f"Created consultation session {self.session_id} for {consultation_type} with {target_model}"
        )

    async def add_turn(self, query: str, response: str, cost: Decimal) -> None:
        """
        Add a conversation turn with cost tracking and structured information extraction.

        Args:
            query: User query for this turn
            response: AI response for this turn
            cost: Cost incurred for this turn
        """
        self.messages.extend(
            [
                Message(role="user", content=query),
                Message(role="assistant", content=response),
            ]
        )
        self.turn_count += 1
        self.record_cost("consult", cost, self.target_model)

        # Extract structured information from the turn
        await self._extract_turn_insights(query, response)

        logger.debug(
            f"Session {self.session_id}: Added turn {self.turn_count}, cost ${cost:.4f}"
        )

    async def _extract_turn_insights(self, query: str, response: str) -> None:
        """
        Extract key insights and structured information from a conversation turn.

        Args:
            query: User query
            response: AI response
        """
        # Extract key topics from query (simple keyword extraction)
        query_topics = self._extract_topics_from_text(query)
        for topic in query_topics:
            if topic not in self.key_topics and len(self.key_topics) < 10:
                self.key_topics.append(topic)

        # Update original goal if this is the first turn
        if self.turn_count == 1:
            self.context_hierarchy["original_goal"] = query[:200] + (
                "..." if len(query) > 200 else ""
            )

        # Extract decisions and conclusions from response
        if any(
            indicator in response.lower()
            for indicator in ["recommend", "suggest", "conclude", "decision", "should"]
        ):
            decision = {
                "turn": self.turn_count,
                "type": "recommendation",
                "content": response[:150] + ("..." if len(response) > 150 else ""),
                "confidence": self._estimate_confidence(response),
            }
            self.decisions_made.append(decision)

        # Update current focus
        self.context_hierarchy["current_focus"] = query[:100] + (
            "..." if len(query) > 100 else ""
        )

        # Track progress markers
        if self.turn_count > 1:
            progress_marker = f"Turn {self.turn_count}: {query[:50]}..."
            self.context_hierarchy["progress_markers"].append(progress_marker)
            # Keep only last 5 progress markers
            if len(self.context_hierarchy["progress_markers"]) > 5:
                self.context_hierarchy["progress_markers"] = self.context_hierarchy[
                    "progress_markers"
                ][-5:]

    def _extract_topics_from_text(self, text: str) -> list[str]:
        """Extract key topics from text using simple heuristics."""
        # Simple topic extraction - look for technical terms, proper nouns, key concepts
        words = text.lower().split()
        topics = []

        # Common technical/domain keywords that indicate topics
        topic_indicators = [
            "architecture",
            "database",
            "api",
            "performance",
            "security",
            "scalability",
            "microservices",
            "python",
            "javascript",
            "react",
            "node",
            "aws",
            "docker",
            "kubernetes",
            "machine learning",
            "ai",
            "algorithm",
            "optimization",
            "authentication",
            "authorization",
            "cache",
            "redis",
            "mongodb",
            "postgresql",
        ]

        for word in words:
            clean_word = word.strip(".,!?;:")
            if clean_word in topic_indicators and clean_word not in topics:
                topics.append(clean_word)

        return topics[:3]  # Return top 3 topics

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence level of AI response based on language used."""
        response_lower = response.lower()

        # High confidence indicators
        high_confidence = [
            "definitely",
            "certainly",
            "clearly",
            "obviously",
            "recommend",
            "should",
        ]
        # Low confidence indicators
        low_confidence = [
            "might",
            "maybe",
            "possibly",
            "consider",
            "could be",
            "uncertain",
        ]

        high_count = sum(
            1 for indicator in high_confidence if indicator in response_lower
        )
        low_count = sum(
            1 for indicator in low_confidence if indicator in response_lower
        )

        if high_count > low_count:
            return 0.8
        elif low_count > high_count:
            return 0.4
        else:
            return 0.6

    def can_continue(self, max_turns: int, cost_limit: Decimal) -> bool:
        """
        Check if conversation can continue.

        Args:
            max_turns: Maximum allowed turns
            cost_limit: Maximum allowed cost

        Returns:
            True if conversation can continue
        """
        return (
            self.turn_count < max_turns
            and self.total_cost < cost_limit
            and self.status == "active"
        )

    def get_conversation_context(
        self, max_exchanges: int = 4, max_chars_per_exchange: int = 350
    ) -> str:
        """
        Get conversation context for follow-up turns with enhanced context window.

        Args:
            max_exchanges: Maximum number of exchanges to include (default: 4 = 8 messages)
            max_chars_per_exchange: Maximum characters per message (default: 350)

        Returns:
            Formatted conversation history with intelligent truncation
        """
        if not self.messages:
            return ""

        context_parts = []
        total_chars = 0
        max_total_chars = max_exchanges * max_chars_per_exchange * 2  # Q&A pairs

        # Process messages in reverse order (most recent first)
        for i in range(len(self.messages) - 2, -1, -2):  # Step by 2, backwards
            if i + 1 < len(self.messages) and len(context_parts) < max_exchanges * 2:
                user_msg = self.messages[i].content
                ai_msg = self.messages[i + 1].content

                # Smart truncation preserving key information
                truncated_user = self._smart_truncate(user_msg, max_chars_per_exchange)
                truncated_ai = self._smart_truncate(ai_msg, max_chars_per_exchange)

                # Check if adding this exchange would exceed character limit
                exchange_chars = (
                    len(truncated_user) + len(truncated_ai) + 40
                )  # formatting
                if total_chars + exchange_chars > max_total_chars:
                    break

                # Add to context (prepend to maintain chronological order)
                context_parts.insert(0, f"Previous A: {truncated_ai}")
                context_parts.insert(0, f"Previous Q: {truncated_user}")
                total_chars += exchange_chars

        return "\n".join(context_parts)

    def get_enhanced_session_summary(self) -> dict[str, Any]:
        """
        Get comprehensive session summary with structured information.

        Returns:
            Dictionary containing session metrics, topics, decisions, and context
        """
        return {
            "session_id": self.session_id,
            "consultation_type": self.consultation_type,
            "domain": self.domain,
            "turn_count": self.turn_count,
            "status": self.status,
            "total_cost": float(self.total_cost),
            "key_topics": self.key_topics,
            "decisions_made": self.decisions_made,
            "context_hierarchy": self.context_hierarchy,
            "conversation_quality_metrics": self.conversation_quality_metrics,
            "context_summary": self.get_conversation_context(
                max_exchanges=2, max_chars_per_exchange=200
            ),
        }

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Intelligently truncate text preserving key information.

        Args:
            text: Text to truncate
            max_chars: Maximum characters to preserve

        Returns:
            Truncated text with key information preserved
        """
        if len(text) <= max_chars:
            return text

        # For very long text, try to preserve:
        # 1. Beginning (context/question)
        # 2. Key phrases (ending sentences, conclusions)
        # 3. Technical terms and specifics

        # Simple smart truncation: preserve beginning and try to end at sentence boundary
        if max_chars < 100:
            return text[: max_chars - 3] + "..."

        # Take first 70% of available chars from beginning
        beginning_chars = int(max_chars * 0.7)
        beginning = text[:beginning_chars]

        # Try to find a good ending point in the remaining text
        remaining_chars = max_chars - beginning_chars - 3  # -3 for "..."
        if remaining_chars > 20:
            # Look for sentence endings in the latter part of the text
            latter_part = text[-(remaining_chars + 50) :]  # Look ahead a bit
            sentence_endings = [". ", "! ", "? ", "\n"]

            best_ending = ""
            for ending in sentence_endings:
                pos = latter_part.find(ending)
                if pos != -1 and pos <= remaining_chars:
                    best_ending = latter_part[: pos + 1]
                    break

            if best_ending:
                return beginning + "..." + best_ending.strip()

        return beginning + "..."

    @staticmethod
    def _build_session_recovery_context(
        conversation_history: list[dict],
        max_conversations: int = 3,
        max_chars_per_exchange: int = 300,
    ) -> str:
        """
        Build context from conversation history for session recovery.

        Args:
            conversation_history: List of conversation dictionaries from storage
            max_conversations: Maximum number of conversations to include
            max_chars_per_exchange: Maximum characters per exchange

        Returns:
            Formatted context string for session recovery
        """
        if not conversation_history:
            return ""

        context_parts = []
        total_chars = 0
        max_total_chars = max_conversations * max_chars_per_exchange * 2

        # Take most recent conversations up to the limit
        recent_conversations = conversation_history[-max_conversations:]

        for conv in recent_conversations:
            user_prompt = conv.get("user_prompt", "")

            # Smart truncation for user prompt
            if len(user_prompt) > max_chars_per_exchange:
                # Use the same smart truncation logic
                session = ConsultationSession("temp", "temp")
                truncated_prompt = session._smart_truncate(
                    user_prompt, max_chars_per_exchange
                )
            else:
                truncated_prompt = user_prompt

            # Find primary response
            primary_response = ""
            for resp in conv.get("responses", []):
                if resp.get("response_type") == "primary":
                    primary_response = resp.get("content", "")
                    break

            # Smart truncation for response
            if len(primary_response) > max_chars_per_exchange:
                session = ConsultationSession("temp", "temp")
                truncated_response = session._smart_truncate(
                    primary_response, max_chars_per_exchange
                )
            else:
                truncated_response = primary_response

            # Check total character limit
            exchange_chars = len(truncated_prompt) + len(truncated_response) + 40
            if total_chars + exchange_chars > max_total_chars:
                break

            context_parts.append(f"Previous Q: {truncated_prompt}")
            if truncated_response:
                context_parts.append(f"Previous A: {truncated_response}")

            total_chars += exchange_chars

        return "\n".join(context_parts)


class ConsultationModelRouter:
    """Smart model selection based on consultation type and context."""

    def recommend_model(
        self,
        consultation_type: str,
        context: str | None = None,
        task_complexity: TaskComplexity = TaskComplexity.MODERATE,
        user_specified_model: str | None = None,
    ) -> str:
        """
        Recommend optimal model for consultation needs.

        Args:
            consultation_type: Type of consultation
            context: Additional context about the task
            task_complexity: Assessed complexity of the task
            user_specified_model: User's explicit model choice (overrides recommendations)

        Returns:
            Model name in OpenRouter format
        """

        # User specified model always wins
        if user_specified_model:
            return user_specified_model

        # Smart routing based on consultation type and complexity
        if consultation_type == "delegate":
            if task_complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
                return "openai/gpt-4o-mini"  # Cost-effective for routine tasks
            else:
                return "anthropic/claude-3-5-sonnet"  # Quality for complex delegation

        elif consultation_type == "expert" or task_complexity == TaskComplexity.COMPLEX:
            return "anthropic/claude-3-opus"  # Premium for expert consultation

        elif consultation_type == "brainstorm":
            return "openai/gpt-4o"  # Creative and collaborative

        else:  # quick consultation
            return "anthropic/claude-3-5-sonnet"  # Reliable default

    async def detect_domain_specialization(
        self, query: str, context: str | None = None
    ) -> str:
        """
        Detect if query needs domain-specific model routing using LLM classification.

        Args:
            query: The consultation query
            context: Additional context

        Returns:
            Domain classification (coding, performance, creative, general)
        """
        try:
            domain = await classify_consultation_domain(query, context)
            logger.debug(
                f"LLM classified domain as '{domain}' for query: {query[:50]}..."
            )
            return domain
        except Exception as e:
            logger.warning(f"Domain classification failed, using 'general': {e}")
            return "general"


class TurnController:
    """Manage multi-turn conversation flow and context."""

    def __init__(self):
        self.model_router = ConsultationModelRouter()

    async def conduct_consultation(
        self,
        session: ConsultationSession,
        initial_query: str,
        max_turns: int,
        cost_limit: Decimal,
    ) -> dict[str, Any]:
        """
        Conduct multi-turn consultation with intelligent flow control.

        Args:
            session: Consultation session
            initial_query: Initial query to start conversation
            max_turns: Maximum conversation turns
            cost_limit: Maximum cost limit

        Returns:
            Detailed consultation results with conversation summary
        """

        results = {
            "conversation_turns": [],
            "total_cost": Decimal("0.0"),
            "consultation_summary": "",
            "recommendations": [],
            "next_steps": [],
        }

        current_query = initial_query

        for turn in range(max_turns):
            if not session.can_continue(max_turns, cost_limit):
                logger.info(
                    f"Session {session.session_id}: Cannot continue - turn {turn}, cost ${session.total_cost:.4f}"
                )
                break

            # Get response from target model
            try:
                response, cost = await self._get_consultation_response(
                    session, current_query, turn
                )

                # Record turn
                await session.add_turn(current_query, response, cost)
                results["conversation_turns"].append(
                    {
                        "turn": turn + 1,
                        "query": current_query,
                        "response": response,
                        "cost": float(cost),
                    }
                )
                results["total_cost"] += cost

                # For single-turn consultations, we're done
                if session.consultation_type in ["quick", "delegate"] or max_turns == 1:
                    break

                # Determine if follow-up is needed for multi-turn types
                follow_up = await self._assess_follow_up_need(
                    session, response, turn, max_turns, current_query
                )

                if not follow_up["needed"]:
                    break

                current_query = follow_up["query"]

            except Exception as e:
                logger.error(
                    f"Turn {turn + 1} failed for session {session.session_id}: {e}"
                )
                # Record error but continue if possible
                results["conversation_turns"].append(
                    {
                        "turn": turn + 1,
                        "query": current_query,
                        "response": f"Error occurred: {str(e)}",
                        "cost": 0.0,
                    }
                )
                # Re-raise the exception for proper error handling
                raise e

        # Generate consultation summary and recommendations
        try:
            results["consultation_summary"] = await self._generate_summary(session)
            results["recommendations"] = await self._extract_recommendations(session)
        except Exception as e:
            logger.warning(
                f"Failed to generate summary for session {session.session_id}: {e}"
            )
            results["consultation_summary"] = (
                "Summary unavailable due to processing error."
            )
            results["recommendations"] = []

        # Mark session as completed
        session.status = "completed"

        return results

    async def _get_consultation_response(
        self, session: ConsultationSession, query: str, turn: int
    ) -> tuple[str, Decimal]:
        """
        Get response from consultation model.

        Args:
            session: Consultation session
            query: Query for this turn
            turn: Turn number (0-indexed)

        Returns:
            Tuple of (response_content, cost)
        """

        # Build context-aware prompt for multi-turn conversations
        if turn > 0 and session.consultation_type in ["deep", "brainstorm"]:
            conversation_context = session.get_conversation_context()
            enhanced_query = f"""Previous conversation context:
{conversation_context}

Current question: {query}

Please provide a thoughtful response that builds on our previous discussion."""
        elif session.conversation_summary and session.consultation_type in [
            "deep",
            "brainstorm",
        ]:
            # Use loaded session context for session recovery
            enhanced_query = f"""Previous conversation context from our session:
{session.conversation_summary}

Current question: {query}

Please provide a thoughtful response that builds on our previous discussion."""
        else:
            enhanced_query = query

        # Detect provider and create client
        provider = detect_model_provider(session.target_model)
        client = create_client_from_config(provider)

        # Create model request
        request = ModelRequest(
            model=session.target_model,
            messages=[Message(role="user", content=enhanced_query)],
            max_tokens=2000,
            temperature=0.1 if session.consultation_type == "delegate" else 0.3,
            system_prompt=self._get_system_prompt(session),
        )

        # Get response
        response = await client.complete(request)
        clean_response = filter_think_tags(response.content)

        return clean_response, response.cost_estimate

    def _get_system_prompt(self, session: ConsultationSession) -> str:
        """
        Get system prompt tailored to consultation type.

        Args:
            session: Consultation session

        Returns:
            System prompt for the consultation
        """
        base_prompt = f"You are an AI assistant providing {session.consultation_type} consultation."

        if session.consultation_type == "quick":
            return (
                f"{base_prompt} Provide a concise, expert opinion with clear reasoning."
            )
        elif session.consultation_type == "delegate":
            return (
                f"{base_prompt} Complete the requested task efficiently and accurately."
            )
        elif session.consultation_type == "deep":
            return f"{base_prompt} Provide comprehensive analysis with detailed explanations and multiple perspectives."
        elif session.consultation_type == "brainstorm":
            return f"{base_prompt} Be creative and explore multiple approaches, thinking outside the box."
        else:
            return base_prompt

    async def _assess_follow_up_need(
        self,
        session: ConsultationSession,
        response: str,
        turn: int,
        max_turns: int,
        user_query: str = "",
    ) -> dict[str, Any]:
        """
        Intelligently assess if follow-up questions are needed using LLM evaluation.

        Args:
            session: Consultation session
            response: AI response from current turn
            turn: Current turn number
            max_turns: Maximum allowed turns
            user_query: User query for this turn

        Returns:
            Dictionary with follow-up assessment
        """
        try:
            # Get conversation context for evaluation
            conversation_context = session.get_conversation_context(
                max_exchanges=2, max_chars_per_exchange=200
            )

            # Use LLM-based evaluation for follow-up assessment
            evaluation_result = await evaluate_follow_up_need(
                consultation_type=session.consultation_type,
                user_query=user_query,
                ai_response=response,
                turn_number=turn + 1,  # Convert 0-indexed to 1-indexed
                max_turns=max_turns,
                conversation_context=conversation_context,
            )

            # Record the evaluation cost in the session
            if "estimated_cost" in evaluation_result:
                session.record_cost(
                    "follow_up_evaluation",
                    evaluation_result["estimated_cost"],
                    "gpt-4o-mini",
                )

            # Convert to expected format
            needs_follow_up = evaluation_result.get("needs_followup", False)
            suggested_query = evaluation_result.get("suggested_query")

            # Use intelligent suggested query or fallback to type-specific defaults
            if needs_follow_up and suggested_query:
                follow_up_query = suggested_query
            elif needs_follow_up:
                # Fallback to consultation-type specific defaults
                if session.consultation_type == "deep":
                    follow_up_query = "Please elaborate on the most important considerations and provide specific implementation guidance."
                elif session.consultation_type == "brainstorm":
                    follow_up_query = "What are 2-3 alternative approaches we should consider, and what are their trade-offs?"
                else:
                    follow_up_query = (
                        "Could you provide more details or explore this further?"
                    )
            else:
                follow_up_query = None

            logger.debug(
                f"LLM follow-up evaluation: needs_followup={needs_follow_up}, "
                f"confidence={evaluation_result.get('confidence', 0.0):.2f}, "
                f"reason={evaluation_result.get('reason', 'N/A')}"
            )

            return {
                "needed": needs_follow_up,
                "query": follow_up_query,
                "confidence": evaluation_result.get("confidence", 0.5),
                "reason": evaluation_result.get("reason", "LLM evaluation completed"),
                "cost": evaluation_result.get("estimated_cost", 0.0),
            }

        except Exception as e:
            logger.warning(f"LLM follow-up evaluation failed, using fallback: {e}")

            # Fallback to simple heuristic if LLM evaluation fails
            follow_up_indicators = [
                "would you like me to elaborate",
                "need more details",
                "want me to explore",
                "should we dive deeper",
                "any specific aspects",
            ]

            response_lower = response.lower()
            needs_follow_up = any(
                indicator in response_lower for indicator in follow_up_indicators
            )

            # Only continue if consultation type supports multi-turn and we haven't reached limit
            if (
                needs_follow_up
                and turn < max_turns - 1
                and session.consultation_type in ["deep", "brainstorm"]
            ):
                if session.consultation_type == "deep":
                    query = "Please elaborate on the most important considerations and provide specific implementation guidance."
                else:
                    query = "What are 2-3 alternative approaches we should consider, and what are their trade-offs?"

                return {
                    "needed": True,
                    "query": query,
                    "confidence": 0.5,
                    "reason": "Fallback heuristic assessment",
                    "cost": 0.0,
                }

            return {
                "needed": False,
                "query": None,
                "confidence": 0.8,
                "reason": "Fallback: no follow-up needed",
                "cost": 0.0,
            }

    async def _generate_summary(self, session: ConsultationSession) -> str:
        """
        Generate a summary of the consultation session.

        Args:
            session: Consultation session

        Returns:
            Summary of the consultation
        """
        if not session.messages:
            return "No conversation occurred."

        # Extract key points from the conversation
        if session.consultation_type == "delegate":
            return f"Task delegation completed using {session.target_model}. The AI assistant completed the requested task with {session.turn_count} interaction(s)."
        elif session.consultation_type == "quick":
            return f"Quick expert consultation using {session.target_model}. Received focused advice on the query in a single interaction."
        elif session.consultation_type in ["deep", "brainstorm"]:
            return f"Multi-turn {session.consultation_type} session with {session.target_model} over {session.turn_count} turns. Explored the topic comprehensively with detailed analysis."
        else:
            return f"Consultation completed with {session.target_model} in {session.turn_count} turn(s)."

    async def _extract_recommendations(self, session: ConsultationSession) -> list[str]:
        """
        Extract key recommendations from the consultation.

        Args:
            session: Consultation session

        Returns:
            List of key recommendations
        """
        if not session.messages:
            return []

        recommendations = []

        # Simple extraction based on consultation type
        if session.consultation_type == "delegate":
            recommendations.append(
                "Review the completed task for accuracy and completeness"
            )
            recommendations.append("Integrate the results into your workflow")
            recommendations.append("Consider similar delegations for routine tasks")
        elif session.consultation_type == "expert":
            recommendations.append("Implement the recommended approach with confidence")
            recommendations.append("Monitor results and iterate as needed")
            recommendations.append("Document learnings for future reference")
        elif session.consultation_type == "deep":
            recommendations.append(
                "Review the comprehensive analysis and choose your approach"
            )
            recommendations.append("Start with the highest-priority recommendations")
            recommendations.append("Schedule follow-up consultation if needed")
        elif session.consultation_type == "brainstorm":
            recommendations.append(
                "Evaluate the explored alternatives based on your constraints"
            )
            recommendations.append("Prototype the most promising approaches")
            recommendations.append("Combine insights from multiple solutions")

        return recommendations


# Removed broken session management functions - using direct ConsultationSession instantiation
# following the pattern of other working MCP tools


def calculate_delegation_savings(target_model: str) -> Decimal:
    """
    Calculate estimated savings from task delegation.

    Args:
        target_model: Model used for delegation

    Returns:
        Estimated cost savings vs premium model
    """
    # Simple heuristic based on model tier
    model_lower = target_model.lower()

    if "mini" in model_lower or "haiku" in model_lower:
        return Decimal("0.08")  # Significant savings vs premium
    elif "sonnet" in model_lower or "gpt-4o" in model_lower:
        return Decimal("0.03")  # Moderate savings
    else:
        return Decimal("0.01")  # Minimal savings


async def consult_tool(
    query: str,
    consultation_type: str = "quick",
    target_model: str | None = None,
    session_id: str | None = None,
    max_turns: int = 3,
    context: str | None = None,
    cost_limit: float | None = None,
) -> str:
    """
    Consult with AI models for expert opinions, task delegation, and problem solving.

    This tool enables AI-to-AI consultation across different specialized models,
    supporting both single-turn expert opinions and multi-turn collaborative
    problem-solving sessions.

    Args:
        query: The question or task to consult about
        consultation_type: Type of consultation:
            - "quick": Single-turn expert opinion (default)
            - "deep": Multi-turn comprehensive analysis
            - "delegate": Task delegation to cost-effective model
            - "brainstorm": Creative collaborative exploration
        target_model: Specific model to consult (auto-selected if None):
            - OpenRouter format: "anthropic/claude-3-5-sonnet", "openai/gpt-4o"
            - Local models: "qwen3-4b-mlx", "codestral-22b-v0.1"
        session_id: Continue existing consultation session (for multi-turn)
        max_turns: Maximum conversation turns for multi-turn consultations (1-5)
        context: Additional context about the task domain for better model routing
        cost_limit: Maximum cost limit for this consultation in USD

    Returns:
        Consultation results with expert insights, cost analysis, and recommendations

    CONSULTATION PATTERNS:

    Quick Expert Opinion:
        result = await consult_tool(
            query="Should I use async/await or threading for this I/O operation?",
            consultation_type="quick",
            context="performance optimization"
        )

    Task Delegation:
        result = await consult_tool(
            query="Write unit tests for this function: def fibonacci(n): ...",
            consultation_type="delegate",
            target_model="openai/gpt-4o-mini"  # Cost-effective
        )

    Deep Problem Solving:
        result = await consult_tool(
            query="Help me design a scalable authentication system",
            consultation_type="deep",
            max_turns=3,
            context="system architecture"
        )

    Creative Brainstorming:
        result = await consult_tool(
            query="Explore different approaches to this performance bottleneck",
            consultation_type="brainstorm",
            max_turns=2
        )
    """
    try:
        # Input validation and sanitization
        clean_query = sanitize_prompt(query, SecurityContext.USER_PROMPT)

        # Validate consultation type
        valid_types = ["quick", "deep", "delegate", "brainstorm"]
        if consultation_type not in valid_types:
            return f"❌ **Invalid consultation type**: {consultation_type}\n\n**Valid types**: {', '.join(valid_types)}"

        # Validate max_turns
        if not (1 <= max_turns <= 5):
            return "❌ **Invalid max_turns**: Must be between 1 and 5"

        # Sanitize context if provided
        clean_context = None
        if context:
            clean_context = sanitize_prompt(context, SecurityContext.USER_PROMPT)

        # Validate cost limit
        cost_limit_decimal = None
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            # Get default cost limit based on consultation type
            if consultation_type == "delegate":
                cost_limit_decimal = Decimal("0.10")  # Lower for delegation
            elif consultation_type == "deep":
                cost_limit_decimal = Decimal("0.50")  # Higher for multi-turn
            else:
                cost_limit_decimal = Decimal("0.25")  # Standard default

        # Validate target model if provided
        if target_model:
            try:
                target_model = validate_model_name(target_model)
            except Exception as e:
                return f"❌ **Invalid target model**: {str(e)}\n\n**Suggested formats:**\n- Cloud: `anthropic/claude-3-5-sonnet`, `openai/gpt-4o-mini`\n- Local: `qwen3-4b-mlx`, `codestral-22b-v0.1`"

        logger.info(
            f"Starting consult tool: type={consultation_type}, query_length={len(clean_query)}, target_model={target_model}"
        )

        # Session management strategy: Load conversation history for session recovery
        consultation_session = None
        previous_context = ""

        if session_id:
            # For multi-turn consultations, attempt to retrieve conversation context
            # from the conversation storage for session recovery
            logger.info(f"Continuing consultation with session_id: {session_id}")

            try:
                from ...database.store import get_conversation_store

                store = get_conversation_store()
                # Use a fresh async session for recovery to avoid greenlet issues
                conversation_history = await store.get_session_conversation_history(
                    session_id
                )

                if conversation_history:
                    logger.info(
                        f"Loaded {len(conversation_history)} previous conversations for session {session_id}"
                    )

                    # Build enhanced context from previous conversations
                    previous_context = (
                        ConsultationSession._build_session_recovery_context(
                            conversation_history,
                            max_conversations=3,
                            max_chars_per_exchange=300,
                        )
                    )
                    logger.debug(
                        f"Built session context: {len(previous_context)} chars"
                    )
                else:
                    logger.info(
                        f"No conversation history found for session {session_id}, starting fresh"
                    )

            except Exception as e:
                logger.warning(f"Failed to load session context for {session_id}: {e}")
                # Continue without context - non-fatal error

        # Initialize components for model selection (always needed)
        evaluator = get_evaluator()
        model_router = ConsultationModelRouter()

        # Classify task complexity for smart model routing
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_query)
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")
            task_complexity = TaskComplexity.MODERATE

        # Select target model
        recommended_model = model_router.recommend_model(
            consultation_type=consultation_type,
            context=clean_context or "",
            task_complexity=task_complexity,
            user_specified_model=target_model or "",
        )

        # Detect domain for specialized routing
        domain = await model_router.detect_domain_specialization(
            clean_query, clean_context or ""
        )

        # Create a consultation session with loaded context if available
        consultation_session = ConsultationSession(
            consultation_type=consultation_type,
            target_model=recommended_model,
            session_id=session_id,  # Use provided session_id or let it generate new one
        )
        consultation_session.domain = domain

        # If we loaded previous context, add it to the session for context-aware responses
        if previous_context:
            consultation_session.conversation_summary = previous_context
            logger.info(
                f"Session {consultation_session.session_id} initialized with previous context"
            )

        logger.info(
            f"Created consultation session {consultation_session.session_id} with {recommended_model} for {domain} domain"
        )

        # Get cost guard for budget management
        cost_guard = get_cost_guard()

        # Estimate total cost for the consultation
        estimated_cost = await _estimate_consultation_cost(
            consultation_session, clean_query, max_turns, clean_context or ""
        )

        # Check budget and reserve
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "consult",
                consultation_session.target_model,
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Conduct consultation
        turn_controller = TurnController()
        try:
            results = await turn_controller.conduct_consultation(
                session=consultation_session,
                initial_query=clean_query,
                max_turns=max_turns,
                cost_limit=cost_limit_decimal,
            )

            # Record actual cost
            await cost_guard.record_actual_cost(
                reservation_id,
                results["total_cost"],
                consultation_session.target_model,
                "consult",
            )

            # Store conversation (optional, non-fatal if it fails)
            try:
                orchestrator = get_conversation_orchestrator()
                storage_context = StorageContext(
                    interface_type="mcp",  # This tool is called from MCP
                    tool_name="consult",
                    session_id=consultation_session.session_id,  # Use the consultation session ID
                    context=clean_context,
                    save_conversation=False,  # Temporarily disabled due to async issues
                )

                # Extract responses from consultation results for storage
                consultation_responses = []
                for turn_data in results.get("conversation_turns", []):
                    # Create mock ModelResponse for each turn
                    from ...core.models import ModelResponse, TokenUsage

                    mock_response = ModelResponse(
                        content=turn_data["response"],
                        model=consultation_session.target_model,
                        usage=TokenUsage(
                            input_tokens=10,  # Estimated - consult doesn't track exact tokens
                            output_tokens=len(turn_data["response"].split())
                            * 2,  # Rough estimate
                            total_tokens=10 + len(turn_data["response"].split()) * 2,
                        ),
                        cost_estimate=Decimal(str(turn_data["cost"])),
                        provider=detect_model_provider(
                            consultation_session.target_model
                        ),
                    )
                    consultation_responses.append(mock_response)

                if consultation_responses:
                    await orchestrator.handle_interaction(
                        prompt=clean_query,
                        responses=consultation_responses,
                        storage_context=storage_context,
                        evaluation_result={
                            "consultation_type": consultation_type,
                            "consultation_results": results,
                            "session_summary": consultation_session.get_session_summary(),
                        },
                    )
                    logger.debug("Consultation storage completed successfully")
            except Exception as storage_error:
                # Storage failure is non-fatal - continue with normal tool execution
                logger.warning(
                    f"Consultation storage failed (non-fatal): {storage_error}"
                )

            # Format and return results
            return _format_consultation_response(
                consultation_type=consultation_type,
                results=results,
                session=consultation_session,
                context=clean_context or "",
            )

        except Exception as e:
            logger.error(
                f"Consultation failed for session {consultation_session.session_id}: {e}"
            )
            # Note: Cost reservation cleanup happens automatically on context exit
            return f"❌ **Consultation Error**: {str(e)}\n\nPlease try again with simpler parameters or check the logs for details."

    except Exception as e:
        logger.error(f"Unexpected error in consult tool: {e}")
        return (
            f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs and try again."
        )


async def _estimate_consultation_cost(
    session: ConsultationSession, query: str, max_turns: int, context: str | None = None
) -> Decimal:
    """
    Estimate the total cost for a consultation.

    Args:
        session: Consultation session
        query: Initial query
        max_turns: Maximum turns for the consultation
        context: Additional context

    Returns:
        Estimated total cost
    """
    try:
        provider = detect_model_provider(session.target_model)
        client = create_client_from_config(provider)

        # Estimate cost for a typical turn
        request = ModelRequest(
            model=session.target_model,
            messages=[Message(role="user", content=query)],
            max_tokens=2000,
            temperature=0.1,
            system_prompt=context,
        )

        single_turn_cost = await client.estimate_cost(request)

        # For multi-turn consultations, estimate based on conversation length
        if session.consultation_type in ["deep", "brainstorm"] and max_turns > 1:
            # Subsequent turns typically longer due to context
            total_cost = single_turn_cost * max_turns * Decimal("1.3")
        else:
            total_cost = single_turn_cost

        return total_cost

    except Exception as e:
        logger.warning(f"Failed to estimate consultation cost: {e}")
        # Conservative fallback estimate
        return Decimal("0.05") * max_turns


def _format_consultation_response(
    consultation_type: str,
    results: dict[str, Any],
    session: ConsultationSession,
    context: str | None = None,
) -> str:
    """
    Format consultation results for optimal MCP client display.

    Args:
        consultation_type: Type of consultation
        results: Consultation results
        session: Consultation session
        context: Additional context

    Returns:
        Formatted response for MCP client
    """
    sections = []

    # Header based on consultation type
    if consultation_type == "quick":
        sections.append("# 🎯 Quick Expert Consultation")
    elif consultation_type == "delegate":
        sections.append("# 📋 Task Delegation Results")
    elif consultation_type == "deep":
        sections.append("# 🔍 Deep Consultation Session")
    elif consultation_type == "brainstorm":
        sections.append("# 💡 Brainstorming Session")

    sections.append("")

    # Session info
    sections.append("## 📝 Consultation Summary")
    sections.append(f"**Session ID**: {session.session_id}")
    sections.append(f"**Model**: {session.target_model}")
    sections.append(f"**Domain**: {session.domain}")
    if context:
        sections.append(f"**Context**: {context}")
    sections.append("")
    sections.append(results["consultation_summary"])
    sections.append("")

    # Display conversation for multi-turn sessions
    if len(results["conversation_turns"]) > 1:
        sections.append("## 💬 Conversation Flow")
        for turn_data in results["conversation_turns"]:
            sections.append(f"### Turn {turn_data['turn']}")
            sections.append(
                f"**Query**: {turn_data['query'][:200]}{'...' if len(turn_data['query']) > 200 else ''}"
            )
            sections.append("")
            sections.append(
                f"**Response**: {turn_data['response'][:800]}{'...' if len(turn_data['response']) > 800 else ''}"
            )
            sections.append("")
            sections.append(f"**Cost**: ${turn_data['cost']:.4f}")
            sections.append("")
    else:
        # Single turn - show full response
        if results["conversation_turns"]:
            turn_data = results["conversation_turns"][0]
            sections.append("## 💬 Consultation Response")
            sections.append(turn_data["response"])
            sections.append("")

    # Key recommendations
    if results["recommendations"]:
        sections.append("## 🎯 Key Recommendations")
        for i, rec in enumerate(results["recommendations"], 1):
            sections.append(f"{i}. {rec}")
        sections.append("")

    # Cost analysis
    sections.append("## 💰 Cost Analysis")
    sections.append(f"**Total Cost**: ${results['total_cost']:.4f}")
    sections.append(f"**Turns Completed**: {session.turn_count}")

    # Value assessment for delegation
    if consultation_type == "delegate":
        estimated_savings = calculate_delegation_savings(session.target_model)
        sections.append(
            f"**Estimated Savings**: ${estimated_savings:.4f} vs premium model"
        )
        sections.append(
            f"**Cost Efficiency**: {((estimated_savings / (estimated_savings + results['total_cost'])) * 100):.1f}% savings"
        )

    sections.append("")

    # Continue session option for multi-turn types
    if consultation_type in ["deep", "brainstorm"] and session.status == "completed":
        sections.append("## 🔄 Continue Consultation")
        sections.append(f"**Session ID**: `{session.session_id}`")
        sections.append(
            "Use this session ID to continue the conversation with follow-up questions."
        )
        sections.append("")

    # Actionable next steps
    sections.append("## 🚀 Next Steps")
    if consultation_type == "delegate":
        sections.append(
            "1. **Review the completed task** for accuracy and completeness"
        )
        sections.append("2. **Integrate the results** into your workflow")
        sections.append(
            "3. **Consider similar delegations** for routine tasks to save time and cost"
        )
    elif consultation_type == "quick":
        sections.append("1. **Implement the expert recommendation** with confidence")
        sections.append("2. **Monitor results** and adjust as needed")
        sections.append(
            "3. **Use deep consultation** for more complex follow-up questions"
        )
    elif consultation_type == "deep":
        sections.append(
            "1. **Review the comprehensive analysis** and select your preferred approach"
        )
        sections.append(
            "2. **Start implementation** with the highest-priority recommendations"
        )
        sections.append(
            "3. **Continue the session** with specific implementation questions if needed"
        )
    elif consultation_type == "brainstorm":
        sections.append(
            "1. **Evaluate the explored alternatives** based on your specific constraints"
        )
        sections.append(
            "2. **Prototype the most promising approaches** to validate assumptions"
        )
        sections.append(
            "3. **Combine insights** from multiple solutions for an optimal approach"
        )

    sections.append("")
    sections.append("---")
    sections.append("*AI Consultation Complete - Ready to assist further! 🤖*")

    return "\n".join(sections)
