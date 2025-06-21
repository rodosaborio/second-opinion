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
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import get_cost_guard
from ...utils.domain_classifier import classify_consultation_domain
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)
from ..session import MCPSession

logger = logging.getLogger(__name__)

# Global consultation sessions storage
_consultation_sessions: dict[str, "ConsultationSession"] = {}


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

        logger.debug(
            f"Created consultation session {self.session_id} for {consultation_type} with {target_model}"
        )

    async def add_turn(self, query: str, response: str, cost: Decimal) -> None:
        """
        Add a conversation turn with cost tracking.

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

        logger.debug(
            f"Session {self.session_id}: Added turn {self.turn_count}, cost ${cost:.4f}"
        )

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

    def get_conversation_context(self) -> str:
        """
        Get conversation context for follow-up turns.

        Returns:
            Formatted conversation history
        """
        if not self.messages:
            return ""

        context_parts = []
        for i in range(0, len(self.messages), 2):
            if i + 1 < len(self.messages):
                user_msg = self.messages[i].content
                ai_msg = self.messages[i + 1].content
                context_parts.append(f"Previous Q: {user_msg[:200]}...")
                context_parts.append(f"Previous A: {ai_msg[:200]}...")

        return "\n".join(context_parts[-4:])  # Last 2 exchanges


class ConsultationModelRouter:
    """Smart model selection based on consultation type and context."""

    def recommend_model(
        self,
        consultation_type: str,
        context: str = None,
        task_complexity: TaskComplexity = TaskComplexity.MODERATE,
        user_specified_model: str = None,
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
        self, query: str, context: str = None
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
                    session, response, turn, max_turns
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
        self, session: ConsultationSession, response: str, turn: int, max_turns: int
    ) -> dict[str, Any]:
        """
        Intelligently assess if follow-up questions are needed.

        Args:
            session: Consultation session
            response: AI response from current turn
            turn: Current turn number
            max_turns: Maximum allowed turns

        Returns:
            Dictionary with follow-up assessment
        """

        # Simple heuristics for follow-up detection
        follow_up_indicators = [
            "would you like me to elaborate",
            "need more details",
            "want me to explore",
            "should we dive deeper",
            "any specific aspects",
            "would be helpful to know",
            "consider discussing",
        ]

        response_lower = response.lower()
        needs_follow_up = any(
            indicator in response_lower for indicator in follow_up_indicators
        )

        # Only continue if we haven't reached max turns and consultation type supports it
        if (
            needs_follow_up
            and turn < max_turns - 1
            and session.consultation_type in ["deep", "brainstorm"]
        ):
            # Generate intelligent follow-up query based on consultation type
            if session.consultation_type == "deep":
                return {
                    "needed": True,
                    "query": "Please elaborate on the most important considerations and provide specific implementation guidance.",
                }
            elif session.consultation_type == "brainstorm":
                return {
                    "needed": True,
                    "query": "What are 2-3 alternative approaches we should consider, and what are their trade-offs?",
                }

        return {"needed": False, "query": None}

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


def get_consultation_session(session_id: str) -> ConsultationSession | None:
    """
    Get existing consultation session by ID.

    Args:
        session_id: Session identifier

    Returns:
        ConsultationSession if found, None otherwise
    """
    return _consultation_sessions.get(session_id)


def create_consultation_session(
    consultation_type: str, target_model: str, session_id: str = None
) -> ConsultationSession:
    """
    Create and store a new consultation session.

    Args:
        consultation_type: Type of consultation
        target_model: AI model to consult with
        session_id: Optional session ID

    Returns:
        New ConsultationSession
    """
    session = ConsultationSession(consultation_type, target_model, session_id)
    _consultation_sessions[session.session_id] = session
    return session


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

        # Get or create consultation session
        if session_id:
            session = get_consultation_session(session_id)
            if not session:
                return f"❌ **Session not found**: {session_id}\n\nStart a new consultation without session_id or use a valid session ID."
            logger.info(f"Continuing session {session_id}")
        else:
            # Initialize components for model selection
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
                context=clean_context,
                task_complexity=task_complexity,
                user_specified_model=target_model,
            )

            # Detect domain for specialized routing
            domain = await model_router.detect_domain_specialization(
                clean_query, clean_context
            )

            # Create new session
            session = create_consultation_session(consultation_type, recommended_model)
            session.domain = domain

            logger.info(
                f"Created session {session.session_id} with {recommended_model} for {domain} domain"
            )

        # Get cost guard for budget management
        cost_guard = get_cost_guard()

        # Estimate total cost for the consultation
        estimated_cost = await _estimate_consultation_cost(
            session, clean_query, max_turns, clean_context
        )

        # Check budget and reserve
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "consult",
                session.target_model,
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"

        # Conduct consultation
        turn_controller = TurnController()
        try:
            results = await turn_controller.conduct_consultation(
                session=session,
                initial_query=clean_query,
                max_turns=max_turns,
                cost_limit=cost_limit_decimal,
            )

            # Record actual cost
            await cost_guard.record_actual_cost(
                reservation_id, results["total_cost"], session.target_model, "consult"
            )

            # Format and return results
            return _format_consultation_response(
                consultation_type=consultation_type,
                results=results,
                session=session,
                context=clean_context,
            )

        except Exception as e:
            logger.error(f"Consultation failed for session {session.session_id}: {e}")
            await cost_guard.release_reservation(reservation_id)
            return f"❌ **Consultation Error**: {str(e)}\n\nPlease try again with simpler parameters or check the logs for details."

    except Exception as e:
        logger.error(f"Unexpected error in consult tool: {e}")
        return (
            f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs and try again."
        )


async def _estimate_consultation_cost(
    session: ConsultationSession, query: str, max_turns: int, context: str = None
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
    context: str = None,
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
