"""
ConversationStore for managing conversation storage and retrieval.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, create_engine, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..config.settings import get_settings
from ..core.models import ModelResponse
from .encryption import get_encryption_manager
from .models import Base, Conversation, Response


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects and Decimal."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class ConversationStore:
    """Manages conversation storage with encryption and analytics."""

    def __init__(self, database_url: str | None = None):
        self.settings = get_settings()
        self.encryption_manager = get_encryption_manager()

        # Use provided URL or construct from settings
        if database_url:
            self.database_url = database_url
        else:
            # Default to SQLite in user data directory
            data_dir = Path.home() / ".second-opinion" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "conversations.db"
            self.database_url = f"sqlite:///{db_path}"

        # Create engines
        self.engine = create_engine(self.database_url, echo=False)
        self.async_engine = create_async_engine(
            self.database_url.replace("sqlite://", "sqlite+aiosqlite://"), echo=False
        )

        # Create session makers
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.AsyncSessionLocal = async_sessionmaker(self.async_engine)

        # Initialize database
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    async def store_conversation(
        self,
        user_prompt: str,
        primary_response: ModelResponse,
        comparison_responses: list[ModelResponse],
        evaluation_result: dict[str, Any] | None = None,
        interface_type: str = "cli",
        session_id: str | None = None,
        tool_name: str | None = None,
        context: str | None = None,
        task_complexity: str | None = None,
        domain_classification: str | None = None,
    ) -> str:
        """
        Store a complete conversation with all responses.

        Returns:
            The conversation ID
        """
        async with self.AsyncSessionLocal() as session:
            # Encrypt user prompt and context
            encrypted_prompt, key_id = self.encryption_manager.encrypt(user_prompt)
            encrypted_context = None
            if context:
                encrypted_context, _ = self.encryption_manager.encrypt(context)

            # Calculate total cost
            total_cost = primary_response.cost_estimate
            for resp in comparison_responses:
                total_cost += resp.cost_estimate

            # Determine if content is sensitive (basic heuristic)
            is_sensitive = self._detect_sensitive_content(
                user_prompt, primary_response.content
            )

            # Create conversation record
            conversation = Conversation(
                id=str(uuid4()),
                session_id=session_id,
                interface_type=interface_type,
                tool_name=tool_name,
                user_prompt_encrypted=encrypted_prompt,
                context_encrypted=encrypted_context,
                total_cost=total_cost,
                estimated_cost=total_cost,  # For now, same as actual
                task_complexity=task_complexity,
                domain_classification=domain_classification,
                encryption_key_id=key_id,
                is_sensitive=is_sensitive,
            )

            session.add(conversation)
            await session.flush()  # Get the conversation ID

            # Store primary response
            await self._store_response(
                session,
                conversation.id,
                primary_response,
                "primary",
                0,
                evaluation_result,
            )

            # Store comparison responses
            for i, response in enumerate(comparison_responses):
                await self._store_response(
                    session,
                    conversation.id,
                    response,
                    "comparison",
                    i + 1,
                    evaluation_result,
                )

            await session.commit()
            return str(conversation.id)

    async def _store_response(
        self,
        session: AsyncSession,
        conversation_id: str,
        model_response: ModelResponse,
        response_type: str,
        order: int,
        evaluation_result: dict[str, Any] | None = None,
    ) -> None:
        """Store a single model response."""
        # Encrypt response content and metadata
        encrypted_content, key_id = self.encryption_manager.encrypt(
            model_response.content
        )

        encrypted_metadata = None
        if model_response.metadata:
            metadata_json = json.dumps(model_response.metadata, cls=DateTimeEncoder)
            encrypted_metadata, _ = self.encryption_manager.encrypt(metadata_json)

        # Extract evaluation data if available
        evaluation_score = None
        encrypted_reasoning = None
        if evaluation_result and response_type == "primary":
            if "overall_score" in evaluation_result:
                evaluation_score = Decimal(str(evaluation_result["overall_score"]))
            if "reasoning" in evaluation_result:
                encrypted_reasoning, _ = self.encryption_manager.encrypt(
                    evaluation_result["reasoning"]
                )

        # Extract token usage
        usage = model_response.usage
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        response = Response(
            conversation_id=conversation_id,
            model=model_response.model,
            provider=model_response.provider,
            response_type=response_type,
            response_order=order,
            content_encrypted=encrypted_content,
            metadata_encrypted=encrypted_metadata,
            cost=model_response.cost_estimate,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            evaluation_score=evaluation_score,
            evaluation_reasoning_encrypted=encrypted_reasoning,
            encryption_key_id=key_id,
        )

        session.add(response)

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation with all responses."""
        async with self.AsyncSessionLocal() as session:
            # Get conversation
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                return None

            # Get responses
            result = await session.execute(
                select(Response)
                .where(Response.conversation_id == conversation_id)
                .order_by(Response.response_type, Response.response_order)
            )
            responses = result.scalars().all()

            # Decrypt and structure data
            decrypted_prompt = self.encryption_manager.decrypt(
                conversation.user_prompt_encrypted, conversation.encryption_key_id
            )

            decrypted_context = None
            if conversation.context_encrypted:
                decrypted_context = self.encryption_manager.decrypt(
                    conversation.context_encrypted, conversation.encryption_key_id
                )

            response_data = []
            for response in responses:
                decrypted_content = self.encryption_manager.decrypt(
                    response.content_encrypted, response.encryption_key_id
                )

                decrypted_metadata = None
                if response.metadata_encrypted:
                    metadata_json = self.encryption_manager.decrypt(
                        response.metadata_encrypted, response.encryption_key_id
                    )
                    decrypted_metadata = json.loads(metadata_json)

                decrypted_reasoning = None
                if response.evaluation_reasoning_encrypted:
                    decrypted_reasoning = self.encryption_manager.decrypt(
                        response.evaluation_reasoning_encrypted,
                        response.encryption_key_id,
                    )

                response_data.append(
                    {
                        "id": response.id,
                        "model": response.model,
                        "provider": response.provider,
                        "response_type": response.response_type,
                        "content": decrypted_content,
                        "metadata": decrypted_metadata,
                        "cost": response.cost,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.total_tokens,
                        "evaluation_score": response.evaluation_score,
                        "evaluation_reasoning": decrypted_reasoning,
                        "created_at": response.created_at,
                    }
                )

            return {
                "id": conversation.id,
                "session_id": conversation.session_id,
                "interface_type": conversation.interface_type,
                "tool_name": conversation.tool_name,
                "user_prompt": decrypted_prompt,
                "context": decrypted_context,
                "total_cost": conversation.total_cost,
                "task_complexity": conversation.task_complexity,
                "domain_classification": conversation.domain_classification,
                "is_sensitive": conversation.is_sensitive,
                "created_at": conversation.created_at,
                "responses": response_data,
            }

    async def get_conversations_by_session(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Get all conversations for a session."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.desc())
            )
            conversations = result.scalars().all()

            return [
                {
                    "id": conv.id,
                    "user_prompt": self.encryption_manager.decrypt(
                        conv.user_prompt_encrypted, conv.encryption_key_id
                    )[:100]
                    + "...",  # Truncated for overview
                    "total_cost": conv.total_cost,
                    "created_at": conv.created_at,
                    "tool_name": conv.tool_name,
                }
                for conv in conversations
            ]

    async def get_usage_analytics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        interface_type: str | None = None,
    ) -> dict[str, Any]:
        """Get usage analytics for the specified period."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        async with self.AsyncSessionLocal() as session:
            # Build filter conditions
            where_clause = and_(
                Conversation.created_at >= start_date,
                Conversation.created_at <= end_date,
            )
            if interface_type:
                where_clause = and_(
                    where_clause, Conversation.interface_type == interface_type
                )
            conv_result = await session.execute(
                select(
                    func.count(Conversation.id).label("total_conversations"),
                    func.sum(Conversation.total_cost).label("total_cost"),
                    func.avg(Conversation.total_cost).label("avg_cost"),
                ).where(where_clause)
            )
            conv_stats = conv_result.first()

            # Get model usage stats
            model_result = await session.execute(
                select(
                    Response.model,
                    func.count(Response.id).label("usage_count"),
                    func.sum(Response.cost).label("total_cost"),
                )
                .join(Conversation)
                .where(where_clause)
                .group_by(Response.model)
                .order_by(func.count(Response.id).desc())
            )
            model_stats = model_result.all()

            # Get interface breakdown
            interface_result = await session.execute(
                select(
                    Conversation.interface_type,
                    func.count(Conversation.id).label("count"),
                    func.sum(Conversation.total_cost).label("cost"),
                )
                .where(where_clause)
                .group_by(Conversation.interface_type)
            )
            interface_stats = interface_result.all()

            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "summary": {
                    "total_conversations": conv_stats.total_conversations or 0,
                    "total_cost": conv_stats.total_cost or Decimal("0"),
                    "average_cost_per_conversation": conv_stats.avg_cost
                    or Decimal("0"),
                },
                "model_usage": [
                    {
                        "model": stat.model,
                        "usage_count": stat.usage_count,
                        "total_cost": stat.total_cost,
                    }
                    for stat in model_stats
                ],
                "interface_breakdown": [
                    {
                        "interface": stat.interface_type,
                        "conversation_count": stat.count,
                        "total_cost": stat.cost,
                    }
                    for stat in interface_stats
                ],
            }

    def _detect_sensitive_content(self, prompt: str, response: str) -> bool:
        """Basic heuristic to detect potentially sensitive content."""
        sensitive_indicators = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "personal",
            "private",
            "confidential",
            "ssn",
            "social security",
            "credit card",
            "bank account",
            "api key",
            "access token",
        ]

        combined_text = (prompt + " " + response).lower()
        return any(indicator in combined_text for indicator in sensitive_indicators)

    def store_conversation_sync(
        self,
        user_prompt: str,
        primary_response: ModelResponse,
        comparison_responses: list[ModelResponse],
        evaluation_result: dict[str, Any] | None = None,
        interface_type: str = "cli",
        session_id: str | None = None,
        tool_name: str = "second_opinion",
        context: str | None = None,
        task_complexity: str | None = None,
        domain_classification: str | None = None,
    ) -> str:
        """
        Store a complete conversation with all responses (synchronous version for CLI).

        Returns:
            The conversation ID
        """
        with self.SessionLocal() as session:
            # Encrypt user prompt and context
            encrypted_prompt, key_id = self.encryption_manager.encrypt(user_prompt)
            encrypted_context = None
            if context:
                encrypted_context, _ = self.encryption_manager.encrypt(context)

            # Calculate total cost
            total_cost = primary_response.cost_estimate
            for resp in comparison_responses:
                total_cost += resp.cost_estimate

            # Determine if content is sensitive (basic heuristic)
            is_sensitive = self._detect_sensitive_content(
                user_prompt, primary_response.content
            )

            # Create conversation record
            conversation = Conversation(
                id=str(uuid4()),
                session_id=session_id,
                interface_type=interface_type,
                tool_name=tool_name,
                user_prompt_encrypted=encrypted_prompt,
                context_encrypted=encrypted_context,
                total_cost=total_cost,
                estimated_cost=total_cost,  # For now, same as actual
                task_complexity=task_complexity,
                domain_classification=domain_classification,
                encryption_key_id=key_id,
                is_sensitive=is_sensitive,
            )

            session.add(conversation)
            session.flush()  # Get the conversation ID

            # Store primary response
            self._store_response_sync(
                session,
                conversation.id,
                primary_response,
                "primary",
                0,
                evaluation_result,
            )

            # Store comparison responses
            for i, response in enumerate(comparison_responses):
                self._store_response_sync(
                    session,
                    conversation.id,
                    response,
                    "comparison",
                    i + 1,
                    evaluation_result,
                )

            session.commit()
            return str(conversation.id)

    def _store_response_sync(
        self,
        session,
        conversation_id: str,
        model_response: ModelResponse,
        response_type: str,
        order: int,
        evaluation_result: dict[str, Any] | None = None,
    ) -> None:
        """Store a single model response (synchronous version)."""
        # Encrypt response content and metadata
        encrypted_content, key_id = self.encryption_manager.encrypt(
            model_response.content
        )

        encrypted_metadata = None
        if model_response.metadata:
            metadata_json = json.dumps(model_response.metadata, cls=DateTimeEncoder)
            encrypted_metadata, _ = self.encryption_manager.encrypt(metadata_json)

        # Extract evaluation data if available
        evaluation_score = None
        encrypted_reasoning = None
        if evaluation_result and response_type == "primary":
            if "overall_score" in evaluation_result:
                evaluation_score = Decimal(str(evaluation_result["overall_score"]))
            if "reasoning" in evaluation_result:
                encrypted_reasoning, _ = self.encryption_manager.encrypt(
                    evaluation_result["reasoning"]
                )

        # Extract token usage
        usage = model_response.usage
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        response = Response(
            conversation_id=conversation_id,
            model=model_response.model,
            provider=model_response.provider,
            response_type=response_type,
            response_order=order,
            content_encrypted=encrypted_content,
            metadata_encrypted=encrypted_metadata,
            cost=model_response.cost_estimate,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            evaluation_score=evaluation_score,
            evaluation_reasoning_encrypted=encrypted_reasoning,
            encryption_key_id=key_id,
        )

        session.add(response)

    async def close(self) -> None:
        """Close database connections."""
        await self.async_engine.dispose()
        self.engine.dispose()


# Global store instance
_conversation_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    """Get the global conversation store instance."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    # Type: ignore since we just ensured it's not None
    return _conversation_store  # type: ignore[return-value]
