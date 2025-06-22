"""
Tests for database models and basic functionality.
"""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from second_opinion.database.encryption import EncryptionManager
from second_opinion.database.models import Conversation, Response
from second_opinion.database.store import ConversationStore


class TestDatabaseModels:
    """Test database model creation and basic functionality."""

    def test_conversation_model_creation(self):
        """Test that Conversation model can be instantiated."""
        conversation = Conversation(
            id="test-id",
            interface_type="cli",
            user_prompt_encrypted="encrypted_prompt",
            encryption_key_id="test-key",
            total_cost=Decimal("0.05"),
        )

        assert conversation.id == "test-id"
        assert conversation.interface_type == "cli"
        assert conversation.total_cost == Decimal("0.05")

    def test_response_model_creation(self):
        """Test that Response model can be instantiated."""
        response = Response(
            id="response-id",
            conversation_id="conv-id",
            model="openai/gpt-4o-mini",
            provider="openrouter",
            response_type="primary",
            content_encrypted="encrypted_content",
            encryption_key_id="test-key",
            cost=Decimal("0.02"),
        )

        assert response.id == "response-id"
        assert response.model == "openai/gpt-4o-mini"
        assert response.cost == Decimal("0.02")


class TestEncryptionManager:
    """Test encryption functionality."""

    def test_encryption_manager_creation(self):
        """Test that EncryptionManager can be created."""
        manager = EncryptionManager()
        assert manager.get_current_key_id() is not None

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly."""
        manager = EncryptionManager()

        original_text = "This is a test message for encryption"
        encrypted_data, key_id = manager.encrypt(original_text)
        decrypted_text = manager.decrypt(encrypted_data, key_id)

        assert decrypted_text == original_text
        assert encrypted_data != original_text  # Should be encrypted


class TestConversationStore:
    """Test conversation store functionality."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary conversation store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            store = ConversationStore(database_url=db_url)
            yield store
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)

    def test_store_creation(self, temp_store):
        """Test that ConversationStore can be created."""
        assert temp_store.database_url.startswith("sqlite://")
        assert hasattr(temp_store, "engine")
        assert hasattr(temp_store, "async_engine")

    def test_database_initialization(self, temp_store):
        """Test that database tables are created."""
        # The tables should be created during store initialization
        # This is verified by the fact that the store creates without error
        assert temp_store.engine is not None
