"""
Tests for the domain classifier utility.

This module tests the LLM-based domain classification functionality.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from second_opinion.core.models import ModelResponse, TokenUsage
from second_opinion.utils.domain_classifier import (
    DomainClassifier,
    _classification_cache,
    classify_consultation_domain,
    clear_classification_cache,
    get_domain_classifier,
    get_domain_specialized_model,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear classification cache between tests."""
    clear_classification_cache()
    yield
    clear_classification_cache()


@pytest.fixture
def mock_domain_classifier_dependencies(monkeypatch):
    """Mock external dependencies for domain classifier tests."""

    # Mock client factory
    mock_client = MagicMock()
    mock_client.complete = AsyncMock()

    def create_mock_client(provider):
        return mock_client

    monkeypatch.setattr(
        "second_opinion.utils.domain_classifier.create_client_from_config",
        create_mock_client,
    )

    # Mock provider detection
    def mock_detect_provider(model):
        return "openrouter"

    monkeypatch.setattr(
        "second_opinion.utils.domain_classifier.detect_model_provider",
        mock_detect_provider,
    )

    return {"client": mock_client}


class TestDomainClassifier:
    """Test DomainClassifier functionality."""

    def test_classifier_initialization(self):
        """Test creating a domain classifier."""
        classifier = DomainClassifier()
        assert classifier.model == "openai/gpt-4o-mini"
        assert "domain classifier" in classifier.classification_prompt.lower()

    def test_custom_model_initialization(self):
        """Test creating classifier with custom model."""
        classifier = DomainClassifier("anthropic/claude-3-haiku")
        assert classifier.model == "anthropic/claude-3-haiku"

    @pytest.mark.asyncio
    async def test_successful_classification(self, mock_domain_classifier_dependencies):
        """Test successful domain classification."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "coding", "confidence": 0.95}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain("Write a Python function")

        assert domain == "coding"
        assert confidence == 0.95
        assert mock_client.complete.called

    @pytest.mark.asyncio
    async def test_classification_with_context(
        self, mock_domain_classifier_dependencies
    ):
        """Test classification with additional context."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "performance", "confidence": 0.88}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=120, output_tokens=20, total_tokens=140),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain(
            "Optimize this system", context="database performance"
        )

        assert domain == "performance"
        assert confidence == 0.88

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_domain_classifier_dependencies):
        """Test handling of invalid JSON response."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content="Invalid JSON response",
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain("Test query")

        assert domain == "general"
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_invalid_domain_fallback(self, mock_domain_classifier_dependencies):
        """Test fallback for invalid domain classification."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "invalid_domain", "confidence": 0.90}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain("Test query")

        assert domain == "general"
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_confidence_bounds(self, mock_domain_classifier_dependencies):
        """Test confidence score bounds handling."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "coding", "confidence": 1.5}',  # Out of bounds
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain("Test query")

        assert domain == "coding"
        assert confidence == 1.0  # Should be clamped to 1.0

    @pytest.mark.asyncio
    async def test_classification_caching(self, mock_domain_classifier_dependencies):
        """Test that classifications are cached."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "coding", "confidence": 0.95}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()

        # First call should make API request
        domain1, confidence1 = await classifier.classify_domain("Write Python code")
        assert mock_client.complete.call_count == 1

        # Second call with same query should use cache
        domain2, confidence2 = await classifier.classify_domain("Write Python code")
        assert mock_client.complete.call_count == 1  # No additional call

        assert domain1 == domain2 == "coding"
        assert confidence1 == confidence2 == 0.95

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_domain_classifier_dependencies):
        """Test handling of exceptions during classification."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.side_effect = Exception("API Error")

        classifier = DomainClassifier()
        domain, confidence = await classifier.classify_domain("Test query")

        assert domain == "general"
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_confidence_threshold_fallback(
        self, mock_domain_classifier_dependencies
    ):
        """Test confidence threshold fallback to general."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "coding", "confidence": 0.3}',  # Low confidence
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain = await classifier.get_domain_with_fallback(
            "Ambiguous query", confidence_threshold=0.7
        )

        assert domain == "general"

    @pytest.mark.asyncio
    async def test_high_confidence_classification(
        self, mock_domain_classifier_dependencies
    ):
        """Test high confidence classification passes threshold."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "coding", "confidence": 0.95}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        classifier = DomainClassifier()
        domain = await classifier.get_domain_with_fallback(
            "Write a Python function", confidence_threshold=0.7
        )

        assert domain == "coding"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_domain_classifier_singleton(self):
        """Test that get_domain_classifier returns the same instance."""
        classifier1 = get_domain_classifier()
        classifier2 = get_domain_classifier()
        assert classifier1 is classifier2

    @pytest.mark.asyncio
    async def test_classify_consultation_domain_convenience(
        self, mock_domain_classifier_dependencies
    ):
        """Test the convenience function for domain classification."""
        mock_client = mock_domain_classifier_dependencies["client"]
        mock_client.complete.return_value = ModelResponse(
            content='{"domain": "creative", "confidence": 0.85}',
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        domain = await classify_consultation_domain("Write a story")
        assert domain == "creative"

    def test_clear_classification_cache(self):
        """Test cache clearing functionality."""
        # Add something to cache
        _classification_cache["test"] = ("coding", 0.9)
        assert len(_classification_cache) == 1

        clear_classification_cache()
        assert len(_classification_cache) == 0

    def test_get_domain_specialized_model(self):
        """Test domain-specific model recommendations."""
        # Currently returns None for all domains (future enhancement)
        assert get_domain_specialized_model("coding") is None
        assert get_domain_specialized_model("performance") is None
        assert get_domain_specialized_model("creative") is None
        assert get_domain_specialized_model("general") is None


@pytest.mark.asyncio
async def test_integration_classification_workflow(mock_domain_classifier_dependencies):
    """Test complete classification workflow integration."""
    mock_client = mock_domain_classifier_dependencies["client"]

    # Test different domain classifications
    test_cases = [
        ("Write Python code", '{"domain": "coding", "confidence": 0.95}', "coding"),
        (
            "Optimize database",
            '{"domain": "performance", "confidence": 0.88}',
            "performance",
        ),
        ("Creative writing", '{"domain": "creative", "confidence": 0.92}', "creative"),
        ("General question", '{"domain": "general", "confidence": 0.80}', "general"),
    ]

    for query, mock_response, expected_domain in test_cases:
        mock_client.complete.return_value = ModelResponse(
            content=mock_response,
            model="openai/gpt-4o-mini",
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            cost_estimate=Decimal("0.001"),
            provider="openrouter",
        )

        domain = await classify_consultation_domain(query)
        assert domain == expected_domain
