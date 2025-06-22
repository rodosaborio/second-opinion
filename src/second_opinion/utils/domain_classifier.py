"""
Domain classification utility using LLM-based intelligent classification.

This module provides smart domain classification for consultation queries using
small, cost-effective language models instead of brittle string matching.
"""

import json
import logging
from functools import lru_cache

from ..clients import detect_model_provider
from ..core.models import Message, ModelRequest
from ..utils.client_factory import create_client_from_config

logger = logging.getLogger(__name__)

# Cache for repeated domain classifications
_classification_cache: dict[str, tuple[str, float]] = {}


class DomainClassifier:
    """Intelligent domain classification using small language models."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        """
        Initialize domain classifier.

        Args:
            model: Classification model to use (default: cost-effective gpt-4o-mini)
        """
        self.model = model
        self.classification_prompt = self._get_classification_prompt()

    def _get_classification_prompt(self) -> str:
        """Get multi-shot classification prompt with clear examples."""
        return """You are a domain classifier for AI consultation queries. Classify each query into exactly one of these domains:

DOMAINS:
- coding: Programming, software development, debugging, algorithms, APIs
- performance: System optimization, scalability, infrastructure, databases
- creative: Writing, content creation, marketing, storytelling, design
- general: Everything else (knowledge questions, advice, explanations)

EXAMPLES:

Query: "Write a Python function to sort a list"
Domain: coding

Query: "Help me debug this JavaScript error"
Domain: coding

Query: "How can I optimize database query performance?"
Domain: performance

Query: "Design a scalable microservices architecture"
Domain: performance

Query: "Write a creative story about AI"
Domain: creative

Query: "Help me with marketing copy for my product"
Domain: creative

Query: "What is the capital of France?"
Domain: general

Query: "Explain quantum computing concepts"
Domain: general

Query: "Should I use async/await or threading?"
Domain: coding

Query: "How to improve application response time?"
Domain: performance

OUTPUT FORMAT:
Respond with ONLY a JSON object in this exact format:
{"domain": "domain_name", "confidence": 0.95}

Where confidence is a number between 0.0 and 1.0 indicating classification certainty.

Query: """

    async def classify_domain(
        self, query: str, context: str | None = None
    ) -> tuple[str, float]:
        """
        Classify domain for a consultation query.

        Args:
            query: The consultation query to classify
            context: Optional additional context

        Returns:
            Tuple of (domain_name, confidence_score)
        """
        # Create cache key
        cache_key = f"{query}|{context or ''}"

        # Check cache first
        if cache_key in _classification_cache:
            logger.debug(f"Domain classification cache hit for query: {query[:50]}...")
            return _classification_cache[cache_key]

        try:
            # Prepare classification input
            classification_input = query
            if context:
                classification_input = f"{query}\n\nContext: {context}"

            # Build classification request
            full_prompt = f"{self.classification_prompt}{classification_input}"

            # Get provider and client
            provider = detect_model_provider(self.model)
            client = create_client_from_config(provider)

            # Create model request
            request = ModelRequest(
                model=self.model,
                messages=[Message(role="user", content=full_prompt)],
                max_tokens=50,  # Small response needed
                temperature=0.1,  # Low temperature for consistency
                system_prompt="You are a precise domain classifier. Always respond with valid JSON.",
            )

            # Get classification
            response = await client.complete(request)

            # Parse JSON response
            try:
                result = json.loads(response.content.strip())
                domain = result.get("domain", "general")
                confidence = float(result.get("confidence", 0.5))

                # Validate domain
                valid_domains = {"coding", "performance", "creative", "general"}
                if domain not in valid_domains:
                    logger.warning(
                        f"Invalid domain '{domain}' from classifier, using 'general'"
                    )
                    domain = "general"
                    confidence = 0.5

                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))

                # Cache the result
                _classification_cache[cache_key] = (domain, confidence)

                logger.debug(
                    f"Classified '{query[:50]}...' as '{domain}' (confidence: {confidence:.2f})"
                )
                return domain, confidence

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse classification response: {e}")
                return "general", 0.5

        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            return "general", 0.5

    async def get_domain_with_fallback(
        self, query: str, context: str | None = None, confidence_threshold: float = 0.7
    ) -> str:
        """
        Get domain classification with confidence-based fallback.

        Args:
            query: Query to classify
            context: Optional context
            confidence_threshold: Minimum confidence for non-general classification

        Returns:
            Domain name (falls back to 'general' for low confidence)
        """
        domain, confidence = await self.classify_domain(query, context)

        # If confidence is low and domain isn't general, fall back to general
        if confidence < confidence_threshold and domain != "general":
            logger.info(
                f"Low confidence ({confidence:.2f}) for '{domain}', using 'general'"
            )
            return "general"

        return domain


# Global classifier instance
_classifier: DomainClassifier | None = None


def get_domain_classifier() -> DomainClassifier:
    """Get global domain classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = DomainClassifier()
    assert _classifier is not None  # Type checker hint
    return _classifier


async def classify_consultation_domain(query: str, context: str | None = None) -> str:
    """
    Convenience function for domain classification.

    Args:
        query: Query to classify
        context: Optional context

    Returns:
        Domain name (coding, performance, creative, general)
    """
    classifier = get_domain_classifier()
    return await classifier.get_domain_with_fallback(query, context)


def clear_classification_cache() -> None:
    """Clear the domain classification cache (useful for testing)."""
    global _classification_cache
    _classification_cache.clear()
    logger.debug("Domain classification cache cleared")


@lru_cache(maxsize=128)
def get_domain_specialized_model(domain: str) -> str | None:
    """
    Get specialized model recommendations for specific domains.

    Args:
        domain: Domain name

    Returns:
        Recommended model for the domain, or None for default routing
    """
    # Future: Add domain-specific model recommendations
    # For now, let the standard model router handle this
    return None
