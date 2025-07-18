"""
Dynamic pricing manager for model costs using LiteLLM pricing data.

This module provides real-time pricing information for AI models by fetching
and caching pricing data from LiteLLM's comprehensive model database.
"""

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# LiteLLM pricing data source
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Default cache TTL (1 hour)
DEFAULT_CACHE_TTL_HOURS = 1

# Request timeout for pricing data fetch
PRICING_FETCH_TIMEOUT = 30.0


class ModelPricingInfo(BaseModel):
    """Pricing information for a specific model."""

    model_name: str = Field(..., description="Model identifier")
    input_cost_per_1k_tokens: Decimal = Field(
        ..., description="Cost per 1000 input tokens"
    )
    output_cost_per_1k_tokens: Decimal = Field(
        ..., description="Cost per 1000 output tokens"
    )
    max_tokens: int | None = Field(None, description="Maximum token limit")
    provider: str | None = Field(None, description="Model provider")
    mode: str | None = Field(None, description="Model mode (chat, embedding, etc.)")
    supports_function_calling: bool | None = Field(
        None, description="Function calling support"
    )


class PricingCache(BaseModel):
    """Cached pricing data with timestamp."""

    data: dict[str, ModelPricingInfo] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(
        default="unknown", description="Data source (network, backup, etc.)"
    )

    def is_expired(self, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS) -> bool:
        """Check if cache has expired."""
        expiry_time = self.last_updated + timedelta(hours=ttl_hours)
        return datetime.now(UTC) > expiry_time


class PricingManager:
    """
    Manages dynamic pricing data for AI models.

    Features:
    - Fetches latest pricing from LiteLLM repository
    - Local caching with TTL expiration
    - Fallback to bundled backup data
    - Thread-safe operations
    - Conservative cost estimates for unknown models
    """

    def __init__(
        self,
        cache_file: Path | None = None,
        backup_file: Path | None = None,
        cache_ttl_hours: int | None = None,
        fetch_timeout: float | None = None,
    ):
        """
        Initialize pricing manager.

        Args:
            cache_file: Path to cache file (default: data/pricing_cache.json)
            backup_file: Path to backup pricing file (default: data/pricing_backup.json)
            cache_ttl_hours: Cache time-to-live in hours (default: from settings)
            fetch_timeout: HTTP request timeout in seconds (default: from settings)
        """
        settings = get_settings()
        pricing_config = settings.pricing

        # Set default paths
        data_dir = Path(settings.data_dir)
        self.cache_file = cache_file or data_dir / "pricing_cache.json"

        # Use custom backup path if provided in config
        if backup_file:
            self.backup_file = backup_file
        elif pricing_config.backup_file_path:
            self.backup_file = Path(pricing_config.backup_file_path)
        else:
            self.backup_file = data_dir / "pricing_backup.json"

        # Use config values or defaults
        self.cache_ttl_hours = cache_ttl_hours or pricing_config.cache_ttl_hours
        self.fetch_timeout = fetch_timeout or pricing_config.fetch_timeout
        self.auto_update_enabled = pricing_config.auto_update_on_startup
        self.pricing_enabled = pricing_config.enabled

        # Thread-safe cache
        self._cache: PricingCache | None = None
        self._lock = threading.RLock()

        # Ensure data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)

        # Load initial cache
        self._load_cache()

        # Schedule automatic update if enabled
        if self.auto_update_enabled and self.pricing_enabled:
            import asyncio

            try:
                # Try to get current event loop
                loop = asyncio.get_running_loop()
                # Try to update pricing in background (non-blocking)
                loop.create_task(self._background_update())
            except RuntimeError:
                # No event loop running, skip auto-update
                logger.debug("No event loop available for auto-update, skipping")

    async def _background_update(self) -> None:
        """Background task to update pricing data."""
        try:
            success = await self.fetch_latest_pricing()
            if success:
                logger.info("Successfully updated pricing data on startup")
            else:
                logger.warning(
                    "Failed to update pricing data on startup, using cached/backup data"
                )
        except Exception as e:
            logger.warning(f"Error during background pricing update: {e}")

    def _load_cache(self) -> None:
        """Load pricing cache from disk."""
        with self._lock:
            # Try to load from cache file first
            if self.cache_file.exists():
                try:
                    with open(self.cache_file) as f:
                        cache_data = json.load(f)

                    # Convert to PricingCache object
                    pricing_data = {}
                    for model_name, raw_info in cache_data.get("data", {}).items():
                        pricing_data[model_name] = ModelPricingInfo(**raw_info)

                    self._cache = PricingCache(
                        data=pricing_data,
                        last_updated=datetime.fromisoformat(cache_data["last_updated"]),
                        source=cache_data.get("source", "cache"),
                    )

                    logger.info(
                        f"Loaded pricing cache with {len(pricing_data)} models from cache file"
                    )
                    return

                except Exception as e:
                    logger.warning(f"Failed to load pricing cache: {e}")

            # Fallback to backup file
            if self.backup_file.exists():
                try:
                    logger.info(
                        f"Loading pricing data from backup file: {self.backup_file}"
                    )
                    with open(self.backup_file) as f:
                        backup_data = json.load(f)

                    pricing_data = self._parse_litellm_data(backup_data)
                    self._cache = PricingCache(
                        data=pricing_data,
                        last_updated=datetime.now(UTC),
                        source="backup",
                    )

                    logger.info(
                        f"Successfully loaded backup pricing data with {len(pricing_data)} models"
                    )

                    # Log some sample models for debugging
                    sample_models = list(pricing_data.keys())[:5]
                    logger.debug(f"Sample models loaded: {sample_models}")

                    # Check for specific models the user is using
                    test_models = [
                        "anthropic/claude-sonnet-4",
                        "openrouter/anthropic/claude-sonnet-4",
                        "claude-sonnet-4",
                        "anthropic/claude-3-haiku",
                        "openrouter/anthropic/claude-3-haiku",
                    ]
                    found_models = [m for m in test_models if m in pricing_data]
                    logger.debug(f"Test models found in pricing data: {found_models}")

                    return

                except Exception as e:
                    logger.error(f"Failed to load backup pricing data: {e}")

            # Initialize empty cache
            self._cache = PricingCache(source="empty")
            logger.error("No pricing data available, using empty cache")

    def _save_cache(self) -> None:
        """Save current cache to disk."""
        if not self._cache:
            return

        try:
            # Convert to serializable format
            cache_data = {
                "data": {
                    model_name: info.model_dump()
                    for model_name, info in self._cache.data.items()
                },
                "last_updated": self._cache.last_updated.isoformat(),
                "source": "cache",  # Always mark as cache when saving to cache file
            }

            # Write to cache file
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

            logger.debug(f"Saved pricing cache with {len(self._cache.data)} models")

        except Exception as e:
            logger.warning(f"Failed to save pricing cache: {e}")

    def _parse_litellm_data(
        self, raw_data: dict[str, Any]
    ) -> dict[str, ModelPricingInfo]:
        """Parse raw LiteLLM pricing data into ModelPricingInfo objects."""
        pricing_data = {}

        for model_name, model_info in raw_data.items():
            try:
                # Extract pricing information
                input_cost = model_info.get("input_cost_per_token", 0)
                output_cost = model_info.get("output_cost_per_token", 0)

                # Convert to per-1k-token pricing
                input_cost_per_1k = Decimal(str(input_cost)) * 1000
                output_cost_per_1k = Decimal(str(output_cost)) * 1000

                # Extract other metadata
                max_tokens = model_info.get("max_tokens")
                provider = model_info.get("litellm_provider")
                mode = model_info.get("mode", "chat")
                supports_function_calling = model_info.get(
                    "supports_function_calling", False
                )

                pricing_info = ModelPricingInfo(
                    model_name=model_name,
                    input_cost_per_1k_tokens=input_cost_per_1k,
                    output_cost_per_1k_tokens=output_cost_per_1k,
                    max_tokens=max_tokens,
                    provider=provider,
                    mode=mode,
                    supports_function_calling=supports_function_calling,
                )

                pricing_data[model_name] = pricing_info

            except Exception as e:
                logger.debug(f"Failed to parse pricing for model {model_name}: {e}")
                continue

        return pricing_data

    async def fetch_latest_pricing(self, force: bool = False) -> bool:
        """
        Fetch latest pricing data from LiteLLM.

        Args:
            force: Force fetch even if cache is not expired

        Returns:
            True if fetch was successful, False otherwise
        """
        with self._lock:
            # Check if fetch is needed
            if (
                not force
                and self._cache
                and not self._cache.is_expired(self.cache_ttl_hours)
            ):
                logger.debug("Pricing cache is still fresh, skipping fetch")
                return True

        try:
            logger.info("Fetching latest pricing data from LiteLLM...")

            async with httpx.AsyncClient(timeout=self.fetch_timeout) as client:
                response = await client.get(LITELLM_PRICING_URL)
                response.raise_for_status()

                raw_data = response.json()
                pricing_data = self._parse_litellm_data(raw_data)

                with self._lock:
                    self._cache = PricingCache(
                        data=pricing_data,
                        last_updated=datetime.now(UTC),
                        source="network",
                    )

                    # Save to cache file
                    self._save_cache()

                logger.info(
                    f"Successfully fetched pricing for {len(pricing_data)} models"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to fetch latest pricing data: {e}")
            return False

    def get_model_pricing(self, model_name: str) -> ModelPricingInfo | None:
        """
        Get pricing information for a specific model.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "claude-3-sonnet")

        Returns:
            ModelPricingInfo if found, None otherwise
        """
        with self._lock:
            if not self._cache:
                logger.warning(f"No pricing cache available for {model_name}")
                return None

            logger.debug(f"Looking up pricing for: {model_name}")

            # Direct lookup first
            if model_name in self._cache.data:
                logger.debug(f"Found direct match for {model_name}")
                return self._cache.data[model_name]

            # Generate comprehensive variations to try
            variations_to_try = self._generate_model_variations(model_name)

            # Try each variation
            for variation in variations_to_try:
                if variation in self._cache.data:
                    logger.debug(
                        f"Found pricing for {model_name} using variation: {variation}"
                    )
                    return self._cache.data[variation]

            # Fallback: fuzzy matching for partial matches
            best_match = self._fuzzy_match_model(model_name)
            if best_match:
                logger.debug(
                    f"Found pricing for {model_name} using fuzzy match: {best_match}"
                )
                return self._cache.data[best_match]

            logger.debug(
                f"No pricing found for {model_name} (tried {len(variations_to_try)} variations)"
            )
            return None

    def _generate_model_variations(self, model_name: str) -> list[str]:
        """Generate common model name variations for lookup."""
        variations = [model_name]  # Always try original first

        # For cloud models, try OpenRouter prefixed versions first
        if "/" in model_name and not model_name.startswith("openrouter/"):
            variations.append(f"openrouter/{model_name}")

        # Try without provider prefix
        if "/" in model_name:
            base_name = model_name.split("/")[-1]
            variations.append(base_name)
            # Also try openrouter prefix with base name
            variations.append(f"openrouter/{base_name}")

        # Try with anthropic/ prefix for anthropic models
        if "claude" in model_name.lower() and not model_name.startswith("anthropic/"):
            if "/" in model_name:
                base_name = model_name.split("/")[-1]
                variations.append(f"anthropic/{base_name}")
                variations.append(f"openrouter/anthropic/{base_name}")
            else:
                variations.append(f"anthropic/{model_name}")
                variations.append(f"openrouter/anthropic/{model_name}")

        # Try common OpenAI variations
        if any(
            term in model_name.lower() for term in ["gpt", "o1", "o3"]
        ) and not model_name.startswith("openai/"):
            if "/" in model_name:
                base_name = model_name.split("/")[-1]
                variations.append(f"openai/{base_name}")
                variations.append(f"openrouter/openai/{base_name}")
            else:
                variations.append(f"openai/{model_name}")
                variations.append(f"openrouter/openai/{model_name}")

        # Try Google variations
        if "gemini" in model_name.lower() and not model_name.startswith("google/"):
            if "/" in model_name:
                base_name = model_name.split("/")[-1]
                variations.append(f"google/{base_name}")
                variations.append(f"openrouter/google/{base_name}")
            else:
                variations.append(f"google/{model_name}")
                variations.append(f"openrouter/google/{model_name}")

        # Try basic underscore/dash variations
        if "_" in model_name:
            dash_version = model_name.replace("_", "-")
            variations.append(dash_version)
        if "-" in model_name:
            underscore_version = model_name.replace("-", "_")
            variations.append(underscore_version)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)

        return unique_variations

    def _fuzzy_match_model(self, model_name: str) -> str | None:
        """Find the best fuzzy match for a model name."""
        model_lower = model_name.lower()
        best_match = None
        best_score = 0

        # Extract key terms from the target model name
        key_terms = []
        if "claude" in model_lower:
            key_terms.append("claude")
            if "sonnet" in model_lower:
                key_terms.append("sonnet")
            if "haiku" in model_lower:
                key_terms.append("haiku")
            if "opus" in model_lower:
                key_terms.append("opus")
        elif "gpt" in model_lower:
            key_terms.append("gpt")
        elif "gemini" in model_lower:
            key_terms.append("gemini")

        if not key_terms:
            return None

        # Score each cached model
        if not self._cache:
            return None
        for cached_name in self._cache.data.keys():
            cached_lower = cached_name.lower()
            score = 0

            # Score based on key terms present
            for term in key_terms:
                if term in cached_lower:
                    score += 10

            # Prefer openrouter versions for cloud models
            if "openrouter" in cached_lower and "/" in model_name:
                score += 5

            # Prefer exact provider matches
            if "/" in model_name and "/" in cached_name:
                requested_provider = model_name.split("/")[0].lower()
                if requested_provider in cached_lower:
                    score += 3

            if (
                score > best_score and score >= 10
            ):  # Require at least one key term match
                best_score = score
                best_match = cached_name

        return best_match

    def estimate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int = 0
    ) -> tuple[Decimal, str]:
        """
        Estimate cost for a model request.

        Args:
            model_name: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Tuple of (estimated_cost, source) where source indicates
            how the pricing was determined
        """
        pricing_info = self.get_model_pricing(model_name)

        if pricing_info:
            input_cost = (
                Decimal(input_tokens) * pricing_info.input_cost_per_1k_tokens
            ) / 1000
            output_cost = (
                Decimal(output_tokens) * pricing_info.output_cost_per_1k_tokens
            ) / 1000
            total_cost = input_cost + output_cost

            return (
                total_cost,
                f"pricing_data_{self._cache.source if self._cache else 'unknown'}",
            )

        # Conservative fallback estimate
        logger.warning(
            f"No pricing data for model {model_name}, using conservative estimate"
        )

        # Use conservative estimates based on model tier
        if any(tier in model_name.lower() for tier in ["gpt-4", "claude-3", "opus"]):
            # High-tier models
            fallback_cost = Decimal("0.15")
        elif any(
            tier in model_name.lower() for tier in ["gpt-3.5", "claude-2", "sonnet"]
        ):
            # Mid-tier models
            fallback_cost = Decimal("0.05")
        else:
            # Low-tier models
            fallback_cost = Decimal("0.02")

        return fallback_cost, "conservative_fallback"

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current pricing cache."""
        with self._lock:
            if not self._cache:
                return {"status": "empty", "models": 0}

            return {
                "status": "loaded",
                "models": len(self._cache.data),
                "last_updated": self._cache.last_updated.isoformat(),
                "source": self._cache.source,
                "is_expired": self._cache.is_expired(self.cache_ttl_hours),
                "cache_ttl_hours": self.cache_ttl_hours,
            }

    def get_model_count(self) -> int:
        """Get the number of models in the pricing cache."""
        with self._lock:
            return len(self._cache.data) if self._cache else 0

    def list_supported_models(self) -> dict[str, str]:
        """Get list of all supported models with their providers."""
        with self._lock:
            if not self._cache:
                return {}

            return {
                model_name: info.provider or "unknown"
                for model_name, info in self._cache.data.items()
            }


# Global pricing manager instance
_global_pricing_manager: PricingManager | None = None
_global_pricing_lock = threading.RLock()


def get_pricing_manager() -> PricingManager:
    """Get the global pricing manager instance."""
    global _global_pricing_manager

    with _global_pricing_lock:
        if _global_pricing_manager is None:
            _global_pricing_manager = PricingManager()
        return cast(PricingManager, _global_pricing_manager)


def set_pricing_manager(manager: PricingManager | None) -> None:
    """Set the global pricing manager instance."""
    global _global_pricing_manager

    with _global_pricing_lock:
        _global_pricing_manager = manager


async def update_pricing_data(force: bool = False) -> bool:
    """Update global pricing data from LiteLLM."""
    manager = get_pricing_manager()
    return await manager.fetch_latest_pricing(force=force)


def estimate_model_cost(
    model_name: str, input_tokens: int, output_tokens: int = 0
) -> tuple[Decimal, str]:
    """Estimate cost using the global pricing manager."""
    manager = get_pricing_manager()
    return manager.estimate_cost(model_name, input_tokens, output_tokens)


def get_model_pricing_info(model_name: str) -> ModelPricingInfo | None:
    """Get pricing info using the global pricing manager."""
    manager = get_pricing_manager()
    return manager.get_model_pricing(model_name)
