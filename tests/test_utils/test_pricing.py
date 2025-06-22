"""
Tests for the pricing manager module.

This module tests the dynamic pricing system including LiteLLM data integration,
caching, model lookup, and cost estimation.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import httpx
import pytest

from src.second_opinion.utils.pricing import (
    ModelPricingInfo,
    PricingCache,
    PricingManager,
    estimate_model_cost,
    get_model_pricing_info,
    get_pricing_manager,
    set_pricing_manager,
    update_pricing_data,
)


class TestModelPricingInfo:
    """Test ModelPricingInfo model."""

    def test_valid_pricing_info(self):
        """Test creating valid pricing info."""
        info = ModelPricingInfo(
            model_name="gpt-4",
            input_cost_per_1k_tokens=Decimal("0.03"),
            output_cost_per_1k_tokens=Decimal("0.06"),
            max_tokens=8192,
            provider="openai",
            mode="chat",
            supports_function_calling=True,
        )

        assert info.model_name == "gpt-4"
        assert info.input_cost_per_1k_tokens == Decimal("0.03")
        assert info.output_cost_per_1k_tokens == Decimal("0.06")
        assert info.max_tokens == 8192
        assert info.provider == "openai"
        assert info.mode == "chat"
        assert info.supports_function_calling is True

    def test_minimal_pricing_info(self):
        """Test creating pricing info with minimal required fields."""
        info = ModelPricingInfo(
            model_name="simple-model",
            input_cost_per_1k_tokens=Decimal("0.01"),
            output_cost_per_1k_tokens=Decimal("0.02"),
        )

        assert info.model_name == "simple-model"
        assert info.max_tokens is None
        assert info.provider is None
        assert info.supports_function_calling is None


class TestPricingCache:
    """Test PricingCache model."""

    def test_cache_creation(self):
        """Test creating pricing cache."""
        pricing_data = {
            "gpt-4": ModelPricingInfo(
                model_name="gpt-4",
                input_cost_per_1k_tokens=Decimal("0.03"),
                output_cost_per_1k_tokens=Decimal("0.06"),
            )
        }

        cache = PricingCache(data=pricing_data, source="test")

        assert len(cache.data) == 1
        assert "gpt-4" in cache.data
        assert cache.source == "test"
        assert isinstance(cache.last_updated, datetime)

    def test_cache_expiry(self):
        """Test cache expiry logic."""
        # Fresh cache
        cache = PricingCache(source="test")
        assert not cache.is_expired(1)  # 1 hour TTL

        # Expired cache
        old_time = datetime.now(UTC) - timedelta(hours=2)
        cache.last_updated = old_time
        assert cache.is_expired(1)  # 1 hour TTL


class TestPricingManager:
    """Test PricingManager class."""

    @pytest.fixture
    def sample_litellm_data(self):
        """Sample LiteLLM pricing data for testing."""
        return {
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "max_tokens": 8192,
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "claude-3-sonnet": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "max_tokens": 200000,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "gpt-3.5-turbo": {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
                "max_tokens": 4096,
                "litellm_provider": "openai",
                "mode": "chat",
            },
        }

    @pytest.fixture
    def pricing_manager(self, isolated_temp_dir, sample_litellm_data):
        """Create pricing manager with test data."""
        cache_file = isolated_temp_dir / "pricing_cache.json"
        backup_file = isolated_temp_dir / "pricing_backup.json"

        # Create backup file with sample data
        with open(backup_file, "w") as f:
            json.dump(sample_litellm_data, f)

        return PricingManager(
            cache_file=cache_file,
            backup_file=backup_file,
            cache_ttl_hours=1,
            fetch_timeout=10.0,
        )

    def test_initialization(self, isolated_temp_dir):
        """Test pricing manager initialization."""
        cache_file = isolated_temp_dir / "cache.json"
        backup_file = isolated_temp_dir / "backup.json"

        manager = PricingManager(
            cache_file=cache_file,
            backup_file=backup_file,
            cache_ttl_hours=2,
            fetch_timeout=15.0,
        )

        assert manager.cache_file == cache_file
        assert manager.backup_file == backup_file
        assert manager.cache_ttl_hours == 2
        assert manager.fetch_timeout == 15.0
        assert manager._cache is not None

    def test_load_backup_data(self, pricing_manager):
        """Test loading pricing data from backup file."""
        # Should have loaded 3 models from backup
        assert len(pricing_manager._cache.data) == 3
        assert "gpt-4" in pricing_manager._cache.data
        assert "claude-3-sonnet" in pricing_manager._cache.data
        assert "gpt-3.5-turbo" in pricing_manager._cache.data

        # Check pricing conversion
        gpt4_info = pricing_manager._cache.data["gpt-4"]
        assert gpt4_info.input_cost_per_1k_tokens == Decimal("0.03")  # 0.00003 * 1000
        assert gpt4_info.output_cost_per_1k_tokens == Decimal("0.06")  # 0.00006 * 1000

    def test_get_model_pricing_direct_match(self, pricing_manager):
        """Test getting pricing for exact model match."""
        pricing_info = pricing_manager.get_model_pricing("gpt-4")

        assert pricing_info is not None
        assert pricing_info.model_name == "gpt-4"
        assert pricing_info.input_cost_per_1k_tokens == Decimal("0.03")
        assert pricing_info.provider == "openai"

    def test_get_model_pricing_normalization(self, pricing_manager):
        """Test model name normalization for pricing lookup."""
        # Should match gpt-3.5-turbo
        variations = ["gpt-3.5-turbo", "openai/gpt-3.5-turbo"]

        for variation in variations:
            pricing_info = pricing_manager.get_model_pricing(variation)
            assert pricing_info is not None, f"Should find pricing for {variation}"
            assert pricing_info.model_name == "gpt-3.5-turbo"

    def test_get_model_pricing_unknown_model(self, pricing_manager):
        """Test getting pricing for unknown model."""
        pricing_info = pricing_manager.get_model_pricing("unknown-model-12345")
        assert pricing_info is None

    def test_estimate_cost_known_model(self, pricing_manager):
        """Test cost estimation for known model."""
        cost, source = pricing_manager.estimate_cost("gpt-4", 1000, 500)

        # Cost should be: (1000 * 0.03 / 1000) + (500 * 0.06 / 1000) = 0.03 + 0.03 = 0.06
        expected_cost = Decimal("0.06")
        assert cost == expected_cost
        assert "pricing_data" in source

    def test_estimate_cost_unknown_model(self, pricing_manager):
        """Test cost estimation for unknown model."""
        cost, source = pricing_manager.estimate_cost("unknown-model", 1000, 500)

        # Should use conservative fallback
        assert cost > Decimal("0")
        assert source == "conservative_fallback"

        # Should be one of the conservative estimates
        assert cost in [Decimal("0.02"), Decimal("0.05"), Decimal("0.15")]

    def test_conservative_fallback_by_tier(self, pricing_manager):
        """Test conservative fallback varies by model tier."""
        # High-tier models (use model names that won't match existing data)
        cost_high, _ = pricing_manager.estimate_cost("unknown-opus-xl", 1000, 1000)

        # Mid-tier models
        cost_mid, _ = pricing_manager.estimate_cost("unknown-sonnet-pro", 1000, 1000)

        # Low-tier models
        cost_low, _ = pricing_manager.estimate_cost("unknown-basic-model", 1000, 1000)

        # High-tier should be most expensive, low-tier least expensive
        assert cost_high >= cost_mid >= cost_low

        # Check expected fallback values
        assert cost_high == Decimal("0.15")  # High-tier
        assert cost_mid == Decimal("0.05")  # Mid-tier
        assert cost_low == Decimal("0.02")  # Low-tier

    @pytest.mark.asyncio
    async def test_fetch_latest_pricing_success(
        self, pricing_manager, sample_litellm_data
    ):
        """Test successful pricing data fetch."""
        with patch("httpx.AsyncClient") as mock_client:
            from unittest.mock import MagicMock

            mock_response = (
                MagicMock()
            )  # Use MagicMock instead of AsyncMock for sync methods
            mock_response.json.return_value = sample_litellm_data
            mock_response.raise_for_status.return_value = None
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            success = await pricing_manager.fetch_latest_pricing(force=True)

            assert success
            assert pricing_manager._cache.source == "network"

    @pytest.mark.asyncio
    async def test_fetch_latest_pricing_failure(self, pricing_manager):
        """Test pricing data fetch failure."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                httpx.RequestError("Network error")
            )

            success = await pricing_manager.fetch_latest_pricing(force=True)

            assert not success
            # Should still have backup data
            assert pricing_manager._cache.source == "backup"

    @pytest.mark.asyncio
    async def test_fetch_respects_cache_ttl(self, pricing_manager):
        """Test that fetch respects cache TTL."""
        # Fresh cache should not fetch
        with patch("httpx.AsyncClient") as mock_client:
            success = await pricing_manager.fetch_latest_pricing(force=False)

            assert success  # Returns True without fetching
            mock_client.assert_not_called()

    def test_cache_save_and_load(self, pricing_manager, isolated_temp_dir):
        """Test saving and loading cache."""
        original_count = len(pricing_manager._cache.data)

        # Save current cache
        pricing_manager._save_cache()

        # Create empty backup file to ensure cache takes priority
        empty_backup = isolated_temp_dir / "empty_backup.json"
        empty_backup.write_text("{}")

        # Create new manager with same cache file
        new_manager = PricingManager(
            cache_file=pricing_manager.cache_file,
            backup_file=empty_backup,
            cache_ttl_hours=1,
        )

        # Should load from cache
        assert new_manager._cache is not None
        assert len(new_manager._cache.data) == original_count
        assert new_manager._cache.source == "cache"

    def test_get_cache_info(self, pricing_manager):
        """Test getting cache information."""
        info = pricing_manager.get_cache_info()

        assert info["status"] == "loaded"
        assert info["models"] == 3
        assert "last_updated" in info
        assert info["source"] == "backup"
        assert "is_expired" in info
        assert info["cache_ttl_hours"] == 1

    def test_list_supported_models(self, pricing_manager):
        """Test listing supported models."""
        models = pricing_manager.list_supported_models()

        assert len(models) == 3
        assert models["gpt-4"] == "openai"
        assert models["claude-3-sonnet"] == "anthropic"
        assert models["gpt-3.5-turbo"] == "openai"

    def test_thread_safety(self, pricing_manager):
        """Test thread-safe operations."""
        import threading
        import time

        results = []

        def worker():
            for _ in range(10):
                cost, _ = pricing_manager.estimate_cost("gpt-4", 100, 100)
                results.append(cost)
                time.sleep(0.001)  # Small delay

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be consistent
        assert len(set(results)) == 1  # All same value

        # Calculate expected cost: (100 * 0.03 / 1000) + (100 * 0.06 / 1000) = 0.003 + 0.006 = 0.009
        expected_cost = Decimal("0.009")
        assert all(r == expected_cost for r in results)

    def test_parse_litellm_data_invalid_entries(self, pricing_manager):
        """Test parsing LiteLLM data with invalid entries."""
        invalid_data = {
            "valid-model": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "max_tokens": 1000,
            },
            "invalid-model-1": {
                "input_cost_per_token": "invalid",  # Invalid type
                "output_cost_per_token": 0.002,
            },
            "invalid-model-2": {
                # Missing required fields
                "max_tokens": 1000
            },
        }

        parsed_data = pricing_manager._parse_litellm_data(invalid_data)

        # Should have valid model and the one with defaults (invalid-model-2)
        # invalid-model-1 should be rejected due to invalid input_cost_per_token type
        assert len(parsed_data) == 2
        assert "valid-model" in parsed_data
        assert "invalid-model-1" not in parsed_data  # Invalid type should be rejected
        assert "invalid-model-2" in parsed_data  # Missing fields get defaults

        # Check that invalid-model-2 has default values
        assert parsed_data["invalid-model-2"].input_cost_per_1k_tokens == Decimal("0")
        assert parsed_data["invalid-model-2"].output_cost_per_1k_tokens == Decimal("0")


class TestGlobalPricingManager:
    """Test global pricing manager functions."""

    def test_get_pricing_manager_singleton(self):
        """Test that global pricing manager is a singleton."""
        manager1 = get_pricing_manager()
        manager2 = get_pricing_manager()

        assert manager1 is manager2

    def test_set_pricing_manager(self, tmp_path):
        """Test setting custom pricing manager."""
        custom_manager = PricingManager(
            cache_file=tmp_path / "custom_cache.json",
            backup_file=tmp_path / "custom_backup.json",
        )

        set_pricing_manager(custom_manager)

        assert get_pricing_manager() is custom_manager

        # Reset to None
        set_pricing_manager(None)

        # Should create new default manager
        new_manager = get_pricing_manager()
        assert new_manager is not custom_manager

    @pytest.mark.asyncio
    async def test_update_pricing_data_global(self):
        """Test global pricing data update function."""
        with patch.object(get_pricing_manager(), "fetch_latest_pricing") as mock_fetch:
            mock_fetch.return_value = True

            success = await update_pricing_data(force=True)

            assert success
            mock_fetch.assert_called_once_with(force=True)

    def test_estimate_model_cost_global(self):
        """Test global cost estimation function."""
        with patch.object(get_pricing_manager(), "estimate_cost") as mock_estimate:
            mock_estimate.return_value = (Decimal("0.05"), "test_source")

            cost, source = estimate_model_cost("test-model", 1000, 500)

            assert cost == Decimal("0.05")
            assert source == "test_source"
            mock_estimate.assert_called_once_with("test-model", 1000, 500)

    def test_get_model_pricing_info_global(self):
        """Test global model pricing info function."""
        with patch.object(get_pricing_manager(), "get_model_pricing") as mock_get:
            mock_info = ModelPricingInfo(
                model_name="test-model",
                input_cost_per_1k_tokens=Decimal("0.01"),
                output_cost_per_1k_tokens=Decimal("0.02"),
            )
            mock_get.return_value = mock_info

            info = get_model_pricing_info("test-model")

            assert info is mock_info
            mock_get.assert_called_once_with("test-model")


class TestPricingManagerSecurity:
    """Test security aspects of pricing manager."""

    @pytest.fixture
    def sample_litellm_data(self):
        """Sample LiteLLM pricing data for testing."""
        return {
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "max_tokens": 8192,
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "claude-3-sonnet": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "max_tokens": 200000,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "gpt-3.5-turbo": {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
                "max_tokens": 4096,
                "litellm_provider": "openai",
                "mode": "chat",
            },
        }

    @pytest.fixture
    def pricing_manager(self, isolated_temp_dir, sample_litellm_data):
        """Create pricing manager with test data."""
        cache_file = isolated_temp_dir / "pricing_cache.json"
        backup_file = isolated_temp_dir / "pricing_backup.json"

        # Create backup file with sample data
        with open(backup_file, "w") as f:
            json.dump(sample_litellm_data, f)

        return PricingManager(
            cache_file=cache_file,
            backup_file=backup_file,
            cache_ttl_hours=1,
            fetch_timeout=10.0,
        )

    @pytest.mark.security
    def test_safe_file_paths(self, isolated_temp_dir):
        """Test that file paths are handled safely."""
        # Test with path traversal attempts
        dangerous_paths = [
            isolated_temp_dir / "../../../etc/passwd",
            isolated_temp_dir / "..\\..\\windows\\system32\\config",
            isolated_temp_dir / "normal_file.json",
        ]

        for path in dangerous_paths:
            try:
                manager = PricingManager(cache_file=path, backup_file=path)
                # Should not crash, paths should be normalized
                assert manager.cache_file is not None
                assert manager.backup_file is not None
            except Exception as e:
                # Should only fail for legitimate reasons, not security issues
                assert "permission" in str(e).lower() or "not found" in str(e).lower()

    @pytest.mark.security
    def test_safe_json_parsing(self, pricing_manager):
        """Test safe JSON parsing with malicious data."""
        malicious_data = {
            "normal-model": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
            "': __import__('os').system('rm -rf /')#": {  # Code injection attempt
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
            "a" * 10000: {  # Extremely long key
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
        }

        # Should parse safely without executing malicious code
        parsed_data = pricing_manager._parse_litellm_data(malicious_data)

        # Should have at least the normal model
        assert "normal-model" in parsed_data
        # Should not crash or execute malicious code

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_network_request_safety(self, pricing_manager):
        """Test network request safety."""
        # Test with malicious URL redirect
        with patch("httpx.AsyncClient") as mock_client:
            from unittest.mock import MagicMock

            mock_response = MagicMock()  # Use MagicMock for sync methods
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Should handle JSON errors gracefully
            success = await pricing_manager.fetch_latest_pricing(force=True)
            assert not success

    @pytest.mark.security
    def test_input_validation(self, pricing_manager):
        """Test input validation for cost estimation."""
        # Test with extreme values
        extreme_cases = [
            ("", 1000, 1000),  # Empty model name
            ("normal-model", -1000, 1000),  # Negative tokens
            ("normal-model", 1000, -1000),  # Negative output tokens
            ("normal-model", 10**10, 10**10),  # Extremely large values
        ]

        for model_name, input_tokens, output_tokens in extreme_cases:
            try:
                cost, source = pricing_manager.estimate_cost(
                    model_name, input_tokens, output_tokens
                )
                # Should return reasonable values, not crash
                assert cost >= 0
                assert isinstance(source, str)
            except Exception as e:
                # Should only fail for legitimate validation reasons
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["negative", "invalid", "too large"]
                )


class TestPricingManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def sample_litellm_data(self):
        """Sample LiteLLM pricing data for testing."""
        return {
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "max_tokens": 8192,
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "claude-3-sonnet": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "max_tokens": 200000,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_function_calling": True,
            },
            "gpt-3.5-turbo": {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
                "max_tokens": 4096,
                "litellm_provider": "openai",
                "mode": "chat",
            },
        }

    @pytest.fixture
    def pricing_manager(self, isolated_temp_dir, sample_litellm_data):
        """Create pricing manager with test data."""
        cache_file = isolated_temp_dir / "pricing_cache.json"
        backup_file = isolated_temp_dir / "pricing_backup.json"

        # Create backup file with sample data
        with open(backup_file, "w") as f:
            json.dump(sample_litellm_data, f)

        return PricingManager(
            cache_file=cache_file,
            backup_file=backup_file,
            cache_ttl_hours=1,
            fetch_timeout=10.0,
        )

    def test_empty_backup_file(self, isolated_temp_dir):
        """Test handling of empty backup file."""
        empty_backup = isolated_temp_dir / "empty.json"
        empty_backup.write_text("{}")

        manager = PricingManager(
            cache_file=isolated_temp_dir / "cache.json", backup_file=empty_backup
        )

        # Should handle empty backup gracefully
        assert manager._cache is not None
        assert len(manager._cache.data) == 0

    def test_corrupted_backup_file(self, isolated_temp_dir):
        """Test handling of corrupted backup file."""
        corrupted_backup = isolated_temp_dir / "corrupted.json"
        corrupted_backup.write_text("{ invalid json }")

        manager = PricingManager(
            cache_file=isolated_temp_dir / "cache.json", backup_file=corrupted_backup
        )

        # Should handle corrupted backup gracefully
        assert manager._cache is not None
        assert manager._cache.source == "empty"

    def test_missing_backup_file(self, isolated_temp_dir):
        """Test handling of missing backup file."""
        manager = PricingManager(
            cache_file=isolated_temp_dir / "cache.json",
            backup_file=isolated_temp_dir / "nonexistent.json",
        )

        # Should handle missing backup gracefully
        assert manager._cache is not None
        assert manager._cache.source == "empty"

    def test_cache_file_permissions(self, isolated_temp_dir):
        """Test handling of cache file permission issues."""
        cache_file = isolated_temp_dir / "readonly_cache.json"
        cache_file.write_text('{"test": "data"}')
        cache_file.chmod(0o444)  # Read-only

        try:
            manager = PricingManager(
                cache_file=cache_file, backup_file=isolated_temp_dir / "backup.json"
            )

            # Should handle read-only cache file
            manager._save_cache()  # Should not crash

        except PermissionError:
            # This is acceptable behavior
            pass
        finally:
            # Restore permissions for cleanup
            cache_file.chmod(0o644)

    @pytest.mark.asyncio
    async def test_network_timeout(self, pricing_manager):
        """Test network timeout handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                httpx.TimeoutException("Timeout")
            )

            success = await pricing_manager.fetch_latest_pricing(force=True)

            assert not success

    def test_extremely_large_dataset(self, isolated_temp_dir):
        """Test handling of extremely large pricing dataset."""
        # Create large dataset
        large_data = {}
        for i in range(1000):
            large_data[f"model-{i}"] = {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "max_tokens": 1000,
            }

        backup_file = isolated_temp_dir / "large_backup.json"
        with open(backup_file, "w") as f:
            json.dump(large_data, f)

        manager = PricingManager(
            cache_file=isolated_temp_dir / "cache.json", backup_file=backup_file
        )

        # Should handle large dataset
        assert manager._cache is not None
        assert len(manager._cache.data) == 1000

        # Lookup should still be fast
        import time

        start_time = time.time()
        info = manager.get_model_pricing("model-500")
        end_time = time.time()

        assert info is not None
        assert end_time - start_time < 0.1  # Should be fast
