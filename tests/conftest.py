"""
Shared test fixtures and configuration for Second Opinion tests.

This file provides global state management, resource cleanup, and common
test utilities to ensure proper test isolation and prevent hanging tests.
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Suppress specific warnings that can slow down tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Configure logging for tests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("second_opinion").setLevel(logging.WARNING)


@pytest.fixture(autouse=True, scope="function")
def isolated_environment(tmp_path):
    """
    Isolate environment variables for each test to prevent test pollution.

    This ensures that configuration tests don't inherit environment variables
    from the host system, .env files, or other tests.
    """
    import os

    # Store original environment variables that could affect settings
    sensitive_env_vars = [
        "LOG_LEVEL",
        "ENVIRONMENT",
        "APP_NAME",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "LMSTUDIO_BASE_URL",
        "DATABASE_ENCRYPTION_KEY",
        "COST_LIMIT_DAILY",
        "COST_LIMIT_MONTHLY",
    ]

    original_env = {}
    for var in sensitive_env_vars:
        original_env[var] = os.environ.get(var)
        # Remove from environment to test defaults
        if var in os.environ:
            del os.environ[var]

    # Change working directory to temp path to avoid loading .env files
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    # Restore original environment and working directory
    os.chdir(original_cwd)
    for var, original_value in original_env.items():
        if original_value is not None:
            os.environ[var] = original_value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture(autouse=True, scope="function")
def reset_global_state():
    """
    Reset all global state between tests to ensure isolation.

    This fixture automatically runs before and after each test to:
    - Reset configuration manager
    - Reset pricing manager
    - Reset cost guard
    - Clear any cached instances
    - Close any open HTTP clients
    """
    # Reset before test
    _reset_all_global_state()

    yield

    # Reset after test
    _reset_all_global_state()


@pytest.fixture(autouse=True, scope="function")
def ensure_async_cleanup():
    """
    Ensure proper cleanup of async resources.

    This helps prevent hanging tests by ensuring all async resources
    are properly closed and event loops are cleaned up.
    """
    yield

    # Force cleanup of any remaining async tasks
    try:
        loop = asyncio.get_running_loop()
        # Cancel any pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()

        # Wait briefly for cancellation
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except RuntimeError:
        # No running loop, which is fine
        pass


@pytest.fixture(scope="function")
def isolated_temp_dir(tmp_path):
    """
    Provide isolated temporary directory for each test.

    Sets up a clean temporary directory and updates environment
    variables to use it for data storage.
    """
    # Set environment variables to use temp directory
    original_env = {}
    temp_env_vars = {
        "DATA_DIR": str(tmp_path / "data"),
        "CONFIG_DIR": str(tmp_path / "config"),
        "PROMPTS_DIR": str(tmp_path / "prompts"),
        "DATABASE__PATH": str(tmp_path / "data" / "test.db"),
    }

    for key, value in temp_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Create directories
    for path in temp_env_vars.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    yield tmp_path

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(scope="function")
def mock_http_client():
    """
    Provide a mock HTTP client for testing.

    This prevents tests from making real network calls and
    ensures consistent, fast test execution.
    """
    mock_client = AsyncMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-response",
        "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.raise_for_status.return_value = None

    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    return mock_client


@pytest.fixture(autouse=True, scope="function")
def mock_asyncio_sleep():
    """
    Mock asyncio.sleep to prevent actual delays during testing.

    This prevents retry logic from causing real delays that slow down tests
    and can cause timeouts in CI environments.
    """
    from unittest.mock import patch

    # Mock asyncio.sleep to return immediately
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        async def instant_sleep(delay):
            # Log the intended delay for debugging but return immediately
            pass

        mock_sleep.side_effect = instant_sleep
        yield mock_sleep


@pytest.fixture(autouse=True, scope="function")
def limit_retries_in_tests(request):
    """
    Limit the number of retries during testing to prevent infinite loops.

    This patches the retry_with_backoff method to use minimal retries during testing,
    but skips the patch for tests that are specifically testing retry logic.
    """
    from unittest.mock import patch

    # Skip patching for tests that specifically test retry behavior
    if "retry_with_backoff" in request.node.name or "test_retry" in request.node.name:
        # Let these tests use the real retry logic (but we'll still have asyncio.sleep mocked)
        yield None
        return

    # Patch retry_with_backoff to have test-friendly behavior for integration tests
    with patch(
        "src.second_opinion.clients.base.BaseClient.retry_with_backoff"
    ) as mock_retry:

        async def test_retry_with_backoff(operation, *args, **kwargs):
            """Test-friendly retry with minimal attempts."""
            last_exception = None
            max_retries = 1  # Only try once for integration tests

            for attempt in range(max_retries + 1):
                try:
                    return await operation(*args, **kwargs)
                except Exception as e:
                    # Import the exception types we need
                    from src.second_opinion.clients.base import (
                        RetryableError,
                    )

                    if isinstance(e, RetryableError):
                        last_exception = e
                        if attempt == max_retries:
                            break
                        # No delay in tests - just continue to next attempt
                        continue
                    else:
                        # Non-retryable errors, re-raise immediately
                        raise

            # All retries exhausted, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise Exception("Unexpected retry logic state")

        mock_retry.side_effect = test_retry_with_backoff
        yield mock_retry


@pytest.fixture(scope="function")
def test_settings():
    """
    Provide test-specific settings configuration.

    Creates a settings instance with safe test defaults that
    won't interfere with production configurations.
    """
    from src.second_opinion.config.settings import AppSettings

    # Test environment variables
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",  # Reduce log noise in tests
        "OPENROUTER_API_KEY": "sk-or-test-key-for-testing",
        "ANTHROPIC_API_KEY": "sk-ant-test-key-for-testing",
        "DATABASE_ENCRYPTION_KEY": "test-encryption-key-32-chars-long",
        "DATA_DIR": "./test_data",
        "PRICING__ENABLED": "false",  # Disable pricing fetches in tests
        "PRICING__AUTO_UPDATE_ON_STARTUP": "false",
        "COST_MANAGEMENT__DAILY_LIMIT": "100.00",  # High limits for testing
        "COST_MANAGEMENT__MONTHLY_LIMIT": "1000.00",
    }

    # Set test environment temporarily
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        settings = AppSettings()
        yield settings
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@pytest.fixture(scope="function")
def mock_pricing_manager():
    """
    Provide a mock pricing manager for testing.

    This prevents tests from making network calls to fetch pricing
    data and provides predictable cost estimates.
    """
    from decimal import Decimal

    from src.second_opinion.utils.pricing import (
        ModelPricingInfo,
        PricingManager,
        set_pricing_manager,
    )

    # Create mock with test data
    mock_manager = MagicMock(spec=PricingManager)

    # Default pricing for common test models
    test_pricing = {
        "gpt-3.5-turbo": ModelPricingInfo(
            model_name="gpt-3.5-turbo",
            input_cost_per_1k_tokens=Decimal("0.0015"),
            output_cost_per_1k_tokens=Decimal("0.002"),
            max_tokens=4096,
            provider="openai",
        ),
        "gpt-4": ModelPricingInfo(
            model_name="gpt-4",
            input_cost_per_1k_tokens=Decimal("0.03"),
            output_cost_per_1k_tokens=Decimal("0.06"),
            max_tokens=8192,
            provider="openai",
        ),
        "claude-3-haiku": ModelPricingInfo(
            model_name="claude-3-haiku",
            input_cost_per_1k_tokens=Decimal("0.00025"),
            output_cost_per_1k_tokens=Decimal("0.00125"),
            max_tokens=200000,
            provider="anthropic",
        ),
    }

    def mock_get_model_pricing(model_name: str):
        return test_pricing.get(model_name)

    def mock_estimate_cost(model_name: str, input_tokens: int, output_tokens: int = 0):
        pricing = test_pricing.get(model_name)
        if pricing:
            input_cost = (
                Decimal(input_tokens) * pricing.input_cost_per_1k_tokens
            ) / 1000
            output_cost = (
                Decimal(output_tokens) * pricing.output_cost_per_1k_tokens
            ) / 1000
            return input_cost + output_cost, "mock_pricing"
        return Decimal("0.05"), "mock_fallback"

    mock_manager.get_model_pricing = mock_get_model_pricing
    mock_manager.estimate_cost = mock_estimate_cost
    mock_manager.get_cache_info.return_value = {
        "status": "loaded",
        "models": len(test_pricing),
        "source": "mock",
        "is_expired": False,
    }

    # Set as global manager
    set_pricing_manager(mock_manager)

    yield mock_manager

    # Clean up
    set_pricing_manager(None)


@pytest.fixture(scope="function")
def clean_cost_guard():
    """
    Provide a clean cost guard instance for testing.

    Creates a new cost guard with test-friendly limits and
    ensures it's reset between tests.
    """
    from decimal import Decimal

    from src.second_opinion.utils.cost_tracking import CostGuard, set_cost_guard

    # Create test cost guard with high limits
    test_guard = CostGuard(
        per_request_limit=Decimal("10.00"),  # High limit for testing
        daily_limit=Decimal("100.00"),
        weekly_limit=Decimal("500.00"),
        monthly_limit=Decimal("1000.00"),
        warning_threshold=0.8,
        reservation_timeout=300,
    )

    set_cost_guard(test_guard)

    yield test_guard

    # Clean up
    set_cost_guard(None)


def _reset_all_global_state():
    """Reset all global state across the application."""

    # Reset configuration manager
    try:
        from src.second_opinion.config.settings import config_manager

        config_manager.reset()
    except ImportError:
        pass

    # Reset pricing manager
    try:
        from src.second_opinion.utils.pricing import set_pricing_manager

        set_pricing_manager(None)
    except ImportError:
        pass

    # Reset cost guard
    try:
        from src.second_opinion.utils.cost_tracking import set_cost_guard

        set_cost_guard(None)
    except ImportError:
        pass

    # Clear any module-level caches
    _clear_module_caches()


def _clear_module_caches():
    """Clear any module-level caches that might persist between tests."""

    # Clear any function caches
    try:
        import functools

        # Clear lru_cache instances if any exist
        # This is a generic approach - specific caches can be added as needed
    except ImportError:
        pass

    # Clear any global variables that might cache instances
    import sys

    for module_name, module in sys.modules.items():
        if module_name.startswith("src.second_opinion"):
            # Look for common global cache patterns
            if hasattr(module, "_cache"):
                if hasattr(module._cache, "clear"):
                    module._cache.clear()
            if hasattr(module, "_instances"):
                if hasattr(module._instances, "clear"):
                    module._instances.clear()


# Test timeout configuration
@pytest.fixture(autouse=True)
def set_test_timeout():
    """Set reasonable timeouts for tests to prevent hanging."""
    # This is more of a documentation fixture
    # Actual timeouts are configured in pytest.ini
    yield


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "security: marks tests as security-focused tests"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers and timeouts."""
    for item in items:
        # Add timeout marker to all async tests
        if asyncio.iscoroutinefunction(item.function):
            # Add timeout to prevent hanging async tests
            item.add_marker(pytest.mark.timeout(30))  # 30 second timeout

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)


def pytest_runtest_setup(item):
    """Setup hook to run before each test."""
    # State reset is handled by the reset_global_state fixture
    pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown hook to run after each test."""
    # State reset is handled by the reset_global_state fixture
    # Force garbage collection to help with resource cleanup
    import gc

    gc.collect()


# Performance optimization for test runs
@pytest.fixture(scope="session", autouse=True)
def optimize_test_performance():
    """Optimize performance for test runs."""

    # Reduce logging overhead
    logging.getLogger().setLevel(logging.ERROR)

    # Disable networking where possible
    os.environ.setdefault("HTTPX_DISABLE_POOL", "1")

    yield

    # Cleanup after all tests
    _reset_all_global_state()
