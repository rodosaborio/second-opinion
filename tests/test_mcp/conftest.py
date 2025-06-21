"""
MCP-specific test fixtures and utilities.

This module provides mock utilities for testing MCP tools without
requiring real API keys or external dependencies.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.second_opinion.clients.base import BaseClient, ModelInfo
from src.second_opinion.core.models import (
    BudgetCheck,
    ModelRequest,
    ModelResponse,
    TaskComplexity,
    TokenUsage,
)


class MockClient(BaseClient):
    """Mock client for testing that doesn't require API keys."""

    def __init__(self, provider_name: str = "mock", **kwargs):
        # Initialize without calling super().__init__ to avoid validation
        self.provider_name = provider_name
        self.api_key = kwargs.get("api_key", "mock-key")
        self.base_url = kwargs.get("base_url", "https://mock.api")
        self.timeout = kwargs.get("timeout", 30)
        self.retries = kwargs.get("retries", 2)
        self.max_backoff = kwargs.get("max_backoff", 60)

        # Mock response data
        self.mock_responses = kwargs.get("mock_responses", {})
        self.mock_costs = kwargs.get("mock_costs", Decimal("0.01"))
        self.should_fail = kwargs.get("should_fail", False)
        self.failure_message = kwargs.get("failure_message", "Mock failure")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return mock response."""
        if self.should_fail:
            raise Exception(self.failure_message)

        # Get mock response for this model or use default
        mock_content = self.mock_responses.get(
            request.model,
            f"Mock response from {request.model} for: {request.messages[0].content[:50]}...",
        )

        return ModelResponse(
            content=mock_content,
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            cost_estimate=self.mock_costs,
            provider=self.provider_name,
            metadata={"mock": True},
        )

    async def estimate_cost(self, request: ModelRequest) -> Decimal:
        """Return mock cost estimate."""
        if self.should_fail:
            raise Exception(self.failure_message)
        return self.mock_costs

    async def list_models(self) -> List[ModelInfo]:
        """Return mock model list."""
        return [
            ModelInfo(
                id=f"{self.provider_name}/mock-model",
                name="Mock Model",
                provider=self.provider_name,
                context_length=4096,
                pricing={"prompt": 0.001, "completion": 0.002},
            )
        ]

    async def get_model_info(self, model: str) -> ModelInfo:
        """Return mock model info."""
        return ModelInfo(
            id=model,
            name=f"Mock {model}",
            provider=self.provider_name,
            context_length=4096,
            pricing={"prompt": 0.001, "completion": 0.002},
        )

    async def get_available_models(self) -> List[ModelInfo]:
        """Return mock available models."""
        return [
            ModelInfo(
                id=f"{self.provider_name}/mock-model-1",
                name="Mock Model 1",
                provider=self.provider_name,
                context_length=4096,
                pricing={"prompt": 0.001, "completion": 0.002},
            ),
            ModelInfo(
                id=f"{self.provider_name}/mock-model-2",
                name="Mock Model 2",
                provider=self.provider_name,
                context_length=8192,
                pricing={"prompt": 0.002, "completion": 0.004},
            ),
        ]


@pytest.fixture
def mock_openrouter_client():
    """Create a mock OpenRouter client with standard responses."""
    responses = {
        "anthropic/claude-3-5-sonnet": "I'm Claude, and here's my thoughtful response to your question.",
        "openai/gpt-4o": "As GPT-4, I can provide this comprehensive answer.",
        "google/gemini-pro": "Gemini here with an accurate and helpful response.",
    }

    return MockClient(
        provider_name="openrouter", mock_responses=responses, mock_costs=Decimal("0.02")
    )


@pytest.fixture
def mock_lmstudio_client():
    """Create a mock LM Studio client for local models."""
    responses = {
        "qwen3-4b-mlx": "Local Qwen model response",
        "llama-3-8b": "Local Llama model response",
    }

    return MockClient(
        provider_name="lmstudio",
        mock_responses=responses,
        mock_costs=Decimal("0.00"),  # Local models are free
    )


@pytest.fixture
def mock_failing_client():
    """Create a mock client that always fails."""
    return MockClient(
        provider_name="mock", should_fail=True, failure_message="API connection failed"
    )


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()

    # Cost management defaults
    settings.cost_management.default_per_request_limit = Decimal("0.05")
    settings.cost_management.daily_limit = Decimal("2.00")
    settings.cost_management.monthly_limit = Decimal("20.00")
    settings.cost_management.warning_threshold = 0.8

    # API configuration
    settings.api.timeout = 30
    settings.api.retries = 2
    settings.api.max_backoff = 60

    # Mock API key availability
    settings.get_api_key.return_value = "mock-api-key"
    settings.has_api_key.return_value = True

    return settings


@pytest.fixture
def mock_cost_guard():
    """Create mock cost guard for testing."""
    cost_guard = AsyncMock()

    # Mock successful budget check
    budget_check = BudgetCheck(
        approved=True,
        reservation_id="mock-reservation-123",
        estimated_cost=Decimal("0.05"),
        budget_remaining=Decimal("1.95"),
        daily_budget_remaining=Decimal("1.95"),
        monthly_budget_remaining=Decimal("19.95"),
        warning_message=None,
    )
    cost_guard.check_and_reserve_budget.return_value = budget_check

    # Mock cost recording
    cost_guard.record_actual_cost.return_value = AsyncMock()

    return cost_guard


@pytest.fixture
def mock_evaluator():
    """Create mock evaluator for testing."""
    evaluator = AsyncMock()

    # Mock task complexity classification
    evaluator.classify_task_complexity.return_value = TaskComplexity.MODERATE

    # Mock response comparison
    comparison_result = {
        "overall_winner": "primary",
        "overall_score": 7.5,
        "reasoning": "Primary response was more comprehensive and accurate.",
        "criteria_scores": {
            "accuracy": 8.0,
            "completeness": 7.0,
            "clarity": 7.5,
            "usefulness": 8.0,
        },
    }
    evaluator.compare_responses.return_value = comparison_result

    return evaluator


@pytest.fixture
def mock_client_factory(mock_openrouter_client, mock_lmstudio_client):
    """Create mock client factory that returns appropriate mock clients."""

    def factory(provider: str):
        if provider == "openrouter":
            return mock_openrouter_client
        elif provider == "lmstudio":
            return mock_lmstudio_client
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    return factory


@pytest.fixture
def mock_model_configs():
    """Create mock model configurations."""
    config_manager = MagicMock()

    # Mock tool config
    tool_config = MagicMock()
    tool_config.cost_limit_per_request = Decimal("0.10")
    tool_config.max_tokens = 2000
    tool_config.temperature = 0.1

    config_manager.get_tool_config.return_value = tool_config

    return config_manager


@pytest.fixture(autouse=True)
def patch_mcp_dependencies(
    monkeypatch,
    mock_settings,
    mock_cost_guard,
    mock_evaluator,
    mock_client_factory,
    mock_model_configs,
):
    """Automatically patch MCP dependencies for all MCP tests."""

    # Patch settings
    monkeypatch.setattr(
        "src.second_opinion.mcp.tools.second_opinion.get_settings",
        lambda: mock_settings,
    )

    # Patch cost guard
    monkeypatch.setattr(
        "src.second_opinion.mcp.tools.second_opinion.get_cost_guard",
        lambda: mock_cost_guard,
    )

    # Patch evaluator
    monkeypatch.setattr(
        "src.second_opinion.mcp.tools.second_opinion.get_evaluator",
        lambda: mock_evaluator,
    )

    # Patch client factory
    monkeypatch.setattr(
        "src.second_opinion.mcp.tools.second_opinion.create_client_from_config",
        mock_client_factory,
    )

    # Patch model configs - commented out since model_config_manager not yet implemented
    # monkeypatch.setattr(
    #     "src.second_opinion.mcp.tools.second_opinion.model_config_manager",
    #     mock_model_configs
    # )

    # Mock model provider detection
    def mock_detect_provider(model: str) -> str:
        if any(pattern in model.lower() for pattern in ["mlx", "qwen", "llama"]):
            return "lmstudio"
        return "openrouter"

    monkeypatch.setattr(
        "src.second_opinion.mcp.tools.second_opinion.detect_model_provider",
        mock_detect_provider,
    )


# Test data constants
SAMPLE_CODE_PROMPT = """What is the cleanest python code that can answer what the square root of a number is?"""

SAMPLE_CODE_RESPONSE = """```python
import math

def square_root(number):
    \"\"\"Calculates the square root of a number.\"\"\"
    if number < 0:
        raise ValueError("Cannot calculate the square root of a negative number")
    return math.sqrt(number)

# Example usage:
num = 16
print(f"The square root of {num} is {square_root(num)}")
```"""

SAMPLE_SHELL_PROMPT = "How do I list files recursively in bash?"

SAMPLE_SHELL_RESPONSE = """You can use the `find` command:

```bash
# List all files recursively
find /path/to/directory -type f

# List with details
find /path/to/directory -type f -ls
```"""
