"""
Tests for CLI main functionality.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from second_opinion.cli.main import CLIError, ComparisonModelSelector, app
from second_opinion.core.models import ModelResponse, TaskComplexity, TokenUsage


# Test fixtures
@pytest.fixture
def cli_runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    return ModelResponse(
        content="Test response content",
        model="test/model",
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        cost_estimate=Decimal("0.025"),
        provider="test_provider",
    )


@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    config = MagicMock()
    config.model_tiers = MagicMock()
    config.model_tiers.budget = ["anthropic/claude-3-haiku", "openai/gpt-4o-mini"]
    config.model_tiers.mid_range = ["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]
    config.model_tiers.premium = ["anthropic/claude-3-opus", "openai/o1-pro"]
    config.model_tiers.reasoning = ["openai/o1-pro", "openai/o1-mini"]
    return config


class TestComparisonModelSelector:
    """Test comparison model selection logic."""

    @patch("second_opinion.cli.main.model_config_manager")
    def test_explicit_model_selection(
        self, mock_model_config_manager, mock_model_config
    ):
        """Test explicit model selection via CLI."""
        mock_model_config_manager.config = mock_model_config
        selector = ComparisonModelSelector()

        result = selector.select_models(
            primary_model="anthropic/claude-3-5-sonnet",
            explicit_models=["openai/gpt-4o", "google/gemini-pro"],
        )

        assert result == ["openai/gpt-4o", "google/gemini-pro"]

    @patch("second_opinion.cli.main.model_config_manager")
    def test_filter_duplicate_primary_model(
        self, mock_model_config_manager, mock_model_config
    ):
        """Test filtering out duplicate primary model."""
        mock_model_config_manager.config = mock_model_config
        selector = ComparisonModelSelector()

        with patch("second_opinion.cli.main.console") as mock_console:
            result = selector.select_models(
                primary_model="anthropic/claude-3-5-sonnet",
                explicit_models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o"],
            )

            assert result == ["openai/gpt-4o"]
            mock_console.print.assert_called_once()
            assert "same as primary model" in str(mock_console.print.call_args)

    @patch("second_opinion.cli.main.model_config_manager")
    def test_invalid_model_name(self, mock_model_config_manager, mock_model_config):
        """Test error handling for invalid model names."""
        mock_model_config_manager.config = mock_model_config
        selector = ComparisonModelSelector()

        with pytest.raises(CLIError) as exc_info:
            selector.select_models(
                primary_model="anthropic/claude-3-5-sonnet",
                explicit_models=["invalid-model-name"],
            )

        assert "Invalid comparison model name" in str(exc_info.value)

    @patch("second_opinion.cli.main.model_config_manager")
    def test_config_based_selection(self, mock_model_config_manager, mock_model_config):
        """Test model selection from configuration."""
        mock_model_config_manager.get_comparison_models.return_value = [
            "openai/gpt-4o",
            "google/gemini-pro",
        ]
        selector = ComparisonModelSelector()

        result = selector.select_models(
            primary_model="anthropic/claude-3-5-sonnet", tool_name="second_opinion"
        )

        assert result == ["openai/gpt-4o", "google/gemini-pro"]

    @patch("second_opinion.cli.main.model_config_manager")
    def test_smart_selection_budget_tier(
        self, mock_model_config_manager, mock_model_config
    ):
        """Test smart selection for budget tier models."""
        mock_model_config_manager.config = mock_model_config
        # Ensure get_comparison_models returns empty to fall through to smart select
        mock_model_config_manager.get_comparison_models.return_value = []
        selector = ComparisonModelSelector()

        result = selector.select_models(
            primary_model="anthropic/claude-3-haiku",
            task_complexity=TaskComplexity.MODERATE,
            max_models=2,
        )

        # Budget tier should compare against mid_range models
        assert any(model in mock_model_config.model_tiers.mid_range for model in result)
        assert "anthropic/claude-3-haiku" not in result  # Exclude primary model

    @patch("second_opinion.cli.main.model_config_manager")
    def test_smart_selection_mid_range_tier(
        self, mock_model_config_manager, mock_model_config
    ):
        """Test smart selection for mid-range tier models."""
        mock_model_config_manager.config = mock_model_config
        selector = ComparisonModelSelector()

        result = selector.select_models(
            primary_model="anthropic/claude-3-5-sonnet",
            task_complexity=TaskComplexity.COMPLEX,
            max_models=2,
        )

        # Should include mix of budget and premium
        assert len(result) <= 2
        assert "anthropic/claude-3-5-sonnet" not in result

    @patch("second_opinion.cli.main.model_config_manager")
    def test_fallback_to_defaults(self, mock_model_config_manager):
        """Test fallback to default models when config unavailable."""
        mock_model_config_manager.config = None

        selector = ComparisonModelSelector()
        result = selector.select_models(primary_model="test/model", max_models=2)

        # Should return default models excluding primary
        assert len(result) <= 2
        assert "test/model" not in result
        assert all("/" in model for model in result)  # Valid format


class TestCLICommands:
    """Test CLI command functionality."""

    @patch("second_opinion.cli.main.execute_second_opinion")
    @patch("second_opinion.cli.main.display_results")
    def test_second_opinion_command_basic(
        self, mock_display, mock_execute, cli_runner, mock_model_response
    ):
        """Test basic second opinion command."""
        # Mock the async execution
        mock_execute.return_value = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.05"),
            "estimated_cost": Decimal("0.048"),
        }

        result = cli_runner.invoke(
            app,
            [
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "What is 2+2?",
            ],
        )

        assert result.exit_code == 0
        mock_execute.assert_called_once()
        mock_display.assert_called_once()

    @patch("second_opinion.cli.main.execute_second_opinion")
    def test_second_opinion_with_comparison_model(
        self, mock_execute, cli_runner, mock_model_response
    ):
        """Test second opinion with explicit comparison model."""
        mock_execute.return_value = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.05"),
            "estimated_cost": Decimal("0.048"),
        }

        result = cli_runner.invoke(
            app,
            [
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "--comparison-model",
                "openai/gpt-4o",
                "What is 2+2?",
            ],
        )

        assert result.exit_code == 0
        # Check that the selector received the explicit model
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        assert call_args["comparison_models"] == ["openai/gpt-4o"]

    @patch("second_opinion.cli.main.execute_second_opinion")
    def test_second_opinion_multiple_comparison_models(
        self, mock_execute, cli_runner, mock_model_response
    ):
        """Test second opinion with multiple comparison models."""
        mock_execute.return_value = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response, mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.08"),
            "estimated_cost": Decimal("0.075"),
        }

        result = cli_runner.invoke(
            app,
            [
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "--comparison-model",
                "openai/gpt-4o",
                "--comparison-model",
                "google/gemini-pro",
                "Complex analysis task",
            ],
        )

        assert result.exit_code == 0
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        assert "openai/gpt-4o" in call_args["comparison_models"]
        assert "google/gemini-pro" in call_args["comparison_models"]

    def test_second_opinion_missing_primary_model(self, cli_runner):
        """Test error when primary model is not specified."""
        result = cli_runner.invoke(app, ["second-opinion", "What is 2+2?"])

        assert result.exit_code != 0
        # Error message appears in stderr for typer
        error_output = (result.stdout + result.stderr).lower()
        assert "primary-model" in error_output or "missing" in error_output

    @patch("second_opinion.cli.main.execute_second_opinion")
    def test_second_opinion_with_context(
        self, mock_execute, cli_runner, mock_model_response
    ):
        """Test second opinion with context parameter."""
        mock_execute.return_value = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.05"),
            "estimated_cost": Decimal("0.048"),
        }

        result = cli_runner.invoke(
            app,
            [
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "--context",
                "This is for a math homework assignment",
                "What is 2+2?",
            ],
        )

        assert result.exit_code == 0
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        assert call_args["context"] == "This is for a math homework assignment"

    @patch("second_opinion.cli.main.execute_second_opinion")
    def test_cost_limit_parameter(self, mock_execute, cli_runner, mock_model_response):
        """Test custom cost limit parameter."""
        mock_execute.return_value = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.15"),
            "estimated_cost": Decimal("0.148"),
        }

        result = cli_runner.invoke(
            app,
            [
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "--cost-limit",
                "0.20",
                "Expensive analysis task",
            ],
        )

        assert result.exit_code == 0
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        assert call_args["cost_limit"] == 0.20


class TestAsyncExecution:
    """Test async execution wrapper and error handling."""

    @patch("second_opinion.cli.main.sanitize_prompt")
    @patch("second_opinion.cli.main.get_cost_guard")
    @patch("second_opinion.cli.main.get_evaluator")
    @patch("second_opinion.cli.main.get_client_for_model")
    async def test_execute_second_opinion_success(
        self,
        mock_get_client_for_model,
        mock_get_evaluator,
        mock_get_cost_guard,
        mock_sanitize_prompt,
        mock_model_response,
    ):
        """Test successful execution of second opinion operation."""
        from second_opinion.cli.main import execute_second_opinion

        # Setup mocks
        mock_sanitize_prompt.side_effect = lambda x, _: x

        mock_cost_guard = AsyncMock()
        mock_cost_guard.check_and_reserve_budget.return_value = MagicMock(
            reservation_id="test123"
        )
        mock_cost_guard.record_actual_cost.return_value = None
        mock_get_cost_guard.return_value = mock_cost_guard

        mock_evaluator = AsyncMock()
        mock_evaluation = MagicMock()
        mock_evaluation.quality_score = 8.5
        mock_evaluator.compare_responses.return_value = mock_evaluation
        mock_get_evaluator.return_value = mock_evaluator

        mock_client = AsyncMock()
        mock_client.estimate_cost.return_value = Decimal("0.025")
        mock_client.complete.return_value = mock_model_response
        mock_get_client_for_model.return_value = mock_client

        # Execute
        result = await execute_second_opinion(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            comparison_models=["openai/gpt-4o"],
            cost_limit=0.10,
        )

        # Verify
        assert "primary_response" in result
        assert "comparison_responses" in result
        assert "total_cost" in result
        assert len(result["comparison_responses"]) == 1

        mock_cost_guard.check_and_reserve_budget.assert_called_once()
        mock_cost_guard.record_actual_cost.assert_called_once()

    @patch("second_opinion.cli.main.sanitize_prompt")
    @patch("second_opinion.cli.main.get_cost_guard")
    @patch("second_opinion.cli.main.get_client_for_model")
    async def test_execute_second_opinion_cost_limit_exceeded(
        self,
        mock_get_client_for_model,
        mock_get_cost_guard,
        mock_sanitize_prompt,
    ):
        """Test cost limit exceeded error."""
        from second_opinion.cli.main import execute_second_opinion

        # Setup mocks
        mock_sanitize_prompt.side_effect = lambda x, _: x

        mock_client = AsyncMock()
        mock_client.estimate_cost.return_value = Decimal("0.15")  # Exceeds limit
        mock_get_client_for_model.return_value = mock_client

        # Execute and verify error
        with pytest.raises(CLIError) as exc_info:
            await execute_second_opinion(
                prompt="Test prompt",
                primary_model="anthropic/claude-3-5-sonnet",
                comparison_models=["openai/gpt-4o"],
                cost_limit=0.10,  # Lower than estimated cost
            )

        assert "exceeds limit" in str(exc_info.value)

    @patch("second_opinion.cli.main.sanitize_prompt")
    @patch("second_opinion.cli.main.get_cost_guard")
    @patch("second_opinion.cli.main.get_client_for_model")
    async def test_execute_second_opinion_client_error(
        self, mock_get_client_for_model, mock_get_cost_guard, mock_sanitize_prompt
    ):
        """Test client error handling."""
        from second_opinion.cli.main import execute_second_opinion

        # Setup mocks
        mock_sanitize_prompt.side_effect = lambda x, _: x

        mock_client = AsyncMock()
        mock_client.estimate_cost.side_effect = Exception("API Error")
        mock_get_client_for_model.return_value = mock_client

        # Execute and verify error
        with pytest.raises(CLIError) as exc_info:
            await execute_second_opinion(
                prompt="Test prompt",
                primary_model="anthropic/claude-3-5-sonnet",
                comparison_models=["openai/gpt-4o"],
                cost_limit=0.10,
            )

        assert "Failed to estimate cost" in str(exc_info.value)


class TestOutputFormatting:
    """Test Rich output formatting."""

    @patch("second_opinion.cli.main.console")
    def test_display_results_basic(self, mock_console, mock_model_response):
        """Test basic results display."""
        from second_opinion.cli.main import display_results

        # Setup test data
        result = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [],
            "total_cost": Decimal("0.05"),
            "estimated_cost": Decimal("0.048"),
        }

        display_results(result)

        # Verify console.print was called for table and cost summary
        assert mock_console.print.call_count >= 2

    @patch("second_opinion.cli.main.console")
    def test_display_results_with_evaluations(self, mock_console, mock_model_response):
        """Test results display with evaluation scores."""
        from second_opinion.cli.main import display_results

        # Setup test data with evaluations
        mock_evaluation = MagicMock()
        mock_evaluation.quality_score = 8.5
        mock_evaluation.recommendation = "Primary model provides better quality"

        result = {
            "primary_response": mock_model_response,
            "comparison_responses": [mock_model_response],
            "evaluations": [mock_evaluation],
            "total_cost": Decimal("0.05"),
            "estimated_cost": Decimal("0.048"),
        }

        display_results(result)

        # Verify recommendations section is printed
        assert mock_console.print.call_count >= 3

        # Check that recommendation text appears in one of the calls
        recommendation_printed = any(
            "recommendation" in str(call).lower()
            for call in mock_console.print.call_args_list
        )
        assert recommendation_printed


class TestErrorHandling:
    """Test CLI error handling."""

    def test_cli_error_display(self, cli_runner):
        """Test that CLI errors are displayed properly."""
        with patch(
            "second_opinion.cli.main.ComparisonModelSelector"
        ) as mock_selector_class:
            mock_selector = MagicMock()
            mock_selector.select_models.side_effect = CLIError("Test error message", 2)
            mock_selector_class.return_value = mock_selector

            result = cli_runner.invoke(
                app,
                [
                    "second-opinion",
                    "--primary-model",
                    "anthropic/claude-3-5-sonnet",
                    "Test prompt",
                ],
            )

            assert result.exit_code == 2
            assert "Test error message" in result.stdout

    def test_keyboard_interrupt_handling(self, cli_runner):
        """Test keyboard interrupt handling."""
        with patch("second_opinion.cli.main.execute_second_opinion") as mock_execute:
            mock_execute.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(
                app,
                [
                    "second-opinion",
                    "--primary-model",
                    "anthropic/claude-3-5-sonnet",
                    "Test prompt",
                ],
            )

            assert result.exit_code == 130
            assert "cancelled by user" in result.stdout.lower()
