"""
CLI interface for Second Opinion.

This module provides the main CLI application using Typer, with support for:
- Primary model specification
- Comparison model selection (explicit or auto-selected)
- Cost limits and budget protection
- Rich output formatting
"""

import asyncio
import re
from datetime import UTC
from decimal import Decimal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from second_opinion.clients import get_client_for_model
from second_opinion.config.model_configs import model_config_manager
from second_opinion.config.settings import get_settings
from second_opinion.core.evaluator import (
    TaskComplexity,
    get_evaluator,
)
from second_opinion.core.models import EvaluationCriteria, Message, ModelRequest
from second_opinion.utils.cost_tracking import get_cost_guard
from second_opinion.utils.sanitization import SecurityContext, sanitize_prompt

# Initialize CLI components
app = typer.Typer(
    name="second-opinion",
    help="AI tool for getting second opinions and optimizing model usage",
    no_args_is_help=True,
)
console = Console()


class CLIError(Exception):
    """User-friendly CLI error."""

    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


def filter_think_tags(text: str) -> str:
    """
    Remove thinking tags from response text.

    Filters out <think>, <thinking>, and similar tags that some models use
    for internal reasoning that shouldn't be shown to users.

    Args:
        text: Response text potentially containing think tags

    Returns:
        Text with think tags removed
    """
    if not text or not text.strip():
        return text or ""

    # Patterns for various thinking tag formats
    think_patterns = [
        # Complete tags with content
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<thought>.*?</thought>",
        r"<reasoning>.*?</reasoning>",
        r"<internal>.*?</internal>",
        r"<analysis>.*?</analysis>",
        # Handle unclosed tags - remove everything from tag to end
        r"<think>.*?(?=\n\n|$)",
        r"<thinking>.*?(?=\n\n|$)",
        r"<thought>.*?(?=\n\n|$)",
        r"<reasoning>.*?(?=\n\n|$)",
        r"<internal>.*?(?=\n\n|$)",
        r"<analysis>.*?(?=\n\n|$)",
        # Handle cases where tags are at the very beginning or end
        r"^\s*</?think[^>]*>.*?(?=\n[A-Za-z]|\n\n|$)",
        r"^\s*</?thinking[^>]*>.*?(?=\n[A-Za-z]|\n\n|$)",
    ]

    filtered_text = text
    for pattern in think_patterns:
        # Remove matched patterns
        filtered_text = re.sub(
            pattern, "", filtered_text, flags=re.DOTALL | re.IGNORECASE
        )

    # Clean up extra whitespace left by removed tags
    filtered_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", filtered_text)  # Multiple newlines
    filtered_text = re.sub(r"^\s*\n+", "", filtered_text)  # Leading newlines
    filtered_text = re.sub(r"\n+\s*$", "", filtered_text)  # Trailing newlines
    filtered_text = filtered_text.strip()

    return filtered_text


class ComparisonModelSelector:
    """Handles comparison model selection logic with priority hierarchy."""

    def __init__(self):
        self.settings = get_settings()
        self.model_config = model_config_manager

    def select_models(
        self,
        primary_model: str,
        tool_name: str = "second_opinion",
        explicit_models: list[str] | None = None,
        task_complexity: TaskComplexity | None = None,
        max_models: int = 2,
    ) -> list[str]:
        """Select comparison models based on priority hierarchy."""

        # Priority 1: Explicit CLI selection
        if explicit_models:
            return self._validate_and_filter_models(explicit_models, primary_model)

        # Priority 2: Tool configuration
        config_models = self._get_config_comparison_models(tool_name, primary_model)
        if config_models:
            return config_models[:max_models]

        # Priority 3: Smart auto-selection based on task complexity
        if task_complexity:
            return self._smart_select(primary_model, task_complexity, max_models)

        # Priority 4: Fallback to tier-based selection
        return self._tier_based_select(primary_model, max_models)

    def _validate_and_filter_models(
        self, models: list[str], primary_model: str
    ) -> list[str]:
        """Validate and filter comparison models."""
        valid_models = []

        for model in models:
            # Remove duplicates of primary model
            if model == primary_model:
                console.print(
                    f"[yellow]Warning:[/yellow] Comparison model '{model}' is the same as primary model, skipping"
                )
                continue

            # Basic model name validation
            if not self._is_valid_model_name(model):
                raise CLIError(f"Invalid comparison model name: {model}")

            valid_models.append(model)

        if not valid_models:
            raise CLIError("No valid comparison models specified")

        return valid_models

    def _get_config_comparison_models(
        self, tool_name: str, primary_model: str
    ) -> list[str]:
        """Get comparison models from configuration."""
        try:
            return self.model_config.get_comparison_models(tool_name, primary_model)
        except Exception:
            return []

    def _smart_select(
        self, primary_model: str, task_complexity: TaskComplexity, max_models: int
    ) -> list[str]:
        """Smart comparison model selection based on task complexity and model tier."""
        try:
            from ..clients import detect_model_provider

            config = self.model_config.config
            if not config:
                return self._get_default_models(primary_model, max_models)

            # Check if primary is local or cloud
            primary_provider = detect_model_provider(primary_model)

            # Get primary model tier
            primary_tier = self._get_model_tier(primary_model, config)

            # Selection strategy based on tier and complexity
            candidates = []

            if primary_tier == "budget":
                # Budget models compare against mid-range for quality upgrade potential
                candidates = config.model_tiers.mid_range[:]
            elif primary_tier == "mid_range":
                # Mid-range compares against budget (cost savings) and premium (quality)
                candidates = (
                    config.model_tiers.budget[:1]  # One budget option
                    + config.model_tiers.premium[:1]  # One premium option
                )
            elif primary_tier == "premium":
                # Premium compares against mid-range and other premium
                candidates = (
                    config.model_tiers.mid_range[:1]
                    + [m for m in config.model_tiers.premium if m != primary_model][:1]
                )
            else:
                # Unknown tier, use default strategy
                return self._get_default_models(primary_model, max_models)

            # Add local model cost optimization for cloud primaries
            if primary_provider == "openrouter" and max_models > len(candidates):
                # Include local model as cost-effective alternative
                local_alternatives = ["qwen3-4b-mlx", "codestral-22b-v0.1"]
                for local_model in local_alternatives:
                    if local_model != primary_model and len(candidates) < max_models:
                        candidates.insert(0, local_model)  # Prioritize cost savings
                        break

            # For expert/complex tasks, prefer higher-tier models
            if task_complexity in [TaskComplexity.EXPERT, TaskComplexity.COMPLEX]:
                # Keep local models for cost analysis but prioritize capable models
                local_models = [
                    m for m in candidates if detect_model_provider(m) == "lmstudio"
                ]
                cloud_models = [
                    m for m in candidates if detect_model_provider(m) == "openrouter"
                ]

                cloud_models = sorted(
                    cloud_models,
                    key=lambda m: self._get_model_capability_score(m),
                    reverse=True,
                )

                # Combine: prioritize capable models but keep one local option
                candidates = cloud_models + local_models[:1]

            # Filter out primary model and limit results
            candidates = [m for m in candidates if m != primary_model]
            return candidates[:max_models]

        except Exception:
            return self._get_default_models(primary_model, max_models)

    def _tier_based_select(self, primary_model: str, max_models: int) -> list[str]:
        """Fallback tier-based selection."""
        return self._get_default_models(primary_model, max_models)

    def _get_model_tier(self, model: str, model_config) -> str:
        """Get the tier of a model."""
        for tier, models in {
            "budget": model_config.model_tiers.budget,
            "mid_range": model_config.model_tiers.mid_range,
            "premium": model_config.model_tiers.premium,
            "reasoning": model_config.model_tiers.reasoning,
        }.items():
            if model in models:
                return tier
        return "unknown"

    def _get_model_capability_score(self, model: str) -> int:
        """Get a rough capability score for model ranking."""
        # Simple heuristic based on model name patterns
        model_lower = model.lower()
        if "o1" in model_lower or "opus" in model_lower:
            return 4  # Highest capability
        elif "gpt-4" in model_lower or "claude-3-5" in model_lower:
            return 3  # High capability
        elif "gemini-pro" in model_lower or "claude-3" in model_lower:
            return 2  # Medium capability
        elif "haiku" in model_lower or "mini" in model_lower or "flash" in model_lower:
            return 1  # Lower capability
        else:
            return 0  # Unknown

    def _get_default_models(self, primary_model: str, max_models: int) -> list[str]:
        """Get default comparison models when configuration is unavailable."""
        from ..clients import detect_model_provider

        # Determine if primary is local or cloud
        primary_provider = detect_model_provider(primary_model)

        if primary_provider == "lmstudio":
            # Primary is local, compare against cloud alternatives
            default_models = [
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-4o",
                "anthropic/claude-3-haiku",
                "openai/gpt-4o-mini",
            ]
        else:
            # Primary is cloud, include local cost-effective alternatives
            default_models = [
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-4o",
                "qwen3-4b-mlx",  # Cost-effective local alternative
                "anthropic/claude-3-haiku",
                "openai/gpt-4o-mini",
            ]

        # Filter out primary model
        candidates = [m for m in default_models if m != primary_model]
        return candidates[:max_models]

    def _is_valid_model_name(self, model: str) -> bool:
        """Basic validation of model name format."""
        # Allow both provider/model format and direct model names (for local models)
        return (
            "/" in model and len(model.split("/")) == 2
        ) or self._is_local_model_name(  # Provider/model format
            model
        )  # Local model format

    def _is_local_model_name(self, model: str) -> bool:
        """Check if this looks like a local model name."""
        # Local models typically don't have a provider prefix and may contain specific patterns
        local_patterns = [
            "mlx",  # MLX models
            "qwen",  # Qwen models
            "llama",  # Llama models
            "mistral",  # Mistral models
            "codestral",  # Codestral models
            "devstral",  # Devstral models
        ]
        model_lower = model.lower()
        return (
            any(pattern in model_lower for pattern in local_patterns)
            and "/" not in model
        )


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        # Check if there's a running event loop
        try:
            asyncio.get_running_loop()
            # If we're already in an async context, create a new loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop, use asyncio.run
            return asyncio.run(coro)
    except RuntimeError:
        # No loop exists, create one
        return asyncio.run(coro)


def handle_cli_error(func):
    """Decorator to handle CLI errors gracefully."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            console.print(f"[red]Error:[/red] {e.message}")
            raise typer.Exit(e.exit_code) from None
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            raise typer.Exit(130) from None
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {str(e)}")
            console.print("[dim]Use --help for usage information[/dim]")
            raise typer.Exit(1) from e

    return wrapper


async def execute_second_opinion(
    prompt: str,
    primary_model: str,
    comparison_models: list[str],
    cost_limit: float,
    context: str | None = None,
    existing_response: str | None = None,
    evaluator_model: str | None = None,
) -> dict:
    """Execute the second opinion operation."""

    # Initialize components
    cost_guard = get_cost_guard()
    evaluator = get_evaluator()

    # Detect task complexity for evaluation insights
    task_complexity = await evaluator.classify_task_complexity(prompt)

    # Sanitize input
    sanitized_prompt = sanitize_prompt(prompt, SecurityContext.USER_PROMPT)
    sanitized_context = (
        sanitize_prompt(context, SecurityContext.USER_PROMPT) if context else None
    )

    # Create request
    messages = [Message(role="user", content=sanitized_prompt)]
    if sanitized_context:
        messages.insert(
            0, Message(role="system", content=f"Context: {sanitized_context}")
        )

    request = ModelRequest(
        model=primary_model, messages=messages, max_tokens=1000, temperature=0.1
    )

    # Check budget for models (skip primary if existing response provided)
    total_estimated_cost = Decimal("0")
    models_to_run = (
        comparison_models if existing_response else [primary_model] + comparison_models
    )

    for model in models_to_run:
        try:
            client = get_client_for_model(model)
            estimated_cost = await client.estimate_cost(
                request.model_copy(update={"model": model})
            )
            total_estimated_cost += estimated_cost
        except Exception as e:
            raise CLIError(
                f"Failed to estimate cost for model {model}: {str(e)}"
            ) from e

    if total_estimated_cost > Decimal(str(cost_limit)):
        raise CLIError(
            f"Estimated total cost ${total_estimated_cost:.4f} exceeds limit ${cost_limit:.2f}"
        )

    # Reserve budget
    reservation = await cost_guard.check_and_reserve_budget(
        total_estimated_cost, "second_opinion", primary_model
    )

    try:
        # Execute primary model request or use existing response
        if existing_response:
            # Sanitize the existing response
            sanitized_existing_response = sanitize_prompt(
                existing_response, SecurityContext.USER_PROMPT
            )

            # Create a mock ModelResponse for existing response
            from datetime import datetime
            from uuid import uuid4

            from second_opinion.core.models import ModelResponse, TokenUsage

            primary_response = ModelResponse(
                content=sanitized_existing_response,
                model=primary_model,
                usage=TokenUsage(
                    input_tokens=0,  # No tokens used for existing response
                    output_tokens=0,
                    total_tokens=0,
                ),
                cost_estimate=Decimal("0.00"),  # No cost for existing response
                provider="existing",
                request_id=str(uuid4()),
                timestamp=datetime.now(UTC),
            )
        else:
            # Execute normal primary model request
            primary_client = get_client_for_model(primary_model)
            primary_response = await primary_client.complete(request)

        # Execute comparison model requests
        comparison_responses = []
        for model in comparison_models:
            try:
                client = get_client_for_model(model)
                response = await client.complete(
                    request.model_copy(update={"model": model})
                )
                comparison_responses.append(response)
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to get response from {model}: {str(e)}"
                )
                continue

        if not comparison_responses:
            raise CLIError("No comparison responses were successful")

        # Evaluate responses
        evaluations = []
        evaluation_criteria = EvaluationCriteria(
            accuracy_weight=0.3,
            completeness_weight=0.3,
            clarity_weight=0.2,
            usefulness_weight=0.2,
        )

        for comp_response in comparison_responses:
            try:
                evaluation = await evaluator.compare_responses(
                    primary_response,
                    comp_response,
                    original_task=prompt,
                    criteria=evaluation_criteria,
                    evaluator_model=evaluator_model,
                )
                evaluations.append(evaluation)
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to evaluate response from {comp_response.model}: {str(e)}"
                )
                continue

        # Calculate actual cost (existing response has zero cost)
        actual_cost = primary_response.cost_estimate + sum(
            r.cost_estimate for r in comparison_responses
        )

        # Record actual cost
        await cost_guard.record_actual_cost(
            reservation.reservation_id, actual_cost, primary_model, "second_opinion"
        )

        return {
            "primary_response": primary_response,
            "comparison_responses": comparison_responses,
            "evaluations": evaluations,
            "total_cost": actual_cost,
            "estimated_cost": total_estimated_cost,
            "task_complexity": task_complexity,
        }

    except Exception:
        # Release reservation on error
        await cost_guard.record_actual_cost(
            reservation.reservation_id, Decimal("0"), primary_model, "second_opinion"
        )
        raise


@app.command("second-opinion")
@handle_cli_error
def second_opinion_command(
    prompt: str = typer.Argument(..., help="The prompt to get a second opinion on"),
    primary_model: str = typer.Option(
        ..., "--primary-model", "-p", help="Primary model to get second opinion for"
    ),
    comparison_model: list[str] | None = typer.Option(
        None,
        "--comparison-model",
        "-c",
        help="Specific model(s) to compare against (can be used multiple times)",
    ),
    cost_limit: float | None = typer.Option(
        None, "--cost-limit", help="Maximum cost for the operation"
    ),
    context: str | None = typer.Option(
        None, "--context", help="Additional context for better comparisons"
    ),
    max_comparisons: int = typer.Option(
        2, "--max-comparisons", help="Maximum number of comparison models to use"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show full responses instead of truncated summaries",
    ),
    existing_response: str | None = typer.Option(
        None,
        "--existing-response",
        help="Provide existing primary model response to save API calls",
    ),
    evaluator_model: str | None = typer.Option(
        None,
        "--evaluator-model",
        help="Model to use for evaluation (defaults to primary model)",
    ),
):
    """Get a second opinion on a prompt using multiple models."""

    # Determine cost limit using hierarchy: CLI flag > model config > settings default
    if cost_limit is None:
        try:
            # Try to get from model configuration first
            from second_opinion.config.model_configs import model_config_manager

            tool_config = model_config_manager.get_tool_config("second_opinion")
            cost_limit = float(tool_config.cost_limit_per_request)
        except Exception:
            # Fall back to settings default
            settings = get_settings()
            cost_limit = float(settings.cost_management.default_per_request_limit)

    # Show operation info
    existing_info = (
        "\nUsing existing response (no API call)" if existing_response else ""
    )
    console.print(
        Panel.fit(
            f"[bold]Second Opinion Analysis[/bold]\n"
            f"Primary Model: {primary_model}{existing_info}\n"
            f"Cost Limit: ${cost_limit:.2f}",
            border_style="blue",
        )
    )

    # Initialize model selector
    selector = ComparisonModelSelector()

    # Select comparison models (task complexity will be detected in execute_second_opinion)
    selected_models = selector.select_models(
        primary_model=primary_model,
        tool_name="second_opinion",
        explicit_models=comparison_model,
        task_complexity=None,  # Will be detected in async function
        max_models=max_comparisons,
    )

    # Get evaluator model from config if not specified
    if evaluator_model is None:
        try:
            from second_opinion.config.model_configs import model_config_manager

            config_evaluator = model_config_manager.get_tool_config(
                "second_opinion"
            ).evaluator_model
            if config_evaluator:
                evaluator_model = config_evaluator
        except Exception:
            # Config not available or no evaluator specified, will use primary model
            pass

    # Show model selection
    if comparison_model:
        model_source = "user specified"
    else:
        model_source = "auto-selected"

    console.print(
        f"[dim]Comparison models: {', '.join(selected_models)} ({model_source})[/dim]"
    )

    # Execute operation
    with console.status("[bold blue]Getting responses from models..."):
        result = run_async(
            execute_second_opinion(
                prompt=prompt,
                primary_model=primary_model,
                comparison_models=selected_models,
                cost_limit=cost_limit,
                context=context,
                existing_response=existing_response,
                evaluator_model=evaluator_model,
            )
        )

    # Display results
    display_results(result, verbose=verbose)


def display_results(result: dict, verbose: bool = False):
    """Display comparison results with rich formatting."""
    evaluations = result["evaluations"]
    total_cost = result["total_cost"]

    if verbose:
        # Verbose mode: Show full responses in separate sections
        _display_verbose_results(result)
    else:
        # Summary mode: Show truncated responses in table
        _display_summary_results(result)

    # Cost summary and task info (always shown)
    task_complexity = result.get("task_complexity")
    complexity_text = (
        f"\n[dim]Task Complexity: {task_complexity.value}[/dim]"
        if task_complexity
        else ""
    )

    cost_panel = Panel.fit(
        f"[bold]Total Cost:[/bold] ${total_cost:.4f}\n"
        f"[dim]Estimated: ${result['estimated_cost']:.4f}[/dim]{complexity_text}",
        title="Cost Summary",
        border_style="green",
    )
    console.print(cost_panel)

    # Show recommendations if available (deduplicated)
    if evaluations:
        console.print("\n[bold]Recommendations:[/bold]")
        all_recommendations = []
        for evaluation in evaluations:
            if hasattr(evaluation, "recommendations") and evaluation.recommendations:
                all_recommendations.extend(evaluation.recommendations)

        # Deduplicate recommendations
        unique_recommendations = list(
            dict.fromkeys(all_recommendations)
        )  # Preserves order

        if unique_recommendations:
            for recommendation in unique_recommendations:
                console.print(f"• {recommendation}")
        else:
            console.print("• No specific recommendations available")


def _display_summary_results(result: dict):
    """Display results in summary table format (original behavior)."""
    primary_response = result["primary_response"]
    comparison_responses = result["comparison_responses"]
    evaluations = result["evaluations"]

    # Create comparison table
    table = Table(
        title="Second Opinion Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Response", style="white", ratio=3)
    table.add_column("Cost", style="green", justify="right")
    table.add_column("Quality", style="yellow", justify="center")

    # Add primary response (with think tag filtering)
    filtered_primary = filter_think_tags(primary_response.content)
    table.add_row(
        f"{primary_response.model} (primary)",
        (
            filtered_primary[:200] + "..."
            if len(filtered_primary) > 200
            else filtered_primary
        ),
        f"${primary_response.cost_estimate:.4f}",
        "—",
    )

    # Add comparison responses with evaluations
    for i, comp_response in enumerate(comparison_responses):
        quality_score = "—"
        if i < len(evaluations) and evaluations[i]:
            eval_result = evaluations[i]
            if hasattr(eval_result, "quality_score"):
                quality_score = f"{eval_result.quality_score:.1f}/10"

        # Filter think tags from comparison response
        filtered_comparison = filter_think_tags(comp_response.content)

        table.add_row(
            comp_response.model,
            (
                filtered_comparison[:200] + "..."
                if len(filtered_comparison) > 200
                else filtered_comparison
            ),
            f"${comp_response.cost_estimate:.4f}",
            quality_score,
        )

    console.print(table)


def _display_verbose_results(result: dict):
    """Display results in verbose format with full responses."""
    primary_response = result["primary_response"]
    comparison_responses = result["comparison_responses"]
    evaluations = result["evaluations"]

    # Display primary response (with think tag filtering)
    console.print(
        Panel.fit(
            f"[bold cyan]{primary_response.model} (Primary Model)[/bold cyan]\n"
            f"[dim]Cost: ${primary_response.cost_estimate:.4f}[/dim]",
            title="Primary Response",
            border_style="blue",
        )
    )
    filtered_primary_verbose = filter_think_tags(primary_response.content)
    console.print(f"\n{filtered_primary_verbose}\n")

    # Display comparison responses
    for i, comp_response in enumerate(comparison_responses):
        quality_info = ""
        if i < len(evaluations) and evaluations[i]:
            eval_result = evaluations[i]
            if hasattr(eval_result, "quality_score"):
                quality_info = f" | Quality: {eval_result.quality_score:.1f}/10"

        console.print(
            Panel.fit(
                f"[bold cyan]{comp_response.model}[/bold cyan]\n"
                f"[dim]Cost: ${comp_response.cost_estimate:.4f}{quality_info}[/dim]",
                title=f"Comparison Response {i + 1}",
                border_style="green",
            )
        )
        # Filter think tags from comparison response in verbose mode
        filtered_comparison_verbose = filter_think_tags(comp_response.content)
        console.print(f"\n{filtered_comparison_verbose}\n")


# Entry point is handled by pyproject.toml script configuration
