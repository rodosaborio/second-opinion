"""
CLI interface for Second Opinion.

This module provides the main CLI application using Typer, with support for:
- Primary model specification
- Comparison model selection (explicit or auto-selected)
- Cost limits and budget protection
- Rich output formatting
"""

import asyncio
from decimal import Decimal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from second_opinion.clients import create_client
from second_opinion.config.model_configs import model_config_manager
from second_opinion.config.settings import get_settings
from second_opinion.core.evaluator import TaskComplexity, get_evaluator
from second_opinion.core.models import Message, ModelRequest
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
            config = self.model_config.config
            if not config:
                return self._get_default_models(primary_model, max_models)

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

            # For expert/complex tasks, prefer higher-tier models
            if task_complexity in [TaskComplexity.EXPERT, TaskComplexity.COMPLEX]:
                candidates = sorted(
                    candidates,
                    key=lambda m: self._get_model_capability_score(m),
                    reverse=True,
                )

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
        default_models = [
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
        ]

        # Filter out primary model
        candidates = [m for m in default_models if m != primary_model]
        return candidates[:max_models]

    def _is_valid_model_name(self, model: str) -> bool:
        """Basic validation of model name format."""
        return "/" in model and len(model.split("/")) == 2


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
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
) -> dict:
    """Execute the second opinion operation."""

    # Initialize components
    cost_guard = get_cost_guard()
    evaluator = get_evaluator()

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

    # Check budget for all models
    total_estimated_cost = Decimal("0")
    all_models = [primary_model] + comparison_models

    for model in all_models:
        try:
            client = create_client("openrouter")
            estimated_cost = await client.estimate_cost(request._replace(model=model))
            total_estimated_cost += estimated_cost
        except Exception as e:
            raise CLIError(f"Failed to estimate cost for model {model}: {str(e)}") from e

    if total_estimated_cost > Decimal(str(cost_limit)):
        raise CLIError(
            f"Estimated total cost ${total_estimated_cost:.4f} exceeds limit ${cost_limit:.2f}"
        )

    # Reserve budget
    reservation = await cost_guard.check_and_reserve_budget(
        total_estimated_cost, "second_opinion", primary_model
    )

    try:
        # Execute primary model request
        primary_client = create_client("openrouter")
        primary_response = await primary_client.complete(request)

        # Execute comparison model requests
        comparison_responses = []
        for model in comparison_models:
            try:
                client = create_client("openrouter")
                response = await client.complete(request._replace(model=model))
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
        for comp_response in comparison_responses:
            try:
                evaluation = await evaluator.compare_responses(
                    primary_response,
                    comp_response,
                    {
                        "accuracy": 0.3,
                        "completeness": 0.3,
                        "clarity": 0.2,
                        "usefulness": 0.2,
                    },
                )
                evaluations.append(evaluation)
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to evaluate response from {comp_response.model}: {str(e)}"
                )
                continue

        # Calculate actual cost
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
        }

    except Exception:
        # Release reservation on error
        await cost_guard.record_actual_cost(
            reservation.reservation_id, Decimal("0"), primary_model, "second_opinion"
        )
        raise


@app.command()
@handle_cli_error
def main(
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
    cost_limit: float = typer.Option(
        0.10, "--cost-limit", help="Maximum cost for the operation"
    ),
    context: str | None = typer.Option(
        None, "--context", help="Additional context for better comparisons"
    ),
    max_comparisons: int = typer.Option(
        2, "--max-comparisons", help="Maximum number of comparison models to use"
    ),
):
    """Get a second opinion on a prompt using multiple models."""

    # Show operation info
    console.print(
        Panel.fit(
            f"[bold]Second Opinion Analysis[/bold]\n"
            f"Primary Model: {primary_model}\n"
            f"Cost Limit: ${cost_limit:.2f}",
            border_style="blue",
        )
    )

    # Initialize model selector
    selector = ComparisonModelSelector()

    # Select comparison models
    selected_models = selector.select_models(
        primary_model=primary_model,
        tool_name="second_opinion",
        explicit_models=comparison_model,
        task_complexity=None,  # TODO: Add task complexity detection
        max_models=max_comparisons,
    )

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
            )
        )

    # Display results
    display_results(result)


def display_results(result: dict):
    """Display comparison results with rich formatting."""
    primary_response = result["primary_response"]
    comparison_responses = result["comparison_responses"]
    evaluations = result["evaluations"]
    total_cost = result["total_cost"]

    # Create comparison table
    table = Table(
        title="Second Opinion Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Response", style="white", ratio=3)
    table.add_column("Cost", style="green", justify="right")
    table.add_column("Quality", style="yellow", justify="center")

    # Add primary response
    table.add_row(
        f"{primary_response.model} (primary)",
        (
            primary_response.content[:200] + "..."
            if len(primary_response.content) > 200
            else primary_response.content
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

        table.add_row(
            comp_response.model,
            (
                comp_response.content[:200] + "..."
                if len(comp_response.content) > 200
                else comp_response.content
            ),
            f"${comp_response.cost_estimate:.4f}",
            quality_score,
        )

    console.print(table)

    # Cost summary
    cost_panel = Panel.fit(
        f"[bold]Total Cost:[/bold] ${total_cost:.4f}\n"
        f"[dim]Estimated: ${result['estimated_cost']:.4f}[/dim]",
        title="Cost Summary",
        border_style="green",
    )
    console.print(cost_panel)

    # Show recommendations if available
    if evaluations:
        console.print("\n[bold]Recommendations:[/bold]")
        for evaluation in evaluations:
            if hasattr(evaluation, "recommendation") and evaluation.recommendation:
                console.print(f"• {evaluation.recommendation}")


# Entry point is handled by pyproject.toml script configuration
