"""
Model Benchmark MCP tool implementation.

This module provides comprehensive model benchmarking capabilities across different
task types, helping users understand model performance patterns and make informed
decisions about model selection for specific use cases.
"""

import logging
from decimal import Decimal
from typing import Any

from ...clients import detect_model_provider
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, Message, ModelRequest, TaskComplexity
from ...orchestration import get_conversation_orchestrator
from ...orchestration.types import StorageContext
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import BudgetPeriod, get_cost_guard
from ...utils.sanitization import (
    SecurityContext,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)

logger = logging.getLogger(__name__)

# Predefined benchmark task categories
BENCHMARK_TASKS = {
    "coding": [
        "Write a Python function to calculate the factorial of a number using recursion",
        "Debug this JavaScript code: function add(a, b) { return a + b; console.log(result); }",
        "Implement a binary search algorithm in Python with proper error handling",
        "Write a SQL query to find the top 5 customers by total order value",
        "Create a React component that displays a list of users with search functionality",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A farmer has 17 sheep, and all but 9 die. How many sheep are left?",
        "What comes next in this sequence: 2, 6, 12, 20, 30, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Three friends split a restaurant bill. If the bill was $75 and they left a 20% tip, how much did each person pay?",
    ],
    "creative": [
        "Write a short story about a robot who discovers emotions for the first time",
        "Create a marketing slogan for a new eco-friendly water bottle",
        "Compose a haiku about artificial intelligence",
        "Write a product description for a smart home device that doesn't exist yet",
        "Create a dialogue between two characters who are meeting for the first time",
    ],
    "analysis": [
        "Analyze the pros and cons of remote work versus office work",
        "What are the key factors that contribute to successful team collaboration?",
        "Compare and contrast renewable energy sources: solar, wind, and hydroelectric",
        "Analyze the impact of social media on modern communication patterns",
        "What are the main challenges facing small businesses in the digital age?",
    ],
    "explanation": [
        "Explain quantum computing to a 12-year-old",
        "How does machine learning work in simple terms?",
        "Explain the difference between weather and climate",
        "What is blockchain technology and how does it work?",
        "Explain photosynthesis and why it's important for life on Earth",
    ],
}


async def model_benchmark_tool(
    models: list[str],
    task_types: list[str] | None = None,
    sample_size: int = 3,
    evaluation_criteria: str = "comprehensive",
    cost_limit: float | None = None,
    session_id: str | None = None,
) -> str:
    """
    Benchmark multiple models across different task types for comprehensive performance analysis.

    This tool systematically tests models across various task categories to provide insights
    into their strengths, weaknesses, and optimal use cases. Perfect for model selection,
    performance analysis, and understanding model capabilities across different domains.

    Args:
        models: List of models to benchmark. Format:
               - Cloud models: "anthropic/claude-3-5-sonnet", "openai/gpt-4o", etc.
               - Local models: "qwen3-4b-mlx", "codestral-22b-v0.1", etc.
        task_types: Task categories to test (default: ["coding", "reasoning", "creative"]).
                   Available: "coding", "reasoning", "creative", "analysis", "explanation"
        sample_size: Number of tasks per category to test (default: 3, max: 5)
        evaluation_criteria: How to evaluate responses:
                           - "comprehensive": Balanced accuracy, clarity, usefulness
                           - "accuracy": Focus on correctness and precision
                           - "creativity": Emphasize originality and engagement
                           - "speed": Optimize for response time and efficiency
        cost_limit: Maximum cost for this operation in USD (default: $2.00)
        session_id: Session ID for conversation tracking

    Returns:
        Comprehensive benchmark report with:
        - Model performance scores across task categories
        - Detailed strengths and weaknesses analysis
        - Cost efficiency and speed comparisons
        - Optimal use case recommendations
        - Statistical significance indicators

    USAGE EXAMPLES:

        # Basic model comparison across core task types
        result = await model_benchmark_tool(
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"],
            task_types=["coding", "reasoning"],
            sample_size=3
        )

        # Comprehensive benchmark with all task categories
        result = await model_benchmark_tool(
            models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o-mini", "google/gemini-pro-1.5"],
            task_types=["coding", "reasoning", "creative", "analysis", "explanation"],
            sample_size=2,
            evaluation_criteria="comprehensive"
        )

        # Cost-focused benchmark for budget optimization
        result = await model_benchmark_tool(
            models=["qwen3-4b-mlx", "anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
            task_types=["coding", "analysis"],
            evaluation_criteria="speed",
            cost_limit=1.0
        )
    """
    try:
        # Input validation and sanitization
        if not models or len(models) < 2:
            return "❌ **Invalid Models**: Please provide at least 2 models for benchmarking.\\n\\nBenchmarking requires multiple models to compare performance patterns."

        if len(models) > 8:
            return "❌ **Too Many Models**: Maximum allowed is 8 models per benchmark.\\n\\nThis limit prevents excessive costs and ensures meaningful analysis."

        # Validate sample size
        if sample_size < 1 or sample_size > 5:
            return "❌ **Invalid Sample Size**: Must be between 1 and 5 tasks per category.\\n\\nRecommended: 3 tasks per category for balanced coverage and cost."

        # Validate and normalize model names
        try:
            models = [validate_model_name(model) for model in models]
        except Exception as e:
            suggestions = _get_model_name_suggestions(str(e))
            return f"❌ **Invalid Model Name**: {str(e)}\\n\\n**Suggested formats:**\\n{suggestions}"

        # Validate task types
        if task_types is None:
            task_types = [
                "coding",
                "reasoning",
                "creative",
            ]  # Default balanced selection

        available_task_types = list(BENCHMARK_TASKS.keys())
        invalid_types = [t for t in task_types if t not in available_task_types]
        if invalid_types:
            return f"❌ **Invalid Task Types**: {', '.join(invalid_types)}\\n\\n**Available types**: {', '.join(available_task_types)}"

        # Validate evaluation criteria
        valid_criteria = ["comprehensive", "accuracy", "creativity", "speed"]
        if evaluation_criteria not in valid_criteria:
            return f"❌ **Invalid Evaluation Criteria**: '{evaluation_criteria}'\\n\\n**Valid options**: {', '.join(valid_criteria)}"

        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            cost_limit_decimal = Decimal(
                "2.00"
            )  # Higher default for comprehensive benchmarking

        logger.info(
            f"Starting model_benchmark: models={len(models)}, task_types={task_types}, "
            f"sample_size={sample_size}, evaluation_criteria={evaluation_criteria}"
        )

        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()

        # Select benchmark tasks
        selected_tasks = []
        for task_type in task_types:
            type_tasks = BENCHMARK_TASKS[task_type][:sample_size]
            for task in type_tasks:
                selected_tasks.append({"type": task_type, "task": task})

        total_tasks = len(selected_tasks) * len(models)
        logger.info(
            f"Selected {len(selected_tasks)} tasks across {len(task_types)} categories, testing {len(models)} models = {total_tasks} total evaluations"
        )

        # Estimate costs
        estimated_cost = Decimal("0.0")

        # Response generation costs
        for model in models:
            try:
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                # Estimate based on average task complexity
                avg_request = ModelRequest(
                    model=model,
                    messages=[
                        Message(role="user", content="Average benchmark task...")
                    ],
                    max_tokens=1500,
                    temperature=0.1,
                )
                model_cost_per_task = await client.estimate_cost(avg_request)
                estimated_cost += model_cost_per_task * len(selected_tasks)
            except Exception as e:
                logger.warning(f"Failed to estimate cost for {model}: {e}")
                estimated_cost += Decimal("0.08") * len(
                    selected_tasks
                )  # Conservative estimate

        # Evaluation costs (pairwise comparisons per task)
        num_comparisons_per_task = len(models) * (len(models) - 1) // 2
        evaluation_cost_per_comparison = Decimal("0.02")
        estimated_cost += (
            evaluation_cost_per_comparison
            * num_comparisons_per_task
            * len(selected_tasks)
        )

        logger.info(f"Estimated total cost: ${estimated_cost:.4f}")

        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost,
                "model_benchmark",
                models[0],
                per_request_override=cost_limit_decimal,
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\\n\\nEstimated cost: ${estimated_cost:.4f}\\nCost limit: ${cost_limit_decimal:.2f}\\n\\nConsider reducing sample_size or number of models."

        # Generate responses from all models for all tasks
        logger.info("Generating responses from all models across benchmark tasks")
        actual_cost = Decimal("0.0")
        benchmark_results = []

        for task_data in selected_tasks:
            task_type = task_data["type"]
            task = task_data["task"]
            clean_task = sanitize_prompt(task, SecurityContext.USER_PROMPT)

            # Classify task complexity
            task_complexity = TaskComplexity.MODERATE  # Default
            try:
                task_complexity = await evaluator.classify_task_complexity(clean_task)
            except Exception as e:
                logger.warning(f"Failed to classify task complexity: {e}")

            model_responses = []

            # Generate responses from each model
            for model in models:
                try:
                    provider = detect_model_provider(model)
                    client = create_client_from_config(provider)
                    request = ModelRequest(
                        model=model,
                        messages=[Message(role="user", content=clean_task)],
                        max_tokens=1500,
                        temperature=0.1,
                    )
                    response = await client.complete(request)
                    model_responses.append(response)
                    actual_cost += response.cost_estimate
                    logger.info(
                        f"Generated {task_type} response from {model}: ${response.cost_estimate:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate response from {model} for {task_type}: {e}"
                    )
                    # Create error response
                    from ...core.models import ModelResponse, TokenUsage

                    error_response = ModelResponse(
                        content=f"Error: Failed to generate response from {model}: {str(e)}",
                        model=model,
                        usage=TokenUsage(
                            input_tokens=0, output_tokens=0, total_tokens=0
                        ),
                        cost_estimate=Decimal("0.0"),
                        provider=detect_model_provider(model),
                    )
                    model_responses.append(error_response)

            # Perform pairwise evaluation for this task
            evaluation_results = []
            evaluation_cost = Decimal("0.0")

            try:
                # Set evaluation criteria based on task type and user preference
                if evaluation_criteria == "accuracy":
                    criteria = EvaluationCriteria(
                        accuracy_weight=0.5,
                        completeness_weight=0.3,
                        clarity_weight=0.1,
                        usefulness_weight=0.1,
                    )
                elif evaluation_criteria == "creativity":
                    criteria = EvaluationCriteria(
                        accuracy_weight=0.1,
                        completeness_weight=0.2,
                        clarity_weight=0.2,
                        usefulness_weight=0.5,
                    )
                elif evaluation_criteria == "speed":
                    criteria = EvaluationCriteria(
                        accuracy_weight=0.3,
                        completeness_weight=0.2,
                        clarity_weight=0.3,
                        usefulness_weight=0.2,
                    )
                else:  # comprehensive
                    criteria = EvaluationCriteria(
                        accuracy_weight=0.3,
                        completeness_weight=0.25,
                        clarity_weight=0.25,
                        usefulness_weight=0.2,
                    )

                # Use cost-effective model for evaluation
                evaluator_model = "openai/gpt-4o-mini"

                # Perform pairwise comparisons
                for i in range(len(model_responses)):
                    for j in range(i + 1, len(model_responses)):
                        try:
                            response_a = model_responses[i]
                            response_b = model_responses[j]

                            # Skip error responses
                            if response_a.content.startswith(
                                "Error:"
                            ) or response_b.content.startswith("Error:"):
                                continue

                            comparison_result = await evaluator.compare_responses(
                                response_a,
                                response_b,
                                original_task=clean_task,
                                criteria=criteria,
                                evaluator_model=evaluator_model,
                            )

                            evaluation_results.append(
                                {
                                    "model_a": response_a.model,
                                    "model_b": response_b.model,
                                    "winner": comparison_result.winner,
                                    "score_a": comparison_result.score_a,
                                    "score_b": comparison_result.score_b,
                                    "overall_score": comparison_result.overall_score,
                                    "reasoning": comparison_result.reasoning,
                                }
                            )

                            evaluation_cost += Decimal("0.01")

                        except Exception as e:
                            logger.warning(
                                f"Evaluation failed for {response_a.model} vs {response_b.model} on {task_type}: {e}"
                            )
                            continue

                actual_cost += evaluation_cost

            except Exception as e:
                logger.error(f"Evaluation system failed for task {task_type}: {e}")
                evaluation_results = []

            # Store task results
            benchmark_results.append(
                {
                    "task_type": task_type,
                    "task": clean_task,
                    "task_complexity": task_complexity,
                    "model_responses": model_responses,
                    "evaluation_results": evaluation_results,
                    "evaluation_cost": evaluation_cost,
                }
            )

        # Calculate comprehensive benchmark scores
        benchmark_scores = _calculate_benchmark_scores(
            models=models,
            benchmark_results=benchmark_results,
            evaluation_criteria=evaluation_criteria,
        )

        # Record actual cost
        await cost_guard.record_actual_cost(
            reservation_id,
            actual_cost,
            models[0],
            "model_benchmark",
        )

        # Store conversation (optional, non-fatal)
        try:
            orchestrator = get_conversation_orchestrator()
            storage_context = StorageContext(
                interface_type="mcp",
                tool_name="model_benchmark",
                session_id=session_id,
                context=f"Benchmark across {len(task_types)} task types: {', '.join(task_types)}",
                save_conversation=True,
            )

            # Create a summary response for storage
            summary_responses = []
            for model in models:
                model_score = next(
                    (s for s in benchmark_scores if s["model"] == model), None
                )
                if model_score:
                    from ...core.models import ModelResponse, TokenUsage

                    summary_response = ModelResponse(
                        content=f"Benchmark Score: {model_score['overall_score']:.1f}/100 across {len(task_types)} task types",
                        model=model,
                        usage=TokenUsage(
                            input_tokens=0, output_tokens=50, total_tokens=50
                        ),
                        cost_estimate=model_score.get("total_cost", Decimal("0.0")),
                        provider=detect_model_provider(model),
                    )
                    summary_responses.append(summary_response)

            await orchestrator.handle_interaction(
                prompt=f"Model benchmark across {len(task_types)} task types with {sample_size} tasks each",
                responses=summary_responses,
                storage_context=storage_context,
                evaluation_result={
                    "benchmark_type": "comprehensive",
                    "task_types": task_types,
                    "sample_size": sample_size,
                    "evaluation_criteria": evaluation_criteria,
                    "benchmark_scores": benchmark_scores,
                    "total_tasks": len(selected_tasks),
                    "total_evaluations": len(selected_tasks) * len(models),
                    "cost_analysis": {
                        "actual_cost": float(actual_cost),
                        "cost_limit": float(cost_limit_decimal),
                        "cost_per_model": float(actual_cost / len(models)),
                        "cost_per_task": float(actual_cost / len(selected_tasks)),
                    },
                },
            )
        except Exception as storage_error:
            logger.warning(f"Conversation storage failed (non-fatal): {storage_error}")

        # Generate comprehensive benchmark report
        return await _format_benchmark_report(
            models=models,
            task_types=task_types,
            sample_size=sample_size,
            evaluation_criteria=evaluation_criteria,
            benchmark_results=benchmark_results,
            benchmark_scores=benchmark_scores,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
        )

    except Exception as e:
        logger.error(f"Unexpected error in model_benchmark tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\\n\\nPlease check the logs and try again with simpler parameters."


def _calculate_benchmark_scores(
    models: list[str],
    benchmark_results: list[dict[str, Any]],
    evaluation_criteria: str,
) -> list[dict[str, Any]]:
    """Calculate comprehensive benchmark scores for each model."""
    model_scores = {
        model: {
            "total_wins": 0,
            "total_comparisons": 0,
            "task_scores": {},
            "avg_scores": {},
            "total_cost": Decimal("0.0"),
            "successful_tasks": 0,
            "failed_tasks": 0,
        }
        for model in models
    }

    # Process results for each task
    for result in benchmark_results:
        task_type = result["task_type"]
        model_responses = result["model_responses"]
        evaluation_results = result["evaluation_results"]

        # Initialize task scores for this type
        for model in models:
            if task_type not in model_scores[model]["task_scores"]:
                model_scores[model]["task_scores"][task_type] = {
                    "wins": 0,
                    "comparisons": 0,
                    "scores": [],
                }

        # Process evaluation results
        for eval_result in evaluation_results:
            model_a = eval_result["model_a"]
            model_b = eval_result["model_b"]
            winner = eval_result["winner"]

            model_scores[model_a]["total_comparisons"] += 1
            model_scores[model_b]["total_comparisons"] += 1
            model_scores[model_a]["task_scores"][task_type]["comparisons"] += 1
            model_scores[model_b]["task_scores"][task_type]["comparisons"] += 1

            if winner == "primary":  # model_a won
                model_scores[model_a]["total_wins"] += 1
                model_scores[model_a]["task_scores"][task_type]["wins"] += 1
            elif winner == "comparison":  # model_b won
                model_scores[model_b]["total_wins"] += 1
                model_scores[model_b]["task_scores"][task_type]["wins"] += 1

            # Track individual scores
            model_scores[model_a]["task_scores"][task_type]["scores"].append(
                eval_result["score_a"]
            )
            model_scores[model_b]["task_scores"][task_type]["scores"].append(
                eval_result["score_b"]
            )

        # Track costs and success rates
        for response in model_responses:
            model = response.model
            model_scores[model]["total_cost"] += response.cost_estimate
            if response.content.startswith("Error:"):
                model_scores[model]["failed_tasks"] += 1
            else:
                model_scores[model]["successful_tasks"] += 1

    # Calculate final scores
    final_scores = []

    for model in models:
        data = model_scores[model]

        # Overall win rate
        overall_win_rate = (
            (data["total_wins"] / data["total_comparisons"]) * 100
            if data["total_comparisons"] > 0
            else 0
        )

        # Task-specific performance
        task_performance = {}
        for task_type, task_data in data["task_scores"].items():
            task_win_rate = (
                (task_data["wins"] / task_data["comparisons"]) * 100
                if task_data["comparisons"] > 0
                else 0
            )
            task_avg_score = (
                sum(task_data["scores"]) / len(task_data["scores"])
                if task_data["scores"]
                else 5.0
            )
            task_performance[task_type] = {
                "win_rate": task_win_rate,
                "avg_score": task_avg_score,
                "comparisons": task_data["comparisons"],
            }

        # Success rate
        total_tasks = data["successful_tasks"] + data["failed_tasks"]
        success_rate = (
            (data["successful_tasks"] / total_tasks) * 100 if total_tasks > 0 else 0
        )

        # Calculate overall benchmark score
        if evaluation_criteria == "accuracy":
            overall_score = (overall_win_rate * 0.6) + (success_rate * 0.4)
        elif evaluation_criteria == "creativity":
            # Weight subjective quality higher for creativity
            avg_all_scores = []
            for task_perf in task_performance.values():
                avg_all_scores.append(task_perf["avg_score"])
            creativity_score = (
                (sum(avg_all_scores) / len(avg_all_scores)) * 10
                if avg_all_scores
                else 50
            )
            overall_score = (creativity_score * 0.7) + (success_rate * 0.3)
        elif evaluation_criteria == "speed":
            # Factor in cost efficiency and success rate
            max_cost = (
                max(s["total_cost"] for s in model_scores.values())
                if model_scores
                else Decimal("1.0")
            )
            cost_efficiency = (
                float(1 - (data["total_cost"] / max_cost)) * 100
                if max_cost > 0
                else 100
            )
            overall_score = (
                (cost_efficiency * 0.4)
                + (success_rate * 0.4)
                + (overall_win_rate * 0.2)
            )
        else:  # comprehensive
            # Balanced scoring across all factors
            avg_all_scores = []
            for task_perf in task_performance.values():
                avg_all_scores.append(task_perf["avg_score"])
            quality_score = (
                (sum(avg_all_scores) / len(avg_all_scores)) * 10
                if avg_all_scores
                else 50
            )
            overall_score = (
                (overall_win_rate * 0.3) + (quality_score * 0.4) + (success_rate * 0.3)
            )

        final_scores.append(
            {
                "model": model,
                "overall_score": min(overall_score, 100.0),  # Cap at 100
                "overall_win_rate": overall_win_rate,
                "success_rate": success_rate,
                "task_performance": task_performance,
                "total_cost": data["total_cost"],
                "total_comparisons": data["total_comparisons"],
                "provider": detect_model_provider(model),
            }
        )

    # Sort by overall score (highest first)
    final_scores.sort(key=lambda x: x["overall_score"], reverse=True)

    # Add rankings
    for i, score in enumerate(final_scores):
        score["rank"] = i + 1

    return final_scores


async def _format_benchmark_report(
    models: list[str],
    task_types: list[str],
    sample_size: int,
    evaluation_criteria: str,
    benchmark_results: list[dict[str, Any]],
    benchmark_scores: list[dict[str, Any]],
    actual_cost: Decimal,
    cost_limit: Decimal,
) -> str:
    """Format comprehensive benchmark report."""
    report = []

    # Header
    report.append("# 🏁 Model Benchmark Results")
    report.append("")

    # Benchmark summary
    report.append("## 📊 Benchmark Overview")
    report.append(f"**Models Tested**: {len(models)}")
    report.append(f"**Task Categories**: {', '.join(task_types)}")
    report.append(f"**Tasks per Category**: {sample_size}")
    report.append(f"**Total Tasks**: {len(benchmark_results)}")
    report.append(f"**Evaluation Criteria**: {evaluation_criteria.title()}")
    report.append("")

    # Cost analysis
    report.append("## 💰 Cost Analysis")
    report.append(f"**Total Cost**: ${actual_cost:.4f}")
    report.append(f"**Cost Limit**: ${cost_limit:.2f}")
    report.append(f"**Cost per Model**: ${actual_cost / len(models):.4f}")
    report.append(f"**Cost per Task**: ${actual_cost / len(benchmark_results):.4f}")

    # Get budget info
    try:
        cost_guard = get_cost_guard()
        budget_usage = await cost_guard.get_usage_summary(BudgetPeriod.DAILY)
        report.append(f"**Daily Budget Remaining**: ${budget_usage.available:.2f}")
    except Exception:
        report.append("**Daily Budget**: Not available")

    report.append("")

    # Overall rankings
    report.append(f"## 🏆 Overall Model Rankings ({evaluation_criteria.title()})")
    report.append("")

    for score in benchmark_scores:
        rank = score["rank"]
        model = score["model"]
        overall_score = score["overall_score"]
        win_rate = score["overall_win_rate"]
        success_rate = score["success_rate"]
        cost = score["total_cost"]
        provider = score["provider"]

        # Medal emoji for top 3
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}."

        report.append(f"### {medal} {model}")
        report.append(f"**Provider**: {provider.title()}")
        report.append(f"**Overall Score**: {overall_score:.1f}/100")
        report.append(f"**Win Rate**: {win_rate:.1f}%")
        report.append(f"**Success Rate**: {success_rate:.1f}%")
        report.append(f"**Total Cost**: ${cost:.4f}")
        report.append("")

    # Task-specific performance
    report.append("## 📋 Task Category Performance")
    report.append("")

    for task_type in task_types:
        report.append(f"### {task_type.title()} Tasks")
        report.append("")

        # Get performance for this task type
        task_rankings = []
        for score in benchmark_scores:
            if task_type in score["task_performance"]:
                task_perf = score["task_performance"][task_type]
                task_rankings.append(
                    {
                        "model": score["model"],
                        "win_rate": task_perf["win_rate"],
                        "avg_score": task_perf["avg_score"],
                        "provider": score["provider"],
                    }
                )

        # Sort by win rate for this task
        task_rankings.sort(key=lambda x: x["win_rate"], reverse=True)

        for i, ranking in enumerate(task_rankings[:3]):  # Show top 3 per category
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            report.append(
                f"{rank_emoji} **{ranking['model']}** - {ranking['win_rate']:.1f}% win rate, {ranking['avg_score']:.1f}/10 avg score"
            )

        report.append("")

    # Performance insights
    report.append("## 📈 Performance Insights")

    if benchmark_scores:
        best_model = benchmark_scores[0]
        report.append(f"**🎯 Top Performer**: {best_model['model']}")
        report.append(f"- **Overall Score**: {best_model['overall_score']:.1f}/100")
        report.append(
            f"- **Strength**: {_get_benchmark_strength(best_model, evaluation_criteria)}"
        )

        # Find best performer per category
        report.append("")
        report.append("**📊 Category Champions**:")
        for task_type in task_types:
            best_for_task = max(
                benchmark_scores,
                key=lambda x: x["task_performance"]
                .get(task_type, {})
                .get("win_rate", 0),
            )
            task_win_rate = (
                best_for_task["task_performance"].get(task_type, {}).get("win_rate", 0)
            )
            report.append(
                f"- **{task_type.title()}**: {best_for_task['model']} ({task_win_rate:.1f}% win rate)"
            )

        # Cost efficiency analysis
        if len(benchmark_scores) > 1:
            most_cost_efficient = min(benchmark_scores, key=lambda x: x["total_cost"])
            report.append("")
            report.append(f"**💰 Most Cost-Efficient**: {most_cost_efficient['model']}")
            report.append(f"- **Cost**: ${most_cost_efficient['total_cost']:.4f}")
            report.append(f"- **Overall Rank**: #{most_cost_efficient['rank']}")

            if most_cost_efficient != benchmark_scores[0]:
                cost_diff = best_model["total_cost"] - most_cost_efficient["total_cost"]
                score_diff = (
                    best_model["overall_score"] - most_cost_efficient["overall_score"]
                )
                report.append(
                    f"- **Trade-off**: {score_diff:.1f} fewer points for ${cost_diff:.4f} savings"
                )

    report.append("")

    # Statistical significance
    total_comparisons = sum(score["total_comparisons"] for score in benchmark_scores)
    avg_comparisons = (
        total_comparisons / len(benchmark_scores) if benchmark_scores else 0
    )

    report.append("## 📊 Statistical Analysis")
    report.append(f"**Total Comparisons**: {total_comparisons}")
    report.append(f"**Average Comparisons per Model**: {avg_comparisons:.1f}")

    if avg_comparisons >= 10:
        report.append("**✅ Statistical Confidence**: High (≥10 comparisons per model)")
    elif avg_comparisons >= 5:
        report.append(
            "**⚠️ Statistical Confidence**: Moderate (5-9 comparisons per model)"
        )
    else:
        report.append("**❌ Statistical Confidence**: Low (<5 comparisons per model)")
        report.append("*Consider increasing sample_size for more reliable results*")

    report.append("")

    # Recommendations
    report.append("## 🎯 Recommendations")

    if benchmark_scores:
        top_model = benchmark_scores[0]

        report.append(f"**🏆 Primary Recommendation**: Use **{top_model['model']}**")
        report.append(
            f"- **Best overall performance** with {top_model['overall_score']:.1f}/100 score"
        )
        report.append(
            f"- **{top_model['overall_win_rate']:.1f}% win rate** across all task types"
        )
        report.append(
            f"- **{top_model['success_rate']:.1f}% success rate** for reliable operation"
        )

        # Task-specific recommendations
        report.append("")
        report.append("**📋 Task-Specific Recommendations**:")
        for task_type in task_types:
            best_for_task = max(
                benchmark_scores,
                key=lambda x: x["task_performance"]
                .get(task_type, {})
                .get("win_rate", 0),
            )
            if best_for_task["model"] != top_model["model"]:
                task_win_rate = (
                    best_for_task["task_performance"]
                    .get(task_type, {})
                    .get("win_rate", 0)
                )
                report.append(
                    f"- **For {task_type} tasks**: Consider {best_for_task['model']} ({task_win_rate:.1f}% win rate)"
                )

        # Cost optimization suggestion
        if len(benchmark_scores) > 1:
            most_efficient = min(benchmark_scores, key=lambda x: x["total_cost"])
            if (
                most_efficient["model"] != top_model["model"]
                and most_efficient["overall_score"] >= 70
            ):
                report.append("")
                report.append(
                    f"**💡 Cost Optimization**: {most_efficient['model']} offers {most_efficient['overall_score']:.1f}/100 performance at ${most_efficient['total_cost']:.4f} cost"
                )

    report.append("")

    # Next steps
    report.append("## 🚀 Next Steps")
    if benchmark_scores:
        report.append(
            f"1. **Deploy** {benchmark_scores[0]['model']} for production workloads"
        )
        report.append("2. **Monitor** performance on your specific use cases")
        report.append("3. **A/B test** the top 2-3 models on real tasks")

        if any(score["success_rate"] < 95 for score in benchmark_scores):
            report.append(
                "4. **Investigate** any reliability issues with failed responses"
            )

        report.append("5. **Re-benchmark** periodically as models are updated")
    else:
        report.append("1. **Review** any error messages and model availability")
        report.append("2. **Try again** with different models or higher cost limits")
        report.append("3. **Reduce** sample_size if cost is a constraint")

    report.append("")
    report.append("---")
    report.append("*Model Benchmark Complete - Data-driven model selection! 🏁*")

    return "\\n".join(report)


def _get_benchmark_strength(score: dict[str, Any], evaluation_criteria: str) -> str:
    """Get the primary strength description for the top model."""
    if evaluation_criteria == "accuracy":
        return f"Highest accuracy with {score['overall_win_rate']:.1f}% win rate"
    elif evaluation_criteria == "creativity":
        return "Most creative and engaging responses"
    elif evaluation_criteria == "speed":
        return f"Optimal speed and cost efficiency (${score['total_cost']:.4f})"
    else:  # comprehensive
        return "Best overall balance across all evaluation criteria"


def _get_model_name_suggestions(invalid_model: str) -> str:
    """Generate helpful model name suggestions for benchmarking."""
    suggestions = []

    suggestions.append("**Cloud Models (OpenRouter format):**")
    suggestions.append("- `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`")
    suggestions.append("- `openai/gpt-4o`, `openai/gpt-4o-mini`")
    suggestions.append("- `google/gemini-pro-1.5`, `google/gemini-flash-1.5`")
    suggestions.append("")
    suggestions.append("**Local Models:**")
    suggestions.append("- `qwen3-4b-mlx`, `codestral-22b-v0.1`")
    suggestions.append("")
    suggestions.append("**Example model list for benchmarking:**")
    suggestions.append("```")
    suggestions.append(
        '["anthropic/claude-3-5-sonnet", "openai/gpt-4o", "qwen3-4b-mlx"]'
    )
    suggestions.append("```")

    return "\\n".join(suggestions)
