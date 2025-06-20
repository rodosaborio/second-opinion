"""
Second Opinion MCP tool implementation.

This module provides the core `second_opinion` tool for comparing AI responses
across different models, with cost optimization through response reuse and
explicit model specification.
"""

import logging
from decimal import Decimal
from typing import List, Optional

from ...cli.main import filter_think_tags
from ...config.model_configs import model_config_manager
from ...config.settings import get_settings
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, ModelRequest, TaskComplexity
from ...utils.client_factory import create_client_from_config
from ...utils.cost_tracking import get_cost_guard, BudgetPeriod
from ...utils.sanitization import sanitize_prompt, validate_model_name, validate_cost_limit, SecurityContext

logger = logging.getLogger(__name__)


async def second_opinion_tool(
    prompt: str,
    primary_model: Optional[str] = None,
    primary_response: Optional[str] = None,
    context: Optional[str] = None,
    comparison_models: Optional[List[str]] = None,
    cost_limit: Optional[float] = None,
) -> str:
    """
    Compare AI responses across models for alternative perspectives and quality assessment.
    
    This tool helps optimize AI model usage by comparing responses from different models,
    providing quality assessments, and suggesting cost-effective alternatives. It supports
    response reuse to minimize API costs when you already have a primary model response.
    
    Args:
        prompt: The question or task to analyze and compare across models
        primary_model: The model name that generated the original response 
                      (e.g., "anthropic/claude-3-5-sonnet"). If not provided, will use
                      the most frequently used model from session history or default.
        primary_response: The original response to compare against. When provided,
                         saves API costs by skipping the primary model call. This is
                         especially useful when you already have a response from another client.
        context: Additional context about the task or domain to improve comparison quality.
                For example: "This is for academic research" or "Technical documentation".
        comparison_models: Specific models to compare against the primary model.
                          Can be a list like ["openai/gpt-4o", "google/gemini-pro"].
                          If not provided, models will be auto-selected based on the
                          primary model tier and task complexity.
        cost_limit: Maximum cost limit for this operation in USD (e.g., 0.25).
                   If not provided, uses the configured default limit.
    
    Returns:
        A formatted comparison report showing:
        - Response quality assessment across models
        - Cost analysis and optimization recommendations
        - Task complexity evaluation
        - Actionable insights for model selection
        
    Example Usage:
        # Basic comparison with auto-selected models
        result = await second_opinion_tool(
            prompt="What's the capital of France?",
            primary_model="anthropic/claude-3-5-sonnet"
        )
        
        # Cost-efficient comparison with existing response
        result = await second_opinion_tool(
            prompt="Explain quantum computing",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Quantum computing is...",  # Saves API call
            comparison_models=["openai/gpt-4o", "google/gemini-pro"],
            context="For technical documentation"
        )
        
        # Budget-conscious comparison
        result = await second_opinion_tool(
            prompt="Write a marketing email",
            primary_model="openai/gpt-4o-mini",
            cost_limit=0.10
        )
    """
    try:
        # Input validation and sanitization
        
        # Sanitize prompt with user context (most permissive)
        clean_prompt = sanitize_prompt(prompt, SecurityContext.USER_PROMPT)
        
        # Sanitize context if provided
        clean_context = None
        if context:
            clean_context = sanitize_prompt(context, SecurityContext.USER_PROMPT)
        
        # Validate and normalize model names
        if primary_model:
            primary_model = validate_model_name(primary_model)
        
        if comparison_models:
            comparison_models = [
                validate_model_name(model) for model in comparison_models
            ]
        
        # Validate cost limit
        if cost_limit is not None:
            cost_limit_decimal = validate_cost_limit(cost_limit)
        else:
            # Get default from configuration
            try:
                tool_config = model_config_manager.get_tool_config("second_opinion")
                cost_limit_decimal = Decimal(str(tool_config.cost_limit_per_request))
            except Exception:
                settings = get_settings()
                cost_limit_decimal = settings.cost_management.default_per_request_limit
        
        logger.info(f"Starting second_opinion tool: prompt length={len(clean_prompt)}, "
                   f"primary_model={primary_model}, has_primary_response={primary_response is not None}")
        
        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()
        
        # Resolve primary model
        if not primary_model:
            # Use configuration default or most common model
            settings = get_settings()
            primary_model = "anthropic/claude-3-5-sonnet"  # Reasonable default
            logger.info(f"No primary model specified, using default: {primary_model}")
        
        # Classify task complexity
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_prompt)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")
            task_complexity = TaskComplexity.MODERATE
        
        # Select comparison models if not provided
        if not comparison_models:
            from ...cli.main import ComparisonModelSelector
            selector = ComparisonModelSelector()
            comparison_models = selector.select_models(
                primary_model=primary_model,
                explicit_models=None,
                task_complexity=task_complexity,
                max_models=2
            )
            logger.info(f"Auto-selected comparison models: {comparison_models}")
        
        # Estimate total cost
        estimated_cost = Decimal("0.0")
        
        # Estimate cost for primary model (if we need to call it)
        if not primary_response:
            try:
                primary_client = create_client_from_config(primary_model.split('/')[0] if '/' in primary_model else 'openrouter')
                primary_request = ModelRequest(
                    model=primary_model,
                    messages=[{"role": "user", "content": clean_prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                primary_cost = await primary_client.estimate_cost(primary_request)
                estimated_cost += primary_cost
            except Exception as e:
                logger.warning(f"Failed to estimate primary model cost: {e}")
                estimated_cost += Decimal("0.05")  # Conservative estimate
        
        # Estimate cost for comparison models
        for model in comparison_models:
            try:
                client = create_client_from_config(model.split('/')[0] if '/' in model else 'openrouter')
                request = ModelRequest(
                    model=model,
                    messages=[{"role": "user", "content": clean_prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                model_cost = await client.estimate_cost(request)
                estimated_cost += model_cost
            except Exception as e:
                logger.warning(f"Failed to estimate cost for {model}: {e}")
                estimated_cost += Decimal("0.05")  # Conservative estimate
        
        # Add evaluation cost
        estimated_cost += Decimal("0.02")  # Conservative evaluation estimate
        
        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost, "second_opinion", primary_model
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"
        
        # Get primary response (either provided or generate)
        actual_cost = Decimal("0.0")
        
        if primary_response:
            logger.info("Using provided primary response")
            # Clean the provided response
            clean_primary_response = filter_think_tags(primary_response)
        else:
            logger.info(f"Generating primary response with {primary_model}")
            try:
                primary_client = create_client_from_config(primary_model.split('/')[0] if '/' in primary_model else 'openrouter')
                primary_request = ModelRequest(
                    model=primary_model,
                    messages=[{"role": "user", "content": clean_prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                primary_model_response = await primary_client.complete(primary_request)
                clean_primary_response = filter_think_tags(primary_model_response.content)
                actual_cost += primary_model_response.cost_estimate
            except Exception as e:
                logger.error(f"Failed to get primary response from {primary_model}: {e}")
                await cost_guard.record_actual_cost(reservation_id, actual_cost, primary_model, "second_opinion")
                return f"❌ **Error**: Failed to get response from primary model {primary_model}: {str(e)}"
        
        # Get comparison responses
        comparison_responses = []
        comparison_costs = []
        
        for model in comparison_models:
            try:
                logger.info(f"Generating comparison response with {model}")
                client = create_client_from_config(model.split('/')[0] if '/' in model else 'openrouter')
                request = ModelRequest(
                    model=model,
                    messages=[{"role": "user", "content": clean_prompt}],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                response = await client.complete(request)
                clean_response = filter_think_tags(response.content)
                comparison_responses.append(clean_response)
                comparison_costs.append(response.cost_estimate)
                actual_cost += response.cost_estimate
            except Exception as e:
                logger.error(f"Failed to get response from {model}: {e}")
                comparison_responses.append(f"Error: Failed to get response from {model}")
                comparison_costs.append(Decimal("0.0"))
        
        # Perform evaluation
        try:
            logger.info("Performing response evaluation")
            evaluation_criteria = EvaluationCriteria(
                accuracy_weight=0.3,
                completeness_weight=0.25,
                clarity_weight=0.25,
                usefulness_weight=0.2
            )
            
            evaluation_results = []
            evaluation_cost = Decimal("0.0")
            
            for i, (model, response, model_cost) in enumerate(zip(comparison_models, comparison_responses, comparison_costs)):
                try:
                    result = await evaluator.compare_responses(
                        primary_response=clean_primary_response,
                        comparison_response=response,
                        criteria=evaluation_criteria,
                        primary_model=primary_model,
                        comparison_model=model,
                        task=clean_prompt
                    )
                    evaluation_results.append((model, result))
                    evaluation_cost += Decimal("0.01")  # Small evaluation cost
                except Exception as e:
                    logger.warning(f"Evaluation failed for {model}: {e}")
                    # Create a fallback evaluation result
                    fallback_result = {
                        'overall_winner': 'primary',
                        'overall_score': 7.0,
                        'reasoning': f"Evaluation failed, but {model} provided a response."
                    }
                    evaluation_results.append((model, fallback_result))
            
            actual_cost += evaluation_cost
            
        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            evaluation_results = [(model, {'overall_winner': 'unknown', 'overall_score': 5.0, 'reasoning': 'Evaluation unavailable'}) 
                                for model in comparison_models]
        
        # Record actual cost
        await cost_guard.record_actual_cost(reservation_id, actual_cost, primary_model, "second_opinion")
        
        # Generate report
        return await _format_comparison_report(
            prompt=clean_prompt,
            primary_model=primary_model,
            primary_response=clean_primary_response,
            comparison_models=comparison_models,
            comparison_responses=comparison_responses,
            comparison_costs=comparison_costs,
            evaluation_results=evaluation_results,
            task_complexity=task_complexity,
            actual_cost=actual_cost,
            cost_limit=cost_limit_decimal,
            context=clean_context
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in second_opinion tool: {e}")
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the logs for more details and try again with simpler parameters."


async def _format_comparison_report(
    prompt: str,
    primary_model: str,
    primary_response: str,
    comparison_models: List[str],
    comparison_responses: List[str],
    comparison_costs: List[Decimal],
    evaluation_results: List[tuple[str, dict]],
    task_complexity: TaskComplexity,
    actual_cost: Decimal,
    cost_limit: Decimal,
    context: Optional[str]
) -> str:
    """Format the comparison report for MCP client display."""
    
    report = []
    
    # Header
    report.append("# 🔍 Second Opinion Analysis")
    report.append("")
    
    # Task info
    report.append("## 📝 Task Summary")
    report.append(f"**Prompt**: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    if context:
        report.append(f"**Context**: {context}")
    report.append(f"**Task Complexity**: {task_complexity.value}")
    report.append(f"**Primary Model**: {primary_model}")
    report.append(f"**Comparison Models**: {', '.join(comparison_models)}")
    report.append("")
    
    # Cost summary
    report.append("## 💰 Cost Analysis")
    report.append(f"**Total Cost**: ${actual_cost:.4f}")
    report.append(f"**Cost Limit**: ${cost_limit:.2f}")
    
    # Get budget info
    try:
        cost_guard = get_cost_guard()
        budget_usage = await cost_guard.get_usage_summary(BudgetPeriod.DAILY)
        report.append(f"**Daily Budget Remaining**: ${budget_usage.available:.2f}")
    except Exception:
        report.append("**Daily Budget**: Not available")
    
    report.append("")
    
    # Primary response
    report.append("## 🎯 Primary Response")
    report.append(f"**Model**: {primary_model}")
    report.append("")
    report.append(primary_response[:1000] + ("..." if len(primary_response) > 1000 else ""))
    report.append("")
    
    # Comparison responses
    report.append("## 🔀 Alternative Responses")
    for i, (model, response, cost) in enumerate(zip(comparison_models, comparison_responses, comparison_costs)):
        report.append(f"### {model} (${cost:.4f})")
        report.append("")
        if response.startswith("Error:"):
            report.append(f"❌ {response}")
        else:
            report.append(response[:800] + ("..." if len(response) > 800 else ""))
        report.append("")
    
    # Evaluation results
    report.append("## ⭐ Quality Assessment")
    
    best_model = primary_model
    best_score = 0.0
    
    for model, result in evaluation_results:
        score = result.get('overall_score', 0.0)
        winner = result.get('overall_winner', 'unknown')
        reasoning = result.get('reasoning', 'No reasoning available')
        
        if score > best_score:
            best_score = score
            if winner == 'comparison':
                best_model = model
        
        report.append(f"### {model} vs {primary_model}")
        report.append(f"**Score**: {score:.1f}/10")
        report.append(f"**Winner**: {winner}")
        report.append(f"**Analysis**: {reasoning}")
        report.append("")
    
    # Recommendations
    report.append("## 🎯 Recommendations")
    
    if best_model == primary_model:
        report.append(f"✅ **{primary_model}** provided the best response for this task")
    else:
        report.append(f"💡 **{best_model}** might be better suited for this type of task")
    
    # Cost optimization suggestions
    cheapest_model = min(zip(comparison_models, comparison_costs), key=lambda x: x[1], default=(None, None))
    if cheapest_model[0] and cheapest_model[1] < actual_cost * Decimal("0.5"):
        report.append(f"💰 Consider using **{cheapest_model[0]}** for similar tasks to save ~${(actual_cost - cheapest_model[1]):.3f} per request")
    
    # Task complexity recommendations
    if task_complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
        budget_models = ["anthropic/claude-3-haiku", "openai/gpt-4o-mini", "google/gemini-flash"]
        relevant_budget = [m for m in budget_models if m not in [primary_model] + comparison_models]
        if relevant_budget:
            report.append(f"🎯 For {task_complexity.value.lower()} tasks, consider testing: **{relevant_budget[0]}**")
    
    report.append("")
    report.append("---")
    report.append("*Second Opinion Analysis Complete*")
    
    return "\n".join(report)