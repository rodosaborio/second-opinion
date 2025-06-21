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
from ...clients import detect_model_provider
from ...config.model_configs import model_config_manager
from ...config.settings import get_settings
from ...core.evaluator import get_evaluator
from ...core.models import EvaluationCriteria, Message, ModelRequest, TaskComplexity
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
    Get a second opinion on an AI response by comparing it against alternative models.
    
    This tool is designed for natural conversation flow where an AI client has already
    provided a response and wants to evaluate it against alternatives. It helps optimize
    AI model usage by providing quality assessments, cost optimization recommendations,
    and suggestions for when to use local vs cloud models.
    
    NATURAL USAGE PATTERN:
    1. User asks: "Write a Python function to calculate fibonacci"
    2. AI responds: <provides code>
    3. User asks: "Can you get a second opinion on that?"
    4. AI calls this tool with its response for comparison
    
    Args:
        prompt: The original question or task that was asked
        primary_model: The model that provided the original response. Use OpenRouter format:
                      - Claude Desktop: "anthropic/claude-3-5-sonnet"
                      - ChatGPT: "openai/gpt-4o" or "openai/gpt-4o-mini"
                      - Gemini: "google/gemini-pro-1.5"
                      - Local models: "qwen3-4b-mlx", "codestral-22b-v0.1", etc.
        primary_response: The response to evaluate (RECOMMENDED). When provided, saves
                         costs and evaluates the actual response the user saw.
        context: Additional context about the task domain for better comparison quality.
                For example: "coding task", "academic research", "creative writing".
        comparison_models: Specific models to compare against. If not provided, will
                          auto-select alternatives including cost-effective local options.
        cost_limit: Maximum cost limit for this operation in USD (default: $0.25).
    
    Returns:
        A second opinion report with:
        - Quality assessment of the original response
        - Comparison with alternative model approaches  
        - Cost optimization recommendations (including local models)
        - Decision guidance: "stick with your model" vs "consider alternatives"
        
    RECOMMENDED USAGE (Natural Conversation Flow):
        # After providing a response to user, get second opinion
        result = await second_opinion_tool(
            prompt="Write a Python function to calculate fibonacci",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="def fibonacci(n):\n    if n <= 1:\n        return n...",
            context="coding task"
        )
        
    OTHER USAGE PATTERNS:
        # Compare model capabilities for a new task
        result = await second_opinion_tool(
            prompt="Explain quantum computing to a 10-year-old",
            primary_model="openai/gpt-4o-mini",  # Will generate response
            context="educational content"
        )
        
        # Test local model vs cloud alternatives
        result = await second_opinion_tool(
            prompt="Debug this code snippet",
            primary_model="qwen3-4b-mlx",  # Local model
            comparison_models=["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]
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
        
        # Validate and normalize model names with helpful suggestions
        if primary_model:
            try:
                primary_model = validate_model_name(primary_model)
            except Exception as e:
                # Provide helpful model name suggestions
                suggestions = _get_model_name_suggestions(primary_model)
                suggestion_text = f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                return f"❌ **Invalid Primary Model**: {str(e)}{suggestion_text}"
        
        if comparison_models:
            try:
                comparison_models = [
                    validate_model_name(model) for model in comparison_models
                ]
            except Exception as e:
                # Provide helpful model name suggestions
                suggestions = _get_model_name_suggestions(str(e))
                suggestion_text = f"\n\n**Suggested formats:**\n{suggestions}" if suggestions else ""
                return f"❌ **Invalid Comparison Model**: {str(e)}{suggestion_text}"
        
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
        logger.info(f"Cost limit: ${cost_limit_decimal:.2f}")
        
        # Get core systems
        evaluator = get_evaluator()
        cost_guard = get_cost_guard()
        
        # Resolve primary model
        if not primary_model:
            # Use configuration default or most common model
            settings = get_settings()
            primary_model = "anthropic/claude-3-5-sonnet"  # Reasonable default
            logger.info(f"No primary model specified, using default: {primary_model}")
        
        # Classify task complexity (initialize with default first)
        task_complexity = TaskComplexity.MODERATE  # Default fallback
        try:
            task_complexity = await evaluator.classify_task_complexity(clean_prompt)
            logger.info(f"Task complexity classified as: {task_complexity.value}")
        except Exception as e:
            logger.warning(f"Failed to classify task complexity: {e}")
            # task_complexity already set to default above
        
        # Select comparison models if not provided
        if not comparison_models:
            from ...cli.main import ComparisonModelSelector
            selector = ComparisonModelSelector()
            comparison_models = selector.select_models(
                primary_model=primary_model,
                tool_name="second_opinion",  # Explicitly specify tool name
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
                provider = detect_model_provider(primary_model)
                primary_client = create_client_from_config(provider)
                primary_request = ModelRequest(
                    model=primary_model,
                    messages=[Message(role="user", content=clean_prompt)],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                primary_cost = await primary_client.estimate_cost(primary_request)
                estimated_cost += primary_cost
                logger.info(f"Primary model cost estimate: ${primary_cost:.4f}")
            except Exception as e:
                logger.warning(f"Failed to estimate primary model cost: {e}")
                estimated_cost += Decimal("0.05")  # Conservative estimate
        
        # Estimate cost for comparison models
        for model in comparison_models:
            try:
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_prompt)],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                model_cost = await client.estimate_cost(request)
                estimated_cost += model_cost
                logger.info(f"Comparison model {model} cost estimate: ${model_cost:.4f}")
            except Exception as e:
                logger.warning(f"Failed to estimate cost for {model}: {e}")
                estimated_cost += Decimal("0.05")  # Conservative estimate
        
        # Add evaluation cost (uses evaluation model)
        try:
            # Use a small, efficient model for evaluation
            evaluation_model = "openai/gpt-4o-mini"
            eval_provider = detect_model_provider(evaluation_model)
            eval_client = create_client_from_config(eval_provider)
            # Small evaluation request
            eval_request = ModelRequest(
                model=evaluation_model,
                messages=[Message(role="user", content="evaluation task")],
                max_tokens=500,
                temperature=0.1
            )
            eval_cost = await eval_client.estimate_cost(eval_request)
            estimated_cost += eval_cost * len(comparison_models)  # One evaluation per comparison
            logger.info(f"Evaluation cost estimate: ${eval_cost * len(comparison_models):.4f}")
        except Exception as e:
            logger.warning(f"Failed to estimate evaluation cost: {e}")
            estimated_cost += Decimal("0.02")  # Conservative fallback
        
        # Check budget
        try:
            budget_check = await cost_guard.check_and_reserve_budget(
                estimated_cost, "second_opinion", primary_model, per_request_override=cost_limit_decimal
            )
            reservation_id = budget_check.reservation_id
        except Exception as e:
            return f"❌ **Budget Error**: {str(e)}\n\nEstimated cost: ${estimated_cost:.4f}\nCost limit: ${cost_limit_decimal:.2f}"
        
        # Get primary response (either provided or generate)
        actual_cost = Decimal("0.0")
        
        if primary_response:
            logger.info("Using provided primary response")
            # Clean the provided response and create a mock ModelResponse
            clean_primary_response_text = filter_think_tags(primary_response)
            
            # Estimate cost for the primary response even when provided
            primary_cost_estimate = Decimal("0.0")
            try:
                provider = detect_model_provider(primary_model)
                primary_client = create_client_from_config(provider)
                primary_request = ModelRequest(
                    model=primary_model,
                    messages=[Message(role="user", content=clean_prompt)],
                    max_tokens=len(clean_primary_response_text.split()) * 2,  # Estimate tokens from response
                    temperature=0.1,
                    system_prompt=clean_context
                )
                primary_cost_estimate = await primary_client.estimate_cost(primary_request)
                logger.info(f"Estimated cost for provided primary response: ${primary_cost_estimate:.4f}")
            except Exception as e:
                logger.warning(f"Failed to estimate cost for primary response: {e}")
                primary_cost_estimate = Decimal("0.01")  # Small fallback estimate
            
            # Create a ModelResponse with realistic cost estimate
            from ...core.models import ModelResponse, TokenUsage
            estimated_input_tokens = int(len(clean_prompt.split()) * 1.3)  # Rough token estimate
            estimated_output_tokens = int(len(clean_primary_response_text.split()) * 1.3)
            total_tokens = estimated_input_tokens + estimated_output_tokens  # Ensure exact sum
            primary_model_response = ModelResponse(
                content=clean_primary_response_text,
                model=primary_model,
                usage=TokenUsage(
                    input_tokens=estimated_input_tokens, 
                    output_tokens=estimated_output_tokens, 
                    total_tokens=total_tokens
                ),
                cost_estimate=primary_cost_estimate,
                provider=detect_model_provider(primary_model)
            )
            actual_cost += primary_cost_estimate
        else:
            logger.info(f"Generating primary response with {primary_model}")
            try:
                provider = detect_model_provider(primary_model)
                primary_client = create_client_from_config(provider)
                primary_request = ModelRequest(
                    model=primary_model,
                    messages=[Message(role="user", content=clean_prompt)],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                primary_model_response = await primary_client.complete(primary_request)
                clean_primary_response_text = filter_think_tags(primary_model_response.content)
                actual_cost += primary_model_response.cost_estimate
            except Exception as e:
                logger.error(f"Failed to get primary response from {primary_model}: {e}")
                await cost_guard.record_actual_cost(reservation_id, actual_cost, primary_model, "second_opinion")
                return f"❌ **Error**: Failed to get response from primary model {primary_model}: {str(e)}"
        
        # Get comparison responses
        comparison_responses = []  # Will store ModelResponse objects
        comparison_costs = []
        
        for model in comparison_models:
            try:
                logger.info(f"Generating comparison response with {model}")
                provider = detect_model_provider(model)
                client = create_client_from_config(provider)
                request = ModelRequest(
                    model=model,
                    messages=[Message(role="user", content=clean_prompt)],
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=clean_context
                )
                response = await client.complete(request)
                # Store the full ModelResponse object for evaluation
                comparison_responses.append(response)
                comparison_costs.append(response.cost_estimate)
                actual_cost += response.cost_estimate
            except Exception as e:
                logger.error(f"Failed to get response from {model}: {e}")
                # Create error response as ModelResponse for consistency
                from ...core.models import ModelResponse, TokenUsage
                error_response = ModelResponse(
                    content=f"Error: Failed to get response from {model}: {str(e)}",
                    model=model,
                    usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    cost_estimate=Decimal("0.0"),
                    provider=detect_model_provider(model)
                )
                comparison_responses.append(error_response)
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
            
            # Use a smaller, cost-effective model for evaluation to prevent evaluation failures
            evaluator_model = "openai/gpt-4o-mini"
            
            for response in comparison_responses:
                try:
                    # Skip error responses
                    if response.content.startswith("Error:"):
                        fallback_result = {
                            'overall_winner': 'primary',
                            'overall_score': 0.0,
                            'reasoning': f"Comparison model {response.model} failed to respond."
                        }
                        evaluation_results.append((response.model, fallback_result))
                        continue
                        
                    result = await evaluator.compare_responses(
                        primary_model_response,
                        response,
                        original_task=clean_prompt,
                        criteria=evaluation_criteria,
                        evaluator_model=evaluator_model
                    )
                    # Convert ComparisonResult to dict format for consistency
                    result_dict = {
                        'overall_winner': result.winner,
                        'overall_score': result.overall_score,
                        'reasoning': result.reasoning  # Fixed: use .reasoning not .detailed_analysis
                    }
                    evaluation_results.append((response.model, result_dict))
                    evaluation_cost += Decimal("0.01")  # Small evaluation cost
                except Exception as e:
                    logger.warning(f"Evaluation failed for {response.model}: {e}")
                    # Create a fallback evaluation result
                    fallback_result = {
                        'overall_winner': 'primary',
                        'overall_score': 5.0,  # Neutral score
                        'reasoning': f"Evaluation failed for {response.model}: {str(e)}. Using fallback assessment."
                    }
                    evaluation_results.append((response.model, fallback_result))
            
            actual_cost += evaluation_cost
            
        except Exception as e:
            logger.error(f"Evaluation system failed: {e}")
            evaluation_results = [(model, {'overall_winner': 'unknown', 'overall_score': 5.0, 'reasoning': 'Evaluation unavailable'}) 
                                for model in comparison_models]
        
        # Record actual cost
        await cost_guard.record_actual_cost(reservation_id, actual_cost, primary_model, "second_opinion")
        logger.info(f"Total operation cost: ${actual_cost:.4f}, Budget used: ${actual_cost:.4f}")
        
        # Generate report
        return await _format_comparison_report(
            prompt=clean_prompt,
            primary_model=primary_model,
            primary_response=clean_primary_response_text,
            comparison_models=[r.model for r in comparison_responses],
            comparison_responses=[filter_think_tags(r.content) for r in comparison_responses],
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
    
    # Header with decision-support framing
    report.append("# 🤔 Second Opinion: Should You Stick or Switch?")
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
    for model, response, cost in zip(comparison_models, comparison_responses, comparison_costs):
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
    
    # Decision-focused recommendations
    report.append("## 🎯 My Recommendation")
    
    primary_provider = detect_model_provider(primary_model)
    
    if best_model == primary_model:
        report.append(f"**✅ STICK with {primary_model}**")
        report.append(f"Your model provided the best response quality for this task.")
    else:
        report.append(f"**💡 CONSIDER switching to {best_model}**")
        report.append(f"This alternative might provide better results for similar tasks.")
    
    report.append("")
    
    # Cost optimization with local model focus
    local_models = [m for m in comparison_models if detect_model_provider(m) == "lmstudio"]
    if local_models and primary_provider == "openrouter":
        local_model = local_models[0]
        report.append("## 💰 Cost Optimization Opportunity")
        report.append(f"**Local Alternative**: Consider testing **{local_model}** for development or high-volume use")
        report.append(f"- **Cost**: $0.00 (local inference)")
        report.append(f"- **Your current cost**: ${actual_cost:.4f} per request")
        report.append(f"- **Potential savings**: 100% for similar quality tasks")
        report.append("")
    
    # Actionable next steps
    report.append("## 🚀 Next Steps")
    if best_model == primary_model:
        if local_models:
            report.append(f"1. **Keep using** {primary_model} for quality")
            report.append(f"2. **Test** {local_models[0]} for cost savings on similar tasks")
        else:
            report.append(f"1. **Continue using** {primary_model} - it's working well!")
            report.append(f"2. **Consider** local models for development and cost optimization")
    else:
        report.append(f"1. **Try** {best_model} for this type of task")
        report.append(f"2. **Compare results** to see if the switch improves your workflow")
        if local_models:
            report.append(f"3. **Experiment** with {local_models[0]} for cost-effective alternatives")
    
    report.append("")
    report.append("---")
    report.append("*Second Opinion Complete - Happy model hunting! 🎯*")
    
    return "\n".join(report)


def _get_model_name_suggestions(invalid_model: str) -> str:
    """
    Generate helpful model name suggestions based on common patterns.
    
    Args:
        invalid_model: The invalid model name that was provided
        
    Returns:
        Formatted string with model name suggestions
    """
    suggestions = []
    
    # Common provider suggestions
    suggestions.append("**Cloud Models (OpenRouter format):**")
    suggestions.append("- Claude: `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`")
    suggestions.append("- ChatGPT: `openai/gpt-4o`, `openai/gpt-4o-mini`")
    suggestions.append("- Gemini: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`")
    suggestions.append("")
    suggestions.append("**Local Models (LM Studio):**")
    suggestions.append("- Qwen: `qwen3-4b-mlx`, `qwen3-0.6b-mlx`")
    suggestions.append("- Codestral: `codestral-22b-v0.1`, `devstral-small-2505-mlx`")
    
    # Analyze the invalid model for specific suggestions
    invalid_lower = invalid_model.lower() if invalid_model else ""
    
    if "claude" in invalid_lower:
        suggestions.append("")
        suggestions.append("**For Claude models, try:**")
        suggestions.append("- `anthropic/claude-3-5-sonnet` (recommended)")
        suggestions.append("- `anthropic/claude-3-haiku` (budget option)")
    elif "gpt" in invalid_lower or "openai" in invalid_lower:
        suggestions.append("")
        suggestions.append("**For OpenAI models, try:**")
        suggestions.append("- `openai/gpt-4o` (latest)")
        suggestions.append("- `openai/gpt-4o-mini` (budget option)")
    elif "gemini" in invalid_lower or "google" in invalid_lower:
        suggestions.append("")
        suggestions.append("**For Google models, try:**")
        suggestions.append("- `google/gemini-pro-1.5`")
        suggestions.append("- `google/gemini-flash-1.5`")
    elif any(local in invalid_lower for local in ["qwen", "codestral", "llama", "mlx"]):
        suggestions.append("")
        suggestions.append("**For local models, try:**")
        suggestions.append("- `qwen3-4b-mlx` (good balance)")
        suggestions.append("- `codestral-22b-v0.1` (code-focused)")
    
    return "\n".join(suggestions)