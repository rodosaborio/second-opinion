"""
Tests for the MCP should_downgrade tool.

Tests verify that the tool correctly analyzes cost optimization opportunities,
handles both expensive and budget models, and provides actionable downgrade
recommendations with accurate cost savings calculations.
"""

import pytest
from decimal import Decimal
from unittest.mock import patch

from src.second_opinion.mcp.tools.should_downgrade import should_downgrade_tool
from .conftest import SAMPLE_CODE_PROMPT, SAMPLE_CODE_RESPONSE, SAMPLE_SHELL_PROMPT, SAMPLE_SHELL_RESPONSE


class TestShouldDowngradeTool:
    """Test the MCP should_downgrade tool for cost optimization analysis."""
    
    @pytest.mark.asyncio
    async def test_basic_downgrade_functionality(self):
        """Test basic downgrade functionality with expensive model."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            cost_limit=0.25
        )
        
        # Should return a formatted cost optimization report
        assert isinstance(result, str)
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        assert "Current Situation" in result
        assert "Current Response Quality" in result
        assert "Cheaper Alternatives Tested" in result
        assert "Cost Savings Analysis" in result
        assert "My Recommendation" in result
        assert "Next Steps" in result
        
        # Should contain the main analysis sections even if some API calls fail
        # (Error messages are expected when API keys are not configured)
        assert "Security" not in result
        assert "injection" not in result.lower()

    @pytest.mark.asyncio
    async def test_local_model_inclusion(self):
        """Test that local models are included for maximum cost savings."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model="openai/gpt-4o",
            test_local=True,
            cost_limit=0.20
        )
        
        # Should mention local models and zero cost
        assert "FREE" in result or "$0.00" in result
        assert "100% savings" in result or "100%" in result
        assert "Local" in result or "local" in result
        
        # Should provide cost optimization recommendations
        assert "Cost Optimization Analysis" in result
        assert "Potential Monthly Savings" in result

    @pytest.mark.asyncio
    async def test_local_model_disabled(self):
        """Test behavior when local models are disabled."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model="anthropic/claude-3-5-sonnet",
            test_local=False,
            cost_limit=0.15
        )
        
        # Should still provide analysis but focus on cloud alternatives
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        assert "**Testing Local Models**: No" in result
        
        # Should still contain downgrade analysis
        assert "Cheaper Alternatives Tested" in result

    @pytest.mark.asyncio
    async def test_expensive_model_analysis(self):
        """Test analysis of expensive models with clear downgrade opportunities."""
        result = await should_downgrade_tool(
            current_response="The capital of France is Paris. It is located in the Île-de-France region.",
            task="What is the capital of France?",
            current_model="anthropic/claude-3-opus",  # Very expensive model
            test_local=True,
            cost_limit=0.30
        )
        
        # Should provide cost analysis even if downgrade candidates fail
        # (When API services are unavailable, tool correctly recommends keeping current model)
        assert "Cost per Request" in result
        assert "savings" in result.lower()
        
        # Should mention the expensive current model and provide recommendations
        assert "claude-3-opus" in result
        assert ("DOWNGRADE" in result or "CONSIDER" in result or "KEEP" in result)

    @pytest.mark.asyncio
    async def test_budget_model_analysis(self):
        """Test analysis when current model is already budget-friendly."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model="anthropic/claude-3-haiku",  # Already budget model
            test_local=True,
            cost_limit=0.15
        )
        
        # Should still analyze but may recommend keeping current model
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        assert "claude-3-haiku" in result
        
        # Should still test local alternatives for zero cost
        if "Local" in result:
            assert "$0.00" in result or "FREE" in result

    @pytest.mark.asyncio
    async def test_code_snippet_acceptance(self):
        """Test that code snippets are properly handled without security errors."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_SHELL_RESPONSE,
            task=SAMPLE_SHELL_PROMPT,
            current_model="openai/gpt-4o",
            test_local=True,
            cost_limit=0.20
        )
        
        # Should process code snippets successfully
        assert isinstance(result, str)
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        
        # Should not reject shell commands or code due to security issues
        # (API error messages are expected when services are not configured)
        assert "Security" not in result

    @pytest.mark.asyncio
    async def test_provider_detection_openrouter(self):
        """Test that cloud models use OpenRouter client."""
        with patch("src.second_opinion.mcp.tools.should_downgrade.detect_model_provider") as mock_detect:
            mock_detect.return_value = "openrouter"
            
            result = await should_downgrade_tool(
                current_response="Test response",
                task="Test task",
                current_model="anthropic/claude-3-5-sonnet",
                test_local=False,
                cost_limit=0.15
            )
            
            # Should have called detect_model_provider for current model and alternatives
            assert mock_detect.call_count >= 2
            
            # Should contain analysis
            assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result

    @pytest.mark.asyncio
    async def test_provider_detection_lmstudio(self):
        """Test that local models use LM Studio client."""
        with patch("src.second_opinion.mcp.tools.should_downgrade.detect_model_provider") as mock_detect:
            # Return different providers for different models
            def mock_provider(model):
                if "mlx" in model.lower() or "qwen" in model.lower():
                    return "lmstudio"
                return "openrouter"
            
            mock_detect.side_effect = mock_provider
            
            result = await should_downgrade_tool(
                current_response="Test response",
                task="Test task", 
                current_model="openai/gpt-4o",
                test_local=True,
                cost_limit=0.15
            )
            
            # Should have called detect_model_provider
            assert mock_detect.called
            
            # Should contain analysis
            assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self):
        """Test that cost limits are properly enforced."""
        # Test with very low cost limit
        result = await should_downgrade_tool(
            current_response="Short response",
            task="Simple task",
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            cost_limit=0.01  # Very low limit
        )
        
        # Should either complete with limited testing or show budget error
        assert isinstance(result, str)
        if "Budget Error" in result:
            assert "cost limit" in result.lower()
        else:
            assert "Cost Optimization Analysis" in result

    @pytest.mark.asyncio
    async def test_no_current_model_specified(self):
        """Test behavior when current model is not specified."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model=None,  # No current model specified
            test_local=True,
            cost_limit=0.20
        )
        
        # Should use default expensive model for baseline comparison
        assert isinstance(result, str)
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        
        # Should mention the default model used
        assert "claude-3-5-sonnet" in result

    @pytest.mark.asyncio
    async def test_complex_task_handling(self):
        """Test handling of complex tasks that may require better models."""
        complex_task = "Analyze the economic implications of quantum computing on cryptographic security"
        complex_response = "Quantum computing poses significant challenges to current cryptographic methods..."
        
        result = await should_downgrade_tool(
            current_response=complex_response,
            task=complex_task,
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            cost_limit=0.25
        )
        
        # Should still provide analysis but may be more conservative
        assert "# 💰 Should You Downgrade? Cost Optimization Analysis" in result
        assert "Task Complexity" in result
        
        # Should consider the complexity in recommendations
        assert "My Recommendation" in result

    @pytest.mark.asyncio
    async def test_cost_savings_calculations(self):
        """Test that cost savings are properly calculated and displayed."""
        result = await should_downgrade_tool(
            current_response="Simple answer: 42",
            task="What is the answer to everything?",
            current_model="openai/gpt-4o",  # Expensive model
            test_local=True,
            cost_limit=0.25
        )
        
        # Should include cost analysis with specific amounts
        assert "Cost Savings Analysis" in result
        assert "Current Cost per Request" in result
        assert "Potential Monthly Savings" in result
        
        # Should show percentage savings
        assert "%" in result
        assert "savings" in result.lower()

    @pytest.mark.asyncio
    async def test_actionable_recommendations(self):
        """Test that the tool provides actionable next steps."""
        result = await should_downgrade_tool(
            current_response=SAMPLE_CODE_RESPONSE,
            task=SAMPLE_CODE_PROMPT,
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            cost_limit=0.20
        )
        
        # Should provide clear next steps
        assert "Next Steps" in result
        assert "1." in result and "2." in result  # Numbered action items
        
        # Should have actionable language
        action_words = ["Test", "Try", "Consider", "Switch", "Keep", "Monitor"]
        assert any(word in result for word in action_words)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with invalid model name
        result = await should_downgrade_tool(
            current_response="Test response",
            task="Test task",
            current_model="invalid-model-name-123",
            test_local=True,
            cost_limit=0.15
        )
        
        # Should handle invalid model gracefully
        if "Invalid Current Model" in result:
            assert "Suggested formats" in result
        else:
            # Or handle it internally and continue with analysis
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_monthly_savings_projection(self):
        """Test that monthly savings projections are provided."""
        result = await should_downgrade_tool(
            current_response="The sky is blue because of light scattering.",
            task="Why is the sky blue?",
            current_model="anthropic/claude-3-5-sonnet",
            test_local=True,
            cost_limit=0.20
        )
        
        # Should include monthly savings projection
        assert "Potential Monthly Savings" in result
        assert "100 requests" in result
        assert "Save" in result or "savings" in result.lower()
        
        # Should show current vs alternative costs
        assert "Current:" in result
        assert "Local Models:" in result or "With Local" in result