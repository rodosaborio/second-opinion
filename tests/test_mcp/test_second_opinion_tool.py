"""
Tests for the MCP second_opinion tool.

Tests verify that the tool works correctly with mocked dependencies,
handles both OpenRouter and LM Studio clients, and processes code
snippets without security errors.
"""

import pytest
from decimal import Decimal
from unittest.mock import patch

from src.second_opinion.mcp.tools.second_opinion import second_opinion_tool
from .conftest import SAMPLE_CODE_PROMPT, SAMPLE_CODE_RESPONSE, SAMPLE_SHELL_PROMPT, SAMPLE_SHELL_RESPONSE


class TestSecondOpinionTool:
    """Test the MCP second_opinion tool."""
    
    @pytest.mark.asyncio
    async def test_basic_tool_functionality(self):
        """Test basic tool functionality with mocked dependencies."""
        result = await second_opinion_tool(
            prompt=SAMPLE_CODE_PROMPT,
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response=SAMPLE_CODE_RESPONSE,
            context="User asking for clean Python code",
            cost_limit=0.25
        )
        
        # Should return a formatted report
        assert isinstance(result, str)
        assert "# 🔍 Second Opinion Analysis" in result
        assert "Task Summary" in result
        assert "Cost Analysis" in result
        assert "Primary Response" in result
        assert "Alternative Responses" in result
        assert "Quality Assessment" in result
        assert "Recommendations" in result
        
        # Should not contain error messages
        assert "❌" not in result
        assert "Security" not in result
        assert "injection" not in result.lower()

    @pytest.mark.asyncio
    async def test_code_snippet_acceptance(self):
        """Test that code snippets with backticks are accepted."""
        result = await second_opinion_tool(
            prompt=SAMPLE_SHELL_PROMPT,
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response=SAMPLE_SHELL_RESPONSE,
            cost_limit=0.15
        )
        
        # Should process successfully
        assert isinstance(result, str)
        assert "# 🔍 Second Opinion Analysis" in result
        
        # Should not reject shell commands
        assert "❌" not in result
        assert "Security" not in result

    @pytest.mark.asyncio 
    async def test_openrouter_model_detection(self):
        """Test that cloud models use OpenRouter client."""
        with patch("src.second_opinion.mcp.tools.second_opinion.detect_model_provider") as mock_detect:
            mock_detect.return_value = "openrouter"
            
            result = await second_opinion_tool(
                prompt="Test prompt",
                primary_model="anthropic/claude-3-5-sonnet",
                cost_limit=0.10
            )
            
            # Should have called detect_model_provider for all models
            assert mock_detect.call_count >= 3  # Primary + 2 comparison models
            
            # Should contain analysis
            assert "# 🔍 Second Opinion Analysis" in result

    @pytest.mark.asyncio
    async def test_lmstudio_model_detection(self):
        """Test that local models use LM Studio client."""
        with patch("src.second_opinion.mcp.tools.second_opinion.detect_model_provider") as mock_detect:
            mock_detect.return_value = "lmstudio"
            
            result = await second_opinion_tool(
                prompt="Test prompt", 
                primary_model="qwen3-4b-mlx",
                cost_limit=0.10
            )
            
            # Should have called detect_model_provider
            assert mock_detect.called
            
            # Should contain analysis
            assert "# 🔍 Second Opinion Analysis" in result

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self):
        """Test that low cost limits are properly enforced."""
        # Mock cost guard to reject low budget
        with patch("src.second_opinion.mcp.tools.second_opinion.get_cost_guard") as mock_guard:
            mock_cost_guard = mock_guard.return_value
            mock_cost_guard.check_and_reserve_budget.side_effect = Exception(
                "Estimated cost $0.12 exceeds per-request limit $0.01"
            )
            
            result = await second_opinion_tool(
                prompt="Test prompt",
                primary_model="anthropic/claude-3-5-sonnet",
                cost_limit=0.01  # Very low limit
            )
            
            # Should return budget error
            assert "❌ **Budget Error**" in result
            assert "exceeds per-request limit" in result

    @pytest.mark.asyncio
    async def test_comparison_model_auto_selection(self):
        """Test that comparison models are auto-selected when not provided."""
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Test response",
            cost_limit=0.20
        )
        
        # Should auto-select comparison models
        assert "Comparison Models" in result
        assert "openai/gpt-4o" in result or "google/gemini" in result

    @pytest.mark.asyncio
    async def test_explicit_comparison_models(self):
        """Test using explicitly provided comparison models."""
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet", 
            primary_response="Test response",
            comparison_models=["openai/gpt-4o", "google/gemini-pro"],
            cost_limit=0.20
        )
        
        # Should use provided models
        assert "openai/gpt-4o" in result
        assert "google/gemini-pro" in result

    @pytest.mark.asyncio
    async def test_context_processing(self):
        """Test that context is properly processed and included."""
        context = "This is for academic research on AI model comparison"
        
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Test response", 
            context=context,
            cost_limit=0.15
        )
        
        # Context should appear in the report
        assert context in result

    @pytest.mark.asyncio
    async def test_primary_response_reuse(self):
        """Test that providing primary_response saves API calls."""
        primary_response = "This is a pre-generated response"
        
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response=primary_response,
            cost_limit=0.15
        )
        
        # Should include the provided response
        assert primary_response in result
        assert "Primary Response" in result

    @pytest.mark.asyncio
    async def test_task_complexity_classification(self):
        """Test that task complexity is classified and displayed."""
        result = await second_opinion_tool(
            prompt="What is 2+2?",
            primary_model="anthropic/claude-3-5-sonnet", 
            primary_response="2+2 equals 4",
            cost_limit=0.10
        )
        
        # Should show task complexity
        assert "Task Complexity" in result
        assert "simple" in result.lower() or "moderate" in result.lower() or "complex" in result.lower()

    @pytest.mark.asyncio
    async def test_error_handling_client_failure(self):
        """Test handling of client creation failures."""
        with patch("src.second_opinion.mcp.tools.second_opinion.create_client_from_config") as mock_factory:
            mock_factory.side_effect = Exception("Client creation failed")
            
            result = await second_opinion_tool(
                prompt="Test prompt",
                primary_model="anthropic/claude-3-5-sonnet",
                cost_limit=0.10
            )
            
            # Should handle the error gracefully
            assert isinstance(result, str)
            # Could be a budget error (due to estimation failure) or other error
            assert "❌" in result

    @pytest.mark.asyncio
    async def test_model_name_validation(self):
        """Test that model names are properly validated."""
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",  # Valid model name
            primary_response="Test response",
            cost_limit=0.10
        )
        
        # Should process successfully
        assert "# 🔍 Second Opinion Analysis" in result
        assert "anthropic/claude-3-5-sonnet" in result

    @pytest.mark.asyncio 
    async def test_prompt_sanitization(self):
        """Test that prompts are properly sanitized."""
        # Test with a prompt that contains code but no malicious content
        safe_prompt = "How do I use `git status` to check repository state?"
        
        result = await second_opinion_tool(
            prompt=safe_prompt,
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Use git status to see current state",
            cost_limit=0.10
        )
        
        # Should process code-related prompts successfully  
        assert "# 🔍 Second Opinion Analysis" in result
        assert "git status" in result

    @pytest.mark.asyncio
    async def test_cost_reporting(self):
        """Test that costs are properly tracked and reported."""
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Test response",
            cost_limit=0.25
        )
        
        # Should show cost information
        assert "Cost Analysis" in result
        assert "Total Cost" in result
        assert "Cost Limit" in result
        assert "$" in result  # Should show dollar amounts

    @pytest.mark.asyncio
    async def test_recommendations_generation(self):
        """Test that recommendations are generated."""
        result = await second_opinion_tool(
            prompt="Test prompt",
            primary_model="anthropic/claude-3-5-sonnet",
            primary_response="Test response",
            cost_limit=0.20
        )
        
        # Should include recommendations
        assert "Recommendations" in result
        assert ("✅" in result or "💡" in result or "💰" in result)  # Should have recommendation indicators