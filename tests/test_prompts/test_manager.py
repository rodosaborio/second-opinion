"""
Tests for prompt manager functionality.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from second_opinion.core.models import SecurityContext
from second_opinion.prompts.manager import (
    PromptManager,
    PromptTemplate,
    get_prompt_manager,
    get_template_params,
    render_template,
    set_prompt_manager,
)


@pytest.fixture
def temp_templates_dir():
    """Create a temporary directory with test templates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        templates_dir = Path(tmpdir)

        # Create test templates
        (templates_dir / "basic.txt").write_text("Hello {name}, welcome to {service}!")

        (templates_dir / "complex.txt").write_text(
            """Task: {task}
Model: {model}
Parameters: {parameters}
Context: {context}

Your response should consider {criteria} when evaluating."""
        )

        (templates_dir / "no_params.txt").write_text("This template has no parameters.")

        (templates_dir / "empty.txt").write_text("")

        yield templates_dir


@pytest.fixture
def prompt_manager(temp_templates_dir):
    """Create a PromptManager instance with test templates."""
    return PromptManager(templates_dir=temp_templates_dir, cache_ttl=60)


class TestPromptTemplate:
    def test_template_creation(self):
        """Test creating a PromptTemplate."""
        template = PromptTemplate(
            name="test",
            content="Hello {name}",
            parameters=["name"],
            description="Test template",
        )

        assert template.name == "test"
        assert template.content == "Hello {name}"
        assert template.parameters == ["name"]
        assert template.description == "Test template"
        assert template.model_optimizations == {}

    def test_template_with_optimizations(self):
        """Test template with model-specific optimizations."""
        template = PromptTemplate(
            name="test",
            content="Default: {text}",
            parameters=["text"],
            model_optimizations={
                "gpt-4": "GPT-4 optimized: {text}",
                "claude-3": "Claude optimized: {text}",
            },
        )

        assert template.model_optimizations is not None
        assert "gpt-4" in template.model_optimizations  # type: ignore
        assert "claude-3" in template.model_optimizations  # type: ignore


class TestPromptManager:
    @pytest.mark.asyncio
    async def test_initialization(self, temp_templates_dir):
        """Test PromptManager initialization."""
        manager = PromptManager(templates_dir=temp_templates_dir)
        assert manager.templates_dir == temp_templates_dir
        assert manager.cache_ttl == 3600  # Default
        assert manager._template_cache == {}

    @pytest.mark.asyncio
    async def test_load_template_basic(self, prompt_manager):
        """Test loading a basic template."""
        template = await prompt_manager.load_template("basic")

        assert template.name == "basic"
        assert "Hello {name}" in template.content
        assert "name" in template.parameters
        assert "service" in template.parameters
        assert len(template.parameters) == 2

    @pytest.mark.asyncio
    async def test_load_template_complex(self, prompt_manager):
        """Test loading a complex template."""
        template = await prompt_manager.load_template("complex")

        assert template.name == "complex"
        expected_params = ["task", "model", "parameters", "context", "criteria"]
        assert all(param in template.parameters for param in expected_params)

    @pytest.mark.asyncio
    async def test_load_template_no_params(self, prompt_manager):
        """Test loading a template with no parameters."""
        template = await prompt_manager.load_template("no_params")

        assert template.name == "no_params"
        assert template.parameters == []

    @pytest.mark.asyncio
    async def test_load_template_not_found(self, prompt_manager):
        """Test loading a non-existent template."""
        with pytest.raises(FileNotFoundError):
            await prompt_manager.load_template("nonexistent")

    @pytest.mark.asyncio
    async def test_load_template_empty(self, prompt_manager):
        """Test loading an empty template."""
        with pytest.raises(ValueError, match="empty"):
            await prompt_manager.load_template("empty")

    @pytest.mark.asyncio
    async def test_template_caching(self, prompt_manager):
        """Test that templates are cached."""
        # First load
        template1 = await prompt_manager.load_template("basic")

        # Second load should come from cache
        template2 = await prompt_manager.load_template("basic")

        # Should be the same object (cached)
        assert template1 is template2
        assert "basic" in prompt_manager._template_cache

    @pytest.mark.asyncio
    async def test_cache_expiration(self, temp_templates_dir):
        """Test that cache expires after TTL."""
        from datetime import timedelta

        # Create manager with very short TTL
        manager = PromptManager(
            templates_dir=temp_templates_dir, cache_ttl=1
        )  # 1 second

        # Load template
        template1 = await manager.load_template("basic")

        # Manually expire the cache by manipulating the timestamp
        # This simulates time passing beyond the TTL
        old_timestamp = datetime.now(UTC) - timedelta(seconds=1)  # 1 second ago
        manager._cache_timestamps["basic"] = old_timestamp

        # Load again - should reload from file due to expired cache
        template2 = await manager.load_template("basic")

        # Should be different objects (cache expired)
        assert template1 is not template2

    @pytest.mark.asyncio
    async def test_render_prompt_basic(self, prompt_manager):
        """Test rendering a basic prompt."""
        result = await prompt_manager.render_prompt(
            "basic", {"name": "Alice", "service": "SecondOpinion"}
        )

        assert result == "Hello Alice, welcome to SecondOpinion!"

    @pytest.mark.asyncio
    async def test_render_prompt_missing_parameter(self, prompt_manager):
        """Test rendering with missing parameters."""
        with pytest.raises(ValueError, match="Missing required parameters"):
            await prompt_manager.render_prompt("basic", {"name": "Alice"})

    @pytest.mark.asyncio
    async def test_render_prompt_extra_parameters(self, prompt_manager):
        """Test rendering with extra parameters (should be okay)."""
        result = await prompt_manager.render_prompt(
            "basic", {"name": "Alice", "service": "SecondOpinion", "extra": "ignored"}
        )

        assert result == "Hello Alice, welcome to SecondOpinion!"

    @pytest.mark.asyncio
    async def test_render_prompt_with_sanitization(self, prompt_manager):
        """Test that parameters are sanitized and malicious input is blocked."""
        from second_opinion.utils.sanitization import SecurityError

        # Test that malicious script tags are blocked
        with pytest.raises(SecurityError, match="injection attempt"):
            await prompt_manager.render_prompt(
                "basic",
                {
                    "name": "Alice<script>alert('xss')</script>",
                    "service": "SecondOpinion",
                },
            )

        # Test with clean input - should work fine
        result = await prompt_manager.render_prompt(
            "basic", {"name": "Alice", "service": "SecondOpinion"}
        )

        assert "Alice" in result
        assert "SecondOpinion" in result

    @pytest.mark.asyncio
    async def test_get_template_parameters(self, prompt_manager):
        """Test getting template parameters."""
        params = await prompt_manager.get_template_parameters("basic")
        assert "name" in params
        assert "service" in params
        assert len(params) == 2

    @pytest.mark.asyncio
    async def test_validate_parameters_valid(self, prompt_manager):
        """Test parameter validation with valid parameters."""
        valid = await prompt_manager.validate_parameters(
            "basic", {"name": "Alice", "service": "SecondOpinion"}
        )
        assert valid is True

    @pytest.mark.asyncio
    async def test_validate_parameters_missing(self, prompt_manager):
        """Test parameter validation with missing parameters."""
        valid = await prompt_manager.validate_parameters("basic", {"name": "Alice"})
        assert valid is False

    @pytest.mark.asyncio
    async def test_validate_parameters_extra(self, prompt_manager):
        """Test parameter validation with extra parameters."""
        valid = await prompt_manager.validate_parameters(
            "basic", {"name": "Alice", "service": "SecondOpinion", "extra": "value"}
        )
        assert valid is True  # Extra parameters are okay

    @pytest.mark.asyncio
    async def test_list_templates(self, prompt_manager):
        """Test listing available templates."""
        templates = await prompt_manager.list_templates()

        expected = ["basic", "complex", "empty", "no_params"]
        assert all(template in templates for template in expected)

    @pytest.mark.asyncio
    async def test_reload_template(self, prompt_manager):
        """Test reloading a template."""
        # Load template first
        template1 = await prompt_manager.load_template("basic")

        # Reload template
        template2 = await prompt_manager.reload_template("basic")

        # Should be different objects
        assert template1 is not template2
        assert template1.name == template2.name

    @pytest.mark.asyncio
    async def test_clear_cache(self, prompt_manager):
        """Test clearing the template cache."""
        # Load a template to populate cache
        await prompt_manager.load_template("basic")
        assert len(prompt_manager._template_cache) == 1

        # Clear cache
        prompt_manager.clear_cache()
        assert len(prompt_manager._template_cache) == 0
        assert len(prompt_manager._cache_timestamps) == 0

    @pytest.mark.asyncio
    async def test_parameter_extraction_edge_cases(self, temp_templates_dir):
        """Test parameter extraction with edge cases."""
        # Create template with duplicate parameters
        (temp_templates_dir / "duplicates.txt").write_text(
            "Hello {name}, {name} is using {service} and {service}."
        )

        manager = PromptManager(templates_dir=temp_templates_dir)
        template = await manager.load_template("duplicates")

        # Should deduplicate parameters
        assert len(template.parameters) == 2
        assert "name" in template.parameters
        assert "service" in template.parameters


class TestGlobalPromptManager:
    def test_global_manager_singleton(self):
        """Test that global manager maintains singleton behavior."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()

        assert manager1 is manager2

    def test_set_global_manager(self, temp_templates_dir):
        """Test setting a custom global manager."""
        custom_manager = PromptManager(templates_dir=temp_templates_dir)
        set_prompt_manager(custom_manager)

        retrieved_manager = get_prompt_manager()
        assert retrieved_manager is custom_manager

    @pytest.mark.asyncio
    async def test_convenience_functions(self, temp_templates_dir):
        """Test convenience functions."""
        # Set up custom manager
        custom_manager = PromptManager(templates_dir=temp_templates_dir)
        set_prompt_manager(custom_manager)

        # Test render_template convenience function
        result = await render_template(
            "basic", {"name": "Alice", "service": "SecondOpinion"}
        )
        assert result == "Hello Alice, welcome to SecondOpinion!"

        # Test get_template_params convenience function
        params = await get_template_params("basic")
        assert "name" in params
        assert "service" in params


class TestSecurityContexts:
    @pytest.mark.asyncio
    async def test_different_security_contexts(self, prompt_manager):
        """Test rendering with different security contexts."""
        # Test with USER_PROMPT context
        result1 = await prompt_manager.render_prompt(
            "basic",
            {"name": "Alice", "service": "SecondOpinion"},
            security_context=SecurityContext.USER_PROMPT,
        )

        # Test with SYSTEM_PROMPT context
        result2 = await prompt_manager.render_prompt(
            "basic",
            {"name": "Alice", "service": "SecondOpinion"},
            security_context=SecurityContext.SYSTEM_PROMPT,
        )

        # Both should work (actual sanitization behavior depends on sanitization module)
        assert "Alice" in result1
        assert "Alice" in result2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_invalid_template_path(self):
        """Test with invalid templates directory."""
        manager = PromptManager(templates_dir="/nonexistent/path")

        with pytest.raises(FileNotFoundError):
            await manager.load_template("any_template")

    @pytest.mark.asyncio
    async def test_template_read_error(self, temp_templates_dir):
        """Test handling file read errors."""
        manager = PromptManager(templates_dir=temp_templates_dir)

        # Mock file read to raise permission error
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Access denied")

            with pytest.raises(ValueError, match="Failed to read template"):
                await manager.load_template("basic")

    @pytest.mark.asyncio
    async def test_template_format_error(self, prompt_manager):
        """Test handling template format errors."""
        # Create template with invalid format string
        template_path = prompt_manager.templates_dir / "invalid_format.txt"
        template_path.write_text("Hello {name} {invalid_format")

        # This should still load (parameter extraction is robust)
        template = await prompt_manager.load_template("invalid_format")
        assert template is not None

        # But rendering should fail
        with pytest.raises(ValueError, match="Failed to render template"):
            await prompt_manager.render_prompt("invalid_format", {"name": "Alice"})


@pytest.mark.asyncio
async def test_model_specific_optimization(temp_templates_dir):
    """Test model-specific template optimization (future feature)."""
    # This test is for when model-specific optimizations are implemented
    manager = PromptManager(templates_dir=temp_templates_dir)

    # For now, just test that the model parameter is accepted
    result = await manager.render_prompt(
        "basic", {"name": "Alice", "service": "SecondOpinion"}, model="gpt-4"
    )

    assert result == "Hello Alice, welcome to SecondOpinion!"


# Test cleanup
@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset global manager state after each test."""
    yield
    # Reset to a fresh manager
    set_prompt_manager(PromptManager())
