"""
Prompt template management system.

This module provides functionality for loading, caching, and rendering prompt templates
with parameter injection and model-specific optimizations.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.models import SecurityContext
from ..utils.sanitization import sanitize_prompt

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A loaded prompt template with metadata."""
    name: str
    content: str
    parameters: list[str]
    description: str | None = None
    model_optimizations: dict[str, str] = None
    last_modified: datetime | None = None

    def __post_init__(self):
        if self.model_optimizations is None:
            self.model_optimizations = {}


class PromptManager:
    """
    Manages prompt templates with loading, caching, and parameter injection.
    
    Features:
    - Template loading from files with caching
    - Parameter validation and injection
    - Model-specific prompt optimizations
    - Security-aware template rendering
    """

    def __init__(self, templates_dir: str | Path | None = None, cache_ttl: int = 3600):
        """
        Initialize the prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates (defaults to project templates)
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        if templates_dir is None:
            # Default to project templates directory
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            templates_dir = project_root / "prompts" / "templates"

        self.templates_dir = Path(templates_dir)
        self.cache_ttl = cache_ttl
        self._template_cache: dict[str, PromptTemplate] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Parameter extraction pattern for {parameter_name} format
        self._param_pattern = re.compile(r'\{([^}]+)\}')

        logger.info(f"Initialized PromptManager with templates_dir: {self.templates_dir}")

    async def load_template(self, template_name: str) -> PromptTemplate:
        """
        Load a template by name with caching.
        
        Args:
            template_name: Name of the template (without .txt extension)
            
        Returns:
            Loaded prompt template
            
        Raises:
            FileNotFoundError: Template file not found
            ValueError: Template content is invalid
        """
        # Check cache first
        cached_template = await self._get_cached_template(template_name)
        if cached_template:
            return cached_template

        # Load from file
        template_path = self.templates_dir / f"{template_name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        try:
            with open(template_path, encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read template {template_name}: {e}")

        if not content.strip():
            raise ValueError(f"Template {template_name} is empty")

        # Extract parameters from template
        parameters = self._extract_parameters(content)

        # Get file modification time
        stat = template_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

        # Create template object
        template = PromptTemplate(
            name=template_name,
            content=content,
            parameters=parameters,
            last_modified=last_modified
        )

        # Cache the template
        await self._cache_template(template_name, template)

        logger.debug(f"Loaded template '{template_name}' with parameters: {parameters}")
        return template

    async def render_prompt(
        self,
        template_name: str,
        parameters: dict[str, Any],
        model: str | None = None,
        security_context: SecurityContext = SecurityContext.USER_PROMPT
    ) -> str:
        """
        Render a template with parameter injection.
        
        Args:
            template_name: Name of the template to render
            parameters: Parameters to inject into the template
            model: Target model for optimizations (optional)
            security_context: Security context for sanitization
            
        Returns:
            Rendered prompt string
            
        Raises:
            FileNotFoundError: Template not found
            ValueError: Missing required parameters or invalid template
        """
        template = await self.load_template(template_name)

        # Check for missing required parameters
        missing_params = set(template.parameters) - set(parameters.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters for template '{template_name}': {missing_params}")

        # Use model-specific optimization if available
        content = template.content
        if model and model in template.model_optimizations:
            content = template.model_optimizations[model]
            logger.debug(f"Using model-specific optimization for {model}")

        # Sanitize parameter values
        sanitized_params = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                sanitized_params[key] = sanitize_prompt(value, security_context)
            else:
                sanitized_params[key] = str(value)

        # Render template with parameters
        try:
            rendered = content.format(**sanitized_params)
        except KeyError as e:
            raise ValueError(f"Template '{template_name}' contains undefined parameter: {e}")
        except Exception as e:
            raise ValueError(f"Failed to render template '{template_name}': {e}")

        logger.debug(f"Rendered template '{template_name}' for model '{model}'")
        return rendered

    async def get_template_parameters(self, template_name: str) -> list[str]:
        """
        Get the list of parameters required by a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            List of parameter names
        """
        template = await self.load_template(template_name)
        return template.parameters.copy()

    async def validate_parameters(self, template_name: str, parameters: dict[str, Any]) -> bool:
        """
        Validate that all required parameters are provided.
        
        Args:
            template_name: Name of the template
            parameters: Parameters to validate
            
        Returns:
            True if all required parameters are present
        """
        template = await self.load_template(template_name)
        required_params = set(template.parameters)
        provided_params = set(parameters.keys())
        return required_params.issubset(provided_params)

    async def list_templates(self) -> list[str]:
        """
        List available template names.
        
        Returns:
            List of template names (without .txt extension)
        """
        if not self.templates_dir.exists():
            return []

        templates = []
        for file_path in self.templates_dir.glob("*.txt"):
            if file_path.is_file():
                templates.append(file_path.stem)

        return sorted(templates)

    async def reload_template(self, template_name: str) -> PromptTemplate:
        """
        Force reload a template, bypassing cache.
        
        Args:
            template_name: Name of the template to reload
            
        Returns:
            Reloaded template
        """
        # Remove from cache
        self._template_cache.pop(template_name, None)
        self._cache_timestamps.pop(template_name, None)

        # Load fresh copy
        return await self.load_template(template_name)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Template cache cleared")

    async def _get_cached_template(self, template_name: str) -> PromptTemplate | None:
        """Get template from cache if valid."""
        if template_name not in self._template_cache:
            return None

        cache_timestamp = self._cache_timestamps.get(template_name)
        if not cache_timestamp:
            return None

        # Check if cache is still valid
        age = (datetime.now(UTC) - cache_timestamp).total_seconds()
        if age > self.cache_ttl:
            # Cache expired
            self._template_cache.pop(template_name, None)
            self._cache_timestamps.pop(template_name, None)
            return None

        return self._template_cache[template_name]

    async def _cache_template(self, template_name: str, template: PromptTemplate) -> None:
        """Cache a template with timestamp."""
        self._template_cache[template_name] = template
        self._cache_timestamps[template_name] = datetime.now(UTC)

    def _extract_parameters(self, content: str) -> list[str]:
        """Extract parameter names from template content."""
        matches = self._param_pattern.findall(content)
        return sorted(list(set(matches)))  # Remove duplicates and sort


# Global prompt manager instance
_global_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Returns:
        Global PromptManager instance
    """
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    return _global_prompt_manager


def set_prompt_manager(manager: PromptManager) -> None:
    """
    Set the global prompt manager instance.
    
    Args:
        manager: PromptManager instance to set as global
    """
    global _global_prompt_manager
    _global_prompt_manager = manager


async def render_template(
    template_name: str,
    parameters: dict[str, Any],
    model: str | None = None,
    security_context: SecurityContext = SecurityContext.USER_PROMPT
) -> str:
    """
    Convenience function to render a template using the global manager.
    
    Args:
        template_name: Name of the template to render
        parameters: Parameters to inject
        model: Target model for optimizations
        security_context: Security context for sanitization
        
    Returns:
        Rendered prompt string
    """
    manager = get_prompt_manager()
    return await manager.render_prompt(template_name, parameters, model, security_context)


async def get_template_params(template_name: str) -> list[str]:
    """
    Convenience function to get template parameters using the global manager.
    
    Args:
        template_name: Name of the template
        
    Returns:
        List of parameter names
    """
    manager = get_prompt_manager()
    return await manager.get_template_parameters(template_name)
