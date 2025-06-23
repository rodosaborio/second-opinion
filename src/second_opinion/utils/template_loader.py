"""
Template loader utility for managing externalized prompts.

This module provides centralized access to prompt templates stored in the
prompts/ directory, with proper caching and error handling.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache for loaded templates
_template_cache: dict[str, str] = {}


def get_prompts_root() -> Path:
    """Get the root prompts directory path."""
    # Get the repository root (second-opinion directory)
    current_file = Path(__file__)
    repo_root = (
        current_file.parent.parent.parent.parent
    )  # src/second_opinion/utils/template_loader.py -> repo root
    prompts_root = repo_root / "prompts"

    if not prompts_root.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_root}")

    return prompts_root


def load_template(category: str, template_name: str, use_cache: bool = True) -> str:
    """
    Load a template from the prompts directory.

    Args:
        category: The category subdirectory (evaluation, mcp, system)
        template_name: The template file name (without .txt extension)
        use_cache: Whether to use cached templates for performance

    Returns:
        The template content as a string

    Raises:
        FileNotFoundError: If the template file doesn't exist
        IOError: If there's an error reading the file

    Examples:
        # Load evaluation template
        template = load_template("evaluation", "comparison")

        # Load MCP tool description
        description = load_template("mcp", "tool_descriptions/second_opinion")

        # Load system prompt
        prompt = load_template("system", "followup_evaluation")
    """
    cache_key = f"{category}/{template_name}"

    # Check cache first if enabled
    if use_cache and cache_key in _template_cache:
        logger.debug(f"Using cached template: {cache_key}")
        return _template_cache[cache_key]

    # Build file path
    prompts_root = get_prompts_root()
    template_path = prompts_root / category / f"{template_name}.txt"

    # Check if file exists
    if not template_path.exists():
        available_templates = list_available_templates(category)
        available_str = "\n".join(f"  - {t}" for t in available_templates)
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            f"Available templates in '{category}':\n{available_str}"
        )

    try:
        # Read template content
        with open(template_path, encoding="utf-8") as f:
            content = f.read().strip()

        # Cache the content if caching is enabled
        if use_cache:
            _template_cache[cache_key] = content
            logger.debug(f"Cached template: {cache_key}")

        logger.debug(f"Loaded template: {cache_key} ({len(content)} chars)")
        return content

    except Exception as e:
        logger.error(f"Error reading template {template_path}: {e}")
        raise OSError(f"Failed to read template {template_path}: {e}") from e


def list_available_templates(category: str) -> list[str]:
    """
    List all available templates in a category.

    Args:
        category: The category subdirectory to list

    Returns:
        List of available template names (without .txt extension)
    """
    try:
        prompts_root = get_prompts_root()
        category_path = prompts_root / category

        if not category_path.exists():
            return []

        templates = []
        for item in category_path.rglob("*.txt"):
            # Get relative path from category root and remove .txt extension
            relative_path = item.relative_to(category_path)
            template_name = str(relative_path).replace(".txt", "")
            templates.append(template_name)

        return sorted(templates)

    except Exception as e:
        logger.warning(f"Error listing templates in category '{category}': {e}")
        return []


def clear_template_cache() -> None:
    """Clear the template cache. Useful for testing or reloading templates."""
    global _template_cache
    _template_cache.clear()
    logger.debug("Template cache cleared")


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the template cache."""
    return {
        "cached_templates": len(_template_cache),
        "cache_size_chars": sum(len(content) for content in _template_cache.values()),
    }


# Convenience functions for common template categories


def load_evaluation_template(template_name: str) -> str:
    """Load an evaluation template."""
    return load_template("evaluation", template_name)


def load_mcp_tool_description(tool_name: str) -> str:
    """Load an MCP tool description."""
    return load_template("mcp", f"tool_descriptions/{tool_name}")


def load_system_template(template_name: str) -> str:
    """Load a system template."""
    return load_template("system", template_name)


def load_mcp_parameter_examples() -> str:
    """Load the MCP parameter examples."""
    return load_template("mcp", "parameter_examples")
