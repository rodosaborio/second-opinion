"""
Input sanitization and validation utilities for Second Opinion.

This module provides multi-layer input validation and sanitization to ensure
secure handling of user prompts, API requests, and configuration data.
"""

import re
from decimal import Decimal
from typing import Any

from ..core.models import SecurityContext


class SecurityError(Exception):
    """Security-related validation error."""

    pass


class ValidationError(Exception):
    """General validation error."""

    pass


class InputSanitizer:
    """Multi-layer input validation and sanitization."""

    # Maximum lengths for different input types
    MAX_PROMPT_LENGTH = 50000
    MAX_MODEL_NAME_LENGTH = 100
    MAX_SYSTEM_PROMPT_LENGTH = 10000

    # Patterns for detecting potential security issues
    API_KEY_PATTERNS = [
        r"sk-[a-zA-Z0-9-_]{20,}",  # OpenAI/Anthropic style
        r"sk-or-[a-zA-Z0-9-_]{20,}",  # OpenRouter style
        r"sk-ant-[a-zA-Z0-9-_]{20,}",  # Anthropic style
        r"(?<![a-zA-Z0-9])[a-zA-Z0-9]{40,}(?![a-zA-Z0-9])",  # Generic long tokens with boundaries
        r"Bearer\s+[a-zA-Z0-9\-_=]{20,}",  # Bearer tokens
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{20,}',  # API key assignments
    ]

    # Patterns for potentially malicious content
    INJECTION_PATTERNS = [
        r"<script[^>]*>",  # Script tags (opening)
        r"</script>",  # Script tags (closing)
        r"javascript:",  # JavaScript URLs
        r"data:.*base64",  # Base64 data URLs
        r"\\x[0-9a-fA-F]{2}",  # Hex escapes
        r"%[0-9a-fA-F]{2}",  # URL encoding
        r"UNION\s+SELECT",  # SQL injection attempts
        r"DROP\s+TABLE",  # SQL injection attempts
        r"DELETE\s+FROM",  # SQL injection attempts
        r"INSERT\s+INTO",  # SQL injection attempts
        r"<iframe[^>]*>",  # Iframe tags
        r"`[^`]*`",  # Backtick commands
        r"\$\([^)]*\)",  # Command substitution
    ]

    def __init__(self) -> None:
        self._api_key_regex = re.compile("|".join(self.API_KEY_PATTERNS), re.IGNORECASE)
        self._injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS), re.IGNORECASE
        )

    def sanitize_prompt(
        self, prompt: str, context: SecurityContext = SecurityContext.USER_PROMPT
    ) -> str:
        """
        Sanitize user prompts for safe API consumption.

        Args:
            prompt: The user prompt to sanitize
            context: Security context for validation level

        Returns:
            Sanitized prompt

        Raises:
            SecurityError: If potential security issues are detected
            ValidationError: If validation fails
        """
        if not isinstance(prompt, str):
            raise ValidationError("Prompt must be a string")

        # Check for potential API keys
        if self._contains_api_key_pattern(prompt):
            raise SecurityError(
                "Potential API key or sensitive token detected in prompt"
            )

        # Check for injection attempts (context-aware)
        if self._contains_injection_pattern(prompt, context):
            raise SecurityError("Potential injection attempt detected in prompt")

        # Normalize whitespace
        prompt = self._normalize_whitespace(prompt)

        # Check size limits based on context
        max_length = self._get_max_length_for_context(context)
        if len(prompt) > max_length:
            raise ValidationError(
                f"Prompt exceeds maximum length of {max_length} characters"
            )

        # Additional sanitization based on context
        if context == SecurityContext.SYSTEM_PROMPT:
            prompt = self._sanitize_system_prompt(prompt)
        elif context == SecurityContext.API_REQUEST:
            prompt = self._sanitize_api_request(prompt)

        return prompt

    def validate_model_name(self, model_name: str) -> str:
        """
        Validate and normalize model names with intelligent format conversion.

        Args:
            model_name: The model name to validate

        Returns:
            Normalized model name in standard format

        Raises:
            ValidationError: If model name is invalid
        """
        if not isinstance(model_name, str):
            raise ValidationError("Model name must be a string")

        # Remove extra whitespace
        original_model_name = model_name.strip()

        if not original_model_name:
            raise ValidationError("Model name cannot be empty")

        if len(original_model_name) > self.MAX_MODEL_NAME_LENGTH:
            raise ValidationError(
                f"Model name exceeds maximum length of {self.MAX_MODEL_NAME_LENGTH}"
            )

        # Check for injection patterns BEFORE normalization to prevent bypassing security
        # Use CONFIGURATION context for stricter validation of model names
        if self._contains_injection_pattern(
            original_model_name, SecurityContext.CONFIGURATION
        ):
            raise SecurityError("Potential injection attempt in model name")

        # Try to normalize common model name patterns
        normalized_name = self._normalize_model_name(original_model_name)

        # Check for valid model name format (provider/model or just model)
        # Allow alphanumeric start, alphanumeric/hyphen/underscore/dot/slash/colon in middle
        if not re.match(
            r"^[a-zA-Z0-9][a-zA-Z0-9\-_./:]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$", normalized_name
        ):
            raise ValidationError(
                f"Model name '{original_model_name}' contains invalid characters. Normalized to '{normalized_name}' but still invalid."
            )

        # Double-check for injection patterns after normalization as well
        if self._contains_injection_pattern(
            normalized_name, SecurityContext.CONFIGURATION
        ):
            raise SecurityError("Potential injection attempt in model name")

        return normalized_name

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model names to standard OpenRouter format.

        Args:
            model_name: Original model name

        Returns:
            Normalized model name
        """
        # Convert common display names to OpenRouter format
        normalization_map = {
            # Claude variants
            "claude": "anthropic/claude-3-5-sonnet",
            "claude 3.5 sonnet": "anthropic/claude-3-5-sonnet",
            "claude-3.5-sonnet": "anthropic/claude-3-5-sonnet",
            "claude 3 haiku": "anthropic/claude-3-haiku",
            "claude-3-haiku": "anthropic/claude-3-haiku",
            # ChatGPT variants
            "gpt-4o": "openai/gpt-4o",
            "gpt 4o": "openai/gpt-4o",
            "chatgpt": "openai/gpt-4o",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt 4o mini": "openai/gpt-4o-mini",
            "gpt4o": "openai/gpt-4o",
            "gpt4o-mini": "openai/gpt-4o-mini",
            # Gemini variants
            "gemini": "google/gemini-pro-1.5",
            "gemini pro": "google/gemini-pro-1.5",
            "gemini 1.5 pro": "google/gemini-pro-1.5",
            "gemini-1.5-pro": "google/gemini-pro-1.5",
            "gemini 2.5 flash": "google/gemini-2.5-flash-preview",  # Updated for current model
            "gemini 2.5": "google/gemini-2.5-flash-preview",
            "gemini flash": "google/gemini-flash-1.5",
            "gemini-flash": "google/gemini-flash-1.5",
            "gemini-flash-1.5": "google/gemini-flash-1.5",
            # Local model variants (keep as-is, just normalize spacing/hyphens)
            "qwen 3 4b mlx": "qwen3-4b-mlx",
            "qwen3 4b mlx": "qwen3-4b-mlx",
            "qwen 3 0.6b mlx": "qwen3-0.6b-mlx",
            "qwen3 0.6b mlx": "qwen3-0.6b-mlx",
            "codestral 22b": "codestral-22b-v0.1",
            "codestral-22b": "codestral-22b-v0.1",
        }

        # If already in provider/model format, keep as-is (check first to avoid unwanted normalization)
        if "/" in model_name and re.match(
            r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*\/[a-zA-Z0-9][a-zA-Z0-9\-_.]*$", model_name
        ):
            return model_name

        # Convert to lowercase for mapping lookup
        lower_name = model_name.lower().strip()

        # Check exact match first
        if lower_name in normalization_map:
            return normalization_map[lower_name]

        # Check partial matches for common patterns
        for pattern, normalized in normalization_map.items():
            if pattern in lower_name:
                return normalized

        # For local models, normalize spacing and special characters
        if any(
            keyword in lower_name
            for keyword in ["qwen", "codestral", "llama", "mlx", "gguf"]
        ):
            # Replace spaces with hyphens, keep alphanumeric, hyphens, dots
            normalized = re.sub(r"[^a-zA-Z0-9\-.]", "-", model_name)
            # Remove multiple consecutive hyphens
            normalized = re.sub(r"-+", "-", normalized)
            # Remove leading/trailing hyphens
            normalized = normalized.strip("-")
            return normalized

        # Return original if no normalization needed
        return model_name

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize metadata dictionary for safe storage and processing.

        Args:
            metadata: Dictionary of metadata to sanitize

        Returns:
            Sanitized metadata dictionary
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            # Sanitize keys
            if not isinstance(key, str):
                continue

            clean_key = self._sanitize_string_value(key)
            if not clean_key or len(clean_key) > 100:
                continue

            # Sanitize values
            if isinstance(value, str):
                clean_value = self._sanitize_string_value(value)
                if clean_value and len(clean_value) <= 1000:
                    sanitized[clean_key] = clean_value
            elif isinstance(value, int | float | bool):
                sanitized[clean_key] = value
            elif isinstance(value, Decimal):
                sanitized[clean_key] = value

        return sanitized

    def validate_cost_limit(self, cost_limit: str | float | Decimal) -> Decimal:
        """
        Validate and normalize cost limit values.

        Args:
            cost_limit: Cost limit to validate

        Returns:
            Normalized cost limit as Decimal

        Raises:
            ValidationError: If cost limit is invalid
        """
        if cost_limit is None:
            raise ValidationError("Invalid cost limit format")

        try:
            if isinstance(cost_limit, str):
                # Remove any currency symbols or spaces
                cleaned = re.sub(r"[^\d.-]", "", cost_limit)
                if not cleaned or cleaned in [".", "-", "-."]:
                    raise ValueError("Empty or invalid string")
                cost_decimal = Decimal(cleaned)
            else:
                cost_decimal = Decimal(str(cost_limit))
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ValidationError("Invalid cost limit format") from e

        if cost_decimal < 0:
            raise ValidationError("Cost limit cannot be negative")

        if cost_decimal > Decimal("1000"):
            raise ValidationError("Cost limit exceeds maximum allowed value of $1000")

        # Ensure reasonable precision (max 4 decimal places)
        return cost_decimal.quantize(Decimal("0.0001"))

    def _contains_api_key_pattern(self, text: str) -> bool:
        """Check if text contains potential API key patterns."""
        return bool(self._api_key_regex.search(text))

    def _contains_injection_pattern(
        self, text: str, context: SecurityContext = SecurityContext.USER_PROMPT
    ) -> bool:
        """Check if text contains potential injection patterns."""
        # For user prompts, be more permissive to allow code snippets
        if context == SecurityContext.USER_PROMPT:
            # Only check for serious injection attempts, allow code patterns
            serious_patterns = [
                r"<script[^>]*>",  # Script tags (opening)
                r"</script>",  # Script tags (closing)
                r"javascript:",  # JavaScript URLs
                r"data:.*base64",  # Base64 data URLs
                r"<iframe[^>]*>",  # Iframe tags
                r"UNION\s+SELECT",  # SQL injection attempts
                r"DROP\s+TABLE",  # SQL injection attempts
                r"DELETE\s+FROM",  # SQL injection attempts
                r"INSERT\s+INTO",  # SQL injection attempts
                r"\\x[0-9a-fA-F]{2}",  # Hex escapes
                r"%[0-9a-fA-F]{2}",  # URL encoding
            ]
            serious_regex = re.compile("|".join(serious_patterns), re.IGNORECASE)
            return bool(serious_regex.search(text))

        # For system prompts, we're more lenient with command-like patterns since they'll be sanitized
        elif context == SecurityContext.SYSTEM_PROMPT:
            # Check for serious injection attempts but allow backticks (they'll be sanitized)
            serious_patterns = [
                r"<script[^>]*>",  # Script tags (opening)
                r"</script>",  # Script tags (closing)
                r"javascript:",  # JavaScript URLs
                r"data:.*base64",  # Base64 data URLs
                r"<iframe[^>]*>",  # Iframe tags
            ]
            serious_regex = re.compile("|".join(serious_patterns), re.IGNORECASE)
            return bool(serious_regex.search(text))

        # For other contexts (API_REQUEST, CONFIGURATION), check all patterns
        return bool(self._injection_regex.search(text))

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Replace multiple spaces with single space, but preserve newlines and tabs
        text = re.sub(r"[ ]+", " ", text)

        # Remove null bytes and control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text

    def _get_max_length_for_context(self, context: SecurityContext) -> int:
        """Get maximum length based on security context."""
        return {
            SecurityContext.USER_PROMPT: self.MAX_PROMPT_LENGTH,
            SecurityContext.SYSTEM_PROMPT: self.MAX_SYSTEM_PROMPT_LENGTH,
            SecurityContext.API_REQUEST: self.MAX_PROMPT_LENGTH,
            SecurityContext.CONFIGURATION: 1000,
        }.get(context, self.MAX_PROMPT_LENGTH)

    def _sanitize_system_prompt(self, prompt: str) -> str:
        """Additional sanitization for system prompts."""
        # Remove potential command injection attempts
        dangerous_patterns = [
            r"`[^`]*`",  # Backtick commands
            r"\$\([^)]*\)",  # Command substitution
            r">\s*[/\\]",  # File redirection
        ]

        for pattern in dangerous_patterns:
            prompt = re.sub(pattern, "[REMOVED]", prompt, flags=re.IGNORECASE)

        return prompt

    def _sanitize_api_request(self, prompt: str) -> str:
        """Additional sanitization for API requests."""
        # Remove potential API manipulation attempts
        api_patterns = [
            r"Content-Type:\s*[^\n]*",
            r"Authorization:\s*[^\n]*",
            r"X-API-Key:\s*[^\n]*",
            r"Cookie:\s*[^\n]*",
        ]

        for pattern in api_patterns:
            prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)

        return prompt

    def _sanitize_string_value(self, value: str) -> str:
        """Basic string sanitization for metadata values."""
        if not isinstance(value, str):
            return ""

        # Check for potential security issues
        if self._contains_api_key_pattern(value) or self._contains_injection_pattern(
            value, SecurityContext.CONFIGURATION
        ):
            return ""

        # Normalize and return
        return self._normalize_whitespace(value)


# Global sanitizer instance
_sanitizer = InputSanitizer()


def sanitize_prompt(
    prompt: str, context: SecurityContext = SecurityContext.USER_PROMPT
) -> str:
    """Global function to sanitize prompts."""
    return _sanitizer.sanitize_prompt(prompt, context)


def validate_model_name(model_name: str) -> str:
    """Global function to validate model names."""
    return _sanitizer.validate_model_name(model_name)


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Global function to sanitize metadata."""
    return _sanitizer.sanitize_metadata(metadata)


def validate_cost_limit(cost_limit: str | float | Decimal) -> Decimal:
    """Global function to validate cost limits."""
    return _sanitizer.validate_cost_limit(cost_limit)
