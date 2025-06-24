"""Tests for template loader utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.second_opinion.utils.template_loader import (
    clear_template_cache,
    get_cache_stats,
    get_prompts_root,
    list_available_templates,
    load_evaluation_template,
    load_mcp_parameter_examples,
    load_mcp_tool_description,
    load_system_template,
    load_template,
)


class TestTemplateLoader:
    """Test template loading functionality."""

    def test_get_prompts_root_path_calculation(self):
        """Test that prompts root path is calculated correctly."""
        # This test verifies the path calculation logic
        with patch("pathlib.Path.exists", return_value=True):
            root = get_prompts_root()
            assert isinstance(root, Path)
            assert root.name == "prompts"

    def test_get_prompts_root_not_found(self):
        """Test error when prompts root doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                get_prompts_root()
            assert "Prompts directory not found" in str(exc_info.value)

    def test_load_template_with_temp_file(self):
        """Test loading template from temporary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary prompts structure
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "test_template.txt"
            template_content = "This is a test template with {placeholder}"
            template_file.write_text(template_content, encoding="utf-8")

            # Mock the prompts root to point to our temp directory
            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                clear_template_cache()  # Start fresh
                result = load_template(
                    "test_category", "test_template", use_cache=False
                )
                assert result == template_content

    def test_load_template_with_caching(self):
        """Test template caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "cached_template.txt"
            template_content = "Cached template content"
            template_file.write_text(template_content, encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                clear_template_cache()

                # First load - should read from file
                result1 = load_template(
                    "test_category", "cached_template", use_cache=True
                )
                assert result1 == template_content

                # Verify it's cached
                stats = get_cache_stats()
                assert stats["cached_templates"] == 1
                assert stats["cache_size_chars"] == len(template_content)

                # Second load - should use cache (even if file changes)
                template_file.write_text("Changed content", encoding="utf-8")
                result2 = load_template(
                    "test_category", "cached_template", use_cache=True
                )
                assert result2 == template_content  # Should still be original content

    def test_load_template_without_caching(self):
        """Test loading template without caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "no_cache_template.txt"
            template_file.write_text("Original content", encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                clear_template_cache()

                # First load
                result1 = load_template(
                    "test_category", "no_cache_template", use_cache=False
                )
                assert result1 == "Original content"

                # Verify nothing is cached
                stats = get_cache_stats()
                assert stats["cached_templates"] == 0

                # Change file and load again
                template_file.write_text("Updated content", encoding="utf-8")
                result2 = load_template(
                    "test_category", "no_cache_template", use_cache=False
                )
                assert result2 == "Updated content"

    def test_load_template_file_not_found(self):
        """Test error when template file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            # Create a different file to test the available templates listing
            other_file = category_dir / "other_template.txt"
            other_file.write_text("Other template", encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                with pytest.raises(FileNotFoundError) as exc_info:
                    load_template("test_category", "nonexistent_template")

                error_msg = str(exc_info.value)
                assert "Template not found" in error_msg
                assert "Available templates" in error_msg
                assert "other_template" in error_msg

    def test_load_template_read_error(self):
        """Test error handling when file can't be read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "error_template.txt"
            template_file.write_text("Template content", encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                with patch("builtins.open", side_effect=OSError("Permission denied")):
                    with pytest.raises(OSError) as exc_info:
                        load_template("test_category", "error_template")

                    assert "Failed to read template" in str(exc_info.value)
                    assert "Permission denied" in str(exc_info.value)

    def test_list_available_templates(self):
        """Test listing available templates in a category."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            # Create some template files
            (category_dir / "template1.txt").write_text("Template 1", encoding="utf-8")
            (category_dir / "template2.txt").write_text("Template 2", encoding="utf-8")

            # Create subdirectory with template
            subdir = category_dir / "subdir"
            subdir.mkdir()
            (subdir / "template3.txt").write_text("Template 3", encoding="utf-8")

            # Create non-txt file (should be ignored)
            (category_dir / "readme.md").write_text("Not a template", encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                templates = list_available_templates("test_category")

                assert "template1" in templates
                assert "template2" in templates
                assert (
                    "subdir/template3" in templates
                )  # Should include subdirectory path
                assert "readme" not in templates  # Should exclude non-txt files
                assert templates == sorted(templates)  # Should be sorted

    def test_list_available_templates_nonexistent_category(self):
        """Test listing templates in nonexistent category."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            prompts_root.mkdir(parents=True)

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                templates = list_available_templates("nonexistent_category")
                assert templates == []

    def test_list_available_templates_error_handling(self):
        """Test error handling in list_available_templates."""
        with patch(
            "src.second_opinion.utils.template_loader.get_prompts_root",
            side_effect=Exception("Test error"),
        ):
            templates = list_available_templates("test_category")
            assert templates == []

    def test_clear_template_cache(self):
        """Test clearing the template cache."""
        # Manually add something to cache
        from src.second_opinion.utils.template_loader import _template_cache

        _template_cache["test/template"] = "test content"

        stats_before = get_cache_stats()
        assert stats_before["cached_templates"] == 1

        clear_template_cache()

        stats_after = get_cache_stats()
        assert stats_after["cached_templates"] == 0
        assert stats_after["cache_size_chars"] == 0

    def test_get_cache_stats(self):
        """Test cache statistics."""
        clear_template_cache()

        # Add test data to cache
        from src.second_opinion.utils.template_loader import _template_cache

        _template_cache["template1"] = "content1"
        _template_cache["template2"] = "longer content"

        stats = get_cache_stats()
        assert stats["cached_templates"] == 2
        assert stats["cache_size_chars"] == len("content1") + len("longer content")


class TestConvenienceFunctions:
    """Test convenience functions for common template categories."""

    @patch("src.second_opinion.utils.template_loader.load_template")
    def test_load_evaluation_template(self, mock_load_template):
        """Test load_evaluation_template convenience function."""
        mock_load_template.return_value = "evaluation template content"

        result = load_evaluation_template("comparison")

        mock_load_template.assert_called_once_with("evaluation", "comparison")
        assert result == "evaluation template content"

    @patch("src.second_opinion.utils.template_loader.load_template")
    def test_load_mcp_tool_description(self, mock_load_template):
        """Test load_mcp_tool_description convenience function."""
        mock_load_template.return_value = "tool description content"

        result = load_mcp_tool_description("second_opinion")

        mock_load_template.assert_called_once_with(
            "mcp", "tool_descriptions/second_opinion"
        )
        assert result == "tool description content"

    @patch("src.second_opinion.utils.template_loader.load_template")
    def test_load_system_template(self, mock_load_template):
        """Test load_system_template convenience function."""
        mock_load_template.return_value = "system template content"

        result = load_system_template("followup_evaluation")

        mock_load_template.assert_called_once_with("system", "followup_evaluation")
        assert result == "system template content"

    @patch("src.second_opinion.utils.template_loader.load_template")
    def test_load_mcp_parameter_examples(self, mock_load_template):
        """Test load_mcp_parameter_examples convenience function."""
        mock_load_template.return_value = "parameter examples content"

        result = load_mcp_parameter_examples()

        mock_load_template.assert_called_once_with("mcp", "parameter_examples")
        assert result == "parameter examples content"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_template_with_whitespace_trimming(self):
        """Test that templates are properly trimmed of leading/trailing whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "whitespace_template.txt"
            template_content = "\n\n  Template with whitespace  \n\n"
            template_file.write_text(template_content, encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                result = load_template(
                    "test_category", "whitespace_template", use_cache=False
                )
                assert result == "Template with whitespace"

    def test_empty_template_file(self):
        """Test handling of empty template files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "empty_template.txt"
            template_file.write_text("", encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                result = load_template(
                    "test_category", "empty_template", use_cache=False
                )
                assert result == ""

    def test_template_with_unicode_content(self):
        """Test handling of templates with unicode content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "unicode_template.txt"
            unicode_content = "Template with émojis 🚀 and spëcial çharacters"
            template_file.write_text(unicode_content, encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                result = load_template(
                    "test_category", "unicode_template", use_cache=False
                )
                assert result == unicode_content

    def test_large_template_file(self):
        """Test handling of large template files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_root = Path(temp_dir) / "prompts"
            category_dir = prompts_root / "test_category"
            category_dir.mkdir(parents=True)

            template_file = category_dir / "large_template.txt"
            large_content = "Large template content " * 1000  # ~20KB
            template_file.write_text(large_content, encoding="utf-8")

            with patch(
                "src.second_opinion.utils.template_loader.get_prompts_root",
                return_value=prompts_root,
            ):
                result = load_template(
                    "test_category", "large_template", use_cache=False
                )
                # Template loader strips trailing whitespace
                assert result == large_content.rstrip()

                # Test that caching works with large files
                clear_template_cache()
                cached_result = load_template(
                    "test_category", "large_template", use_cache=True
                )
                stats = get_cache_stats()
                assert stats["cache_size_chars"] == len(cached_result)


@pytest.fixture(autouse=True)
def cleanup_template_cache():
    """Clean up template cache after each test."""
    yield
    clear_template_cache()
