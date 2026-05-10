"""
Test OpenCode Exporter - Unit tests for opencode_exporter.py

Tests the export functionality without requiring DSPy modules.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.export.opencode_exporter import OpenCodeExporter


class TestOpenCodeExporter:
    """Tests for OpenCodeExporter class."""

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create an exporter with temporary output directory."""
        return OpenCodeExporter(output_dir=str(tmp_path))

    def test_init(self, tmp_path):
        """Test exporter initialization."""
        exporter = OpenCodeExporter(output_dir=str(tmp_path / "test_output"))
        assert exporter.output_dir.exists()

    def test_extract_instruction_prompt_demos(self, exporter):
        """Test extracting instruction prompt with demos."""
        demo1 = MagicMock()
        demo1.toDict.return_value = {"task": "demo1"}
        demo2 = MagicMock()
        demo2.toDict.return_value = {"task": "demo2"}
        module = MagicMock()
        module.named_predictors.return_value = [
            ("planner", MagicMock(demos=[demo1, demo2]))
        ]

        result = exporter.extract_instruction_prompt(module)

        assert "Few-Shot Demonstrations" in result
        assert "Example 1" in result
        assert "Example 2" in result

    def test_extract_instruction_prompt_planner_demos(self, exporter):
        """Test extracting from planner demos."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.planner = MagicMock()
        demo_mock = MagicMock()
        demo_mock.toDict.return_value = {"task": "demo"}
        module.planner.demos = [demo_mock]

        result = exporter.extract_instruction_prompt(module)

        assert "Few-Shot Demonstrations" in result

    def test_extract_instruction_prompt_signature_docstring(self, exporter):
        """Test extracting from signature docstring."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt instructions"

        result = exporter.extract_instruction_prompt(module)

        assert result == "Test prompt instructions"

    def test_extract_instruction_prompt_signature_instructions(self, exporter):
        """Test extracting from signature instructions."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = None
        module.planner.signature.instructions = "Test instructions"

        result = exporter.extract_instruction_prompt(module)

        assert result == "Test instructions"

    def test_extract_instruction_prompt_fallback(self, exporter):
        """Test fallback when no instructions found."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = None
        module.planner.signature.instructions = None
        module.planner.signature.input_fields = {}
        module.planner.signature.output_fields = {}

        result = exporter.extract_instruction_prompt(module)

        assert "Module Configuration" in result
        assert "did not produce few-shot demonstrations" in result

    def test_format_demos(self, exporter):
        """Test formatting demonstrations."""
        demo1 = MagicMock()
        demo1.toDict.return_value = {"task_description": "Task 1", "_private": "hidden"}
        demo2 = MagicMock()
        demo2.toDict.return_value = {"task_description": "Task 2"}

        result = exporter._format_demos([demo1, demo2])

        assert "Example 1" in result
        assert "Example 2" in result
        assert "Task 1" in result
        assert "Task 2" in result
        assert "_private" not in result  # Private fields skipped

    def test_format_module_structure(self, exporter):
        """Test formatting module structure."""
        module = MagicMock()
        module.planner = MagicMock()
        module.planner.signature = MagicMock()
        module.planner.signature.input_fields = {
            "task": MagicMock(desc="Task description"),
            "context": MagicMock(desc="Environment context")
        }
        module.planner.signature.output_fields = {
            "action": MagicMock(desc="First action")
        }

        result = exporter._format_module_structure(module)

        assert "Module Configuration" in result
        assert "Task description" in result
        assert "First action" in result

    def test_export_agent_config(self, exporter):
        """Test exporting agent configuration."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt"

        path = exporter.export_agent_config(
            optimized_module=module,
            agent_name="build",
            model_name="gpt-4",
            baseline_score=0.5,
            optimized_score=0.75
        )

        assert path.exists()
        assert path.suffix == ".jsonc"
        content = path.read_text()
        assert "build" in content
        assert "gpt-4" in content
        assert "0.500" in content
        assert "0.750" in content

    def test_export_custom_instructions(self, exporter):
        """Test exporting custom instructions."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt"

        path = exporter.export_custom_instructions(module, filename="TEST.md")

        assert path.exists()
        assert path.name == "TEST.md"
        content = path.read_text()
        assert "Optimized Agent Instructions" in content
        assert "Test prompt" in content

    def test_export_prompt_template(self, exporter):
        """Test exporting prompt template."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt"

        path = exporter.export_prompt_template(module, model_name="gpt-4")

        assert path.exists()
        assert "gpt-4" in path.name
        content = path.read_text()
        assert "Test prompt" in content

    def test_export_prompt_template_with_base(self, exporter):
        """Test exporting with base template."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt"

        path = exporter.export_prompt_template(
            module,
            model_name="gpt-4",
            base_template="Base template content"
        )

        content = path.read_text()
        assert "Base template content" in content
        assert "Test prompt" in content

    def test_export_all_formats(self, exporter):
        """Test exporting all formats."""
        module = MagicMock()
        module.named_predictors.return_value = []
        module.demos = None
        module.planner = MagicMock()
        module.planner.demos = []
        module.planner.extended_signature = MagicMock()
        module.planner.extended_signature.demos = []
        module.planner.signature = MagicMock()
        module.planner.signature.__doc__ = "Test prompt"

        paths = exporter.export_all_formats(
            optimized_module=module,
            agent_name="build",
            model_name="gpt-4",
            baseline_score=0.5,
            optimized_score=0.75
        )

        assert "agent_config" in paths
        assert "custom_instructions" in paths
        assert "prompt_template" in paths
        assert paths["agent_config"].exists()
        assert paths["custom_instructions"].exists()
        assert paths["prompt_template"].exists()

    def test_create_usage_guide(self, exporter):
        """Test creating usage guide."""
        export_paths = {
            "agent_config": Path("/tmp/agent.jsonc"),
            "custom_instructions": Path("/tmp/instructions.md"),
            "prompt_template": Path("/tmp/template.txt")
        }

        path = exporter.create_usage_guide("build", "gpt-4", export_paths)

        assert path.exists()
        content = path.read_text()
        assert "Using Optimized Build Agent" in content
        assert "Quick Start" in content
        assert "Option 1" in content
        assert "Option 2" in content
        assert "Option 3" in content
        assert "Troubleshooting" in content