"""
Test Metrics - Unit tests for metrics.py

Tests the pure metric functions without requiring DSPy.
"""

from unittest.mock import MagicMock

import pytest

from src.evaluation.metrics import (
    extract_relevant_terms,
    extract_tools_from_plan,
    parse_action_json,
    tool_validity_score,
    reasoning_quality_score,
    plan_coherence_score,
    first_action_match_score,
    efficiency_score,
    composite_metric,
    correctness_metric,
    simple_metric,
    VALID_TOOLS,
)


class TestExtractRelevantTerms:
    """Tests for extract_relevant_terms function."""

    def test_extract_file_paths(self):
        """Test extracting file paths from context."""
        context = "Working Directory: /home/user/project\nRelevant Files: config.py, main.py"
        terms = extract_relevant_terms(context)

        assert "config.py" in terms
        assert "main.py" in terms

    def test_extract_quoted_terms(self):
        """Test extracting quoted terms."""
        context = 'The file "important.txt" needs fixing'
        terms = extract_relevant_terms(context)

        assert "important.txt" in terms

    def test_empty_context(self):
        """Test with empty context."""
        terms = extract_relevant_terms("")
        assert terms == []


class TestExtractToolsFromPlan:
    """Tests for extract_tools_from_plan function."""

    def test_extract_single_tool(self):
        """Test extracting a single tool mention."""
        plan = "I will use the read tool to examine the file"
        tools = extract_tools_from_plan(plan)

        assert "read" in tools

    def test_extract_multiple_tools(self):
        """Test extracting multiple tools."""
        plan = "First read the file, then edit it, and finally run bash to test"
        tools = extract_tools_from_plan(plan)

        assert "read" in tools
        assert "edit" in tools
        assert "bash" in tools

    def test_no_tools(self):
        """Test with no tool mentions."""
        plan = "This is a plan with no valid tools"
        tools = extract_tools_from_plan(plan)

        assert tools == []

    def test_case_insensitive(self):
        """Test case insensitivity."""
        plan = "Use READ and BASH"
        tools = extract_tools_from_plan(plan)

        assert "read" in tools
        assert "bash" in tools


class TestParseActionJson:
    """Tests for parse_action_json function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        action_str = '{"tool": "read", "args": {"filePath": "test.py"}}'
        result = parse_action_json(action_str)

        assert result == {"tool": "read", "args": {"filePath": "test.py"}}

    def test_parse_json_code_block(self):
        """Test parsing JSON from markdown code block."""
        action_str = '```json\n{"tool": "read", "args": {}}\n```'
        result = parse_action_json(action_str)

        assert result == {"tool": "read", "args": {}}

    def test_parse_action_tags(self):
        """Test parsing from action tags."""
        action_str = '<action>{"tool": "edit", "args": {}}</action>'
        result = parse_action_json(action_str)

        assert result == {"tool": "edit", "args": {}}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        action_str = "not valid json"
        result = parse_action_json(action_str)

        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_action_json("")
        assert result is None


class TestToolValidityScore:
    """Tests for tool_validity_score function."""

    def test_valid_tool(self):
        """Test with valid tool."""
        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {}}'

        score = tool_validity_score(prediction)
        assert score == 1.0

    def test_invalid_tool(self):
        """Test with invalid tool."""
        prediction = MagicMock()
        prediction.first_action = '{"tool": "invalid_tool", "args": {}}'

        score = tool_validity_score(prediction)
        assert score == 0.0

    def test_no_first_action(self):
        """Test with no first_action."""
        prediction = MagicMock()
        prediction.first_action = None

        score = tool_validity_score(prediction)
        assert score == 0.0

    def test_unparseable_action(self):
        """Test with unparseable action."""
        prediction = MagicMock()
        prediction.first_action = "not json"

        score = tool_validity_score(prediction)
        assert score == 0.0


class TestReasoningQualityScore:
    """Tests for reasoning_quality_score function."""

    def test_relevant_terms_mentioned(self):
        """Test when reasoning mentions relevant terms."""
        example = MagicMock()
        example.environment_context = "Files: config.py, main.py"

        prediction = MagicMock()
        prediction.reasoning = "I need to check config.py and main.py"

        score = reasoning_quality_score(example, prediction)
        assert score > 0.0

    def test_no_relevant_terms(self):
        """Test when no relevant terms to check."""
        example = MagicMock()
        example.environment_context = "Empty directory"

        prediction = MagicMock()
        prediction.reasoning = "Some reasoning"

        score = reasoning_quality_score(example, prediction)
        assert score == 0.5  # Neutral when no terms

    def test_empty_reasoning(self):
        """Test with empty reasoning."""
        example = MagicMock()
        example.environment_context = "Files: test.py"

        prediction = MagicMock()
        prediction.reasoning = ""

        score = reasoning_quality_score(example, prediction)
        assert score == 0.0


class TestPlanCoherenceScore:
    """Tests for plan_coherence_score function."""

    def test_matching_tools(self):
        """Test when plan matches expected tools."""
        example = MagicMock()
        example.expected_tools = ["read", "edit", "bash"]

        prediction = MagicMock()
        prediction.tool_plan = "I will read the file, edit it, and run bash"

        score = plan_coherence_score(example, prediction)
        assert score > 0.0

    def test_no_expected_tools(self):
        """Test with no expected tools."""
        example = MagicMock()
        del example.expected_tools

        prediction = MagicMock()
        prediction.tool_plan = "Some plan"

        score = plan_coherence_score(example, prediction)
        assert score == 0.5  # Neutral

    def test_empty_plan(self):
        """Test with empty plan."""
        example = MagicMock()
        example.expected_tools = ["read"]

        prediction = MagicMock()
        prediction.tool_plan = ""

        score = plan_coherence_score(example, prediction)
        assert score == 0.0


class TestFirstActionMatchScore:
    """Tests for first_action_match_score function."""

    def test_exact_match(self):
        """Test exact action match."""
        example = MagicMock()
        example.expected_first_action = {"tool": "read", "args": {"filePath": "test.py"}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {"filePath": "test.py"}}'

        score = first_action_match_score(example, prediction)
        assert score == 1.0

    def test_tool_match_no_critical_args(self):
        """Test tool match without critical args."""
        example = MagicMock()
        example.expected_first_action = {"tool": "read", "args": {}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {"filePath": "test.py"}}'

        score = first_action_match_score(example, prediction)
        assert score == 1.0

    def test_wrong_tool(self):
        """Test with wrong tool."""
        example = MagicMock()
        example.expected_first_action = {"tool": "read", "args": {}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "edit", "args": {}}'

        score = first_action_match_score(example, prediction)
        assert score == 0.0

    def test_no_expected_action(self):
        """Test with no expected action."""
        example = MagicMock()
        del example.expected_first_action

        prediction = MagicMock()
        prediction.first_action = '{"tool": "read"}'

        score = first_action_match_score(example, prediction)
        assert score == 0.5  # Neutral


class TestEfficiencyScore:
    """Tests for efficiency_score function."""

    def test_optimal_length(self):
        """Test optimal reasoning length."""
        prediction = MagicMock()
        prediction.reasoning = "a" * 300  # 300 chars is optimal

        score = efficiency_score(prediction)
        assert score == 1.0

    def test_too_short(self):
        """Test too short reasoning."""
        prediction = MagicMock()
        prediction.reasoning = "short"

        score = efficiency_score(prediction)
        assert score == 0.5

    def test_too_long(self):
        """Test too long reasoning."""
        prediction = MagicMock()
        prediction.reasoning = "a" * 2000

        score = efficiency_score(prediction)
        assert score < 1.0

    def test_acceptable_length(self):
        """Test acceptable reasoning length."""
        prediction = MagicMock()
        prediction.reasoning = "a" * 800

        score = efficiency_score(prediction)
        assert score == 0.8


class TestCompositeMetric:
    """Tests for composite_metric function."""

    def test_valid_prediction(self):
        """Test with valid prediction."""
        example = MagicMock()
        example.task_description = "Test task"
        example.environment_context = "Files: test.py"
        example.expected_tools = ["read"]
        example.expected_first_action = {"tool": "read", "args": {}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {}}'
        prediction.reasoning = "I will read the file test.py"
        prediction.tool_plan = "Use read tool"

        score = composite_metric(example, prediction)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_invalid_prediction(self):
        """Test with invalid prediction."""
        example = MagicMock()
        example.task_description = "Test"
        example.environment_context = ""

        prediction = MagicMock()
        prediction.first_action = '{"tool": "invalid", "args": {}}'
        prediction.reasoning = ""
        prediction.tool_plan = ""

        score = composite_metric(example, prediction)
        assert 0.0 <= score <= 1.0


class TestCorrectnessMetric:
    """Tests for correctness_metric function."""

    def test_correct(self):
        """Test correct prediction."""
        example = MagicMock()
        example.expected_first_action = {"tool": "read", "args": {"filePath": "test.py"}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {"filePath": "test.py"}}'

        score = correctness_metric(example, prediction)
        assert score == 1.0

    def test_incorrect(self):
        """Test incorrect prediction."""
        example = MagicMock()
        example.expected_first_action = {"tool": "read", "args": {}}

        prediction = MagicMock()
        prediction.first_action = '{"tool": "edit", "args": {}}'

        score = correctness_metric(example, prediction)
        assert score == 0.0


class TestSimpleMetric:
    """Tests for simple_metric function."""

    def test_valid(self):
        """Test valid tool."""
        prediction = MagicMock()
        prediction.first_action = '{"tool": "read", "args": {}}'

        result = simple_metric(None, prediction)
        assert result is True

    def test_invalid(self):
        """Test invalid tool."""
        prediction = MagicMock()
        prediction.first_action = '{"tool": "invalid", "args": {}}'

        result = simple_metric(None, prediction)
        assert result is False


class TestValidToolsSet:
    """Tests for VALID_TOOLS constant."""

    def test_contains_common_tools(self):
        """Test that common tools are in the set."""
        assert "read" in VALID_TOOLS
        assert "write" in VALID_TOOLS
        assert "edit" in VALID_TOOLS
        assert "bash" in VALID_TOOLS

    def test_no_duplicates(self):
        """Test that set has no duplicates."""
        assert len(VALID_TOOLS) == len(set(VALID_TOOLS))