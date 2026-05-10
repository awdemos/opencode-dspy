"""
Test Example Builder - Unit tests for example_builder.py

Tests the pure logic functions without requiring DSPy.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.session_parser import SessionExample, ContextInfo, Outcome, AgentConfig
from src.data.example_builder import ExampleBuilder, split_examples, save_examples, load_examples


class TestExampleBuilder:
    """Tests for ExampleBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create an example builder with mocked DSPy."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_dspy.Example = MagicMock()
            mock_dspy.Example.return_value.with_inputs.return_value = MagicMock()
            yield ExampleBuilder()

    @pytest.fixture
    def sample_example(self):
        """Create a sample SessionExample."""
        return SessionExample(
            session_id="test-123",
            task="Fix bug in config.py",
            context=ContextInfo(
                working_directory="/home/user/project",
                relevant_files=["config.py", "main.py", "utils.py"],
                lsp_diagnostics={"errors": [], "warnings": []},
                git_status={"branch": "main", "uncommittedChanges": 2},
                file_count=3
            ),
            conversation_history=[],
            actions=[],
            final_response="Fixed the bug",
            outcome=Outcome(
                success=True,
                task_completed=True,
                correctness=0.9,
                efficiency=0.8,
                minimal_edits=0.7,
                time_to_completion=5.0,
                tool_call_count=3
            ),
            agent_config=AgentConfig(
                name="build",
                model="gpt-4",
                temperature=0.0
            )
        )

    def test_format_context(self, builder, sample_example):
        """Test context formatting."""
        context = builder.format_context(sample_example)

        assert "Working Directory: /home/user/project" in context
        assert "Total Files: 3" in context
        assert "Git Branch: main" in context
        assert "Uncommitted Changes: 2" in context
        assert "config.py" in context
        assert "main.py" in context

    def test_format_context_no_git(self, builder):
        """Test context formatting without git status."""
        example = SessionExample(
            session_id="test",
            task="task",
            context=ContextInfo(
                working_directory="/test",
                relevant_files=[],
                lsp_diagnostics={},
                git_status=None,
                file_count=0
            ),
            conversation_history=[],
            actions=[],
            final_response="",
            outcome=Outcome(True, True, 0.0, 0.0, 0.0, 0.0, 0),
            agent_config=AgentConfig("", "", 0.0)
        )

        context = builder.format_context(example)

        assert "Working Directory: /test" in context
        assert "Git Branch" not in context

    def test_format_conversation_history_empty(self, builder, sample_example):
        """Test formatting empty conversation history."""
        result = builder.format_conversation_history(sample_example)
        assert result == "No prior conversation"

    def test_format_conversation_history_with_messages(self, builder):
        """Test formatting conversation history with messages."""
        from src.data.session_parser import Message
        example = SessionExample(
            session_id="test",
            task="task",
            context=ContextInfo("", [], {}, None, 0),
            conversation_history=[
                Message("user", "Hello", "2024-01-01T00:00:00Z"),
                Message("assistant", "Hi there", "2024-01-01T00:00:01Z")
            ],
            actions=[],
            final_response="",
            outcome=Outcome(True, True, 0.0, 0.0, 0.0, 0.0, 0),
            agent_config=AgentConfig("", "", 0.0)
        )

        result = builder.format_conversation_history(example)

        assert "[user]: Hello" in result
        assert "[assistant]: Hi there" in result

    def test_extract_tool_sequence(self, builder):
        """Test extracting tool sequence."""
        from src.data.session_parser import ToolAction
        example = SessionExample(
            session_id="test",
            task="task",
            context=ContextInfo("", [], {}, None, 0),
            conversation_history=[],
            actions=[
                ToolAction(1, "read", "c1", {}, "", None, None),
                ToolAction(2, "edit", "c2", {}, "", None, None),
                ToolAction(3, "bash", "c3", {}, "", None, None)
            ],
            final_response="",
            outcome=Outcome(True, True, 0.0, 0.0, 0.0, 0.0, 0),
            agent_config=AgentConfig("", "", 0.0)
        )

        tools = builder.extract_tool_sequence(example)

        assert tools == ["read", "edit", "bash"]

    def test_extract_first_action_no_actions(self, builder, sample_example):
        """Test extracting first action when none exist."""
        result = builder.extract_first_action(sample_example)
        assert result is None

    def test_extract_first_action(self, builder):
        """Test extracting first action."""
        from src.data.session_parser import ToolAction
        example = SessionExample(
            session_id="test",
            task="task",
            context=ContextInfo("", [], {}, None, 0),
            conversation_history=[],
            actions=[
                ToolAction(1, "read", "c1", {"filePath": "test.py"}, "", None, None)
            ],
            final_response="",
            outcome=Outcome(True, True, 0.0, 0.0, 0.0, 0.0, 0),
            agent_config=AgentConfig("", "", 0.0)
        )

        result = builder.extract_first_action(example)

        assert result == {"tool": "read", "args": {"filePath": "test.py"}}

    def test_format_available_tools(self, builder):
        """Test available tools formatting."""
        tools = builder.format_available_tools()

        assert "read - Read file contents" in tools
        assert "write - Write/create a file" in tools
        assert "bash - Execute bash commands" in tools

    def test_build_dspy_example(self, builder, sample_example):
        """Test building DSPy example."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_example = MagicMock()
            mock_dspy.Example.return_value.with_inputs.return_value = mock_example

            result = builder.build_dspy_example(sample_example, include_labels=True)

            assert result == mock_example
            mock_dspy.Example.assert_called_once()
            call_kwargs = mock_dspy.Example.call_args[1]
            assert call_kwargs['task_description'] == "Fix bug in config.py"
            assert 'environment_context' in call_kwargs
            assert 'expected_tools' in call_kwargs
            assert 'expected_first_action' in call_kwargs

    def test_build_dspy_example_without_labels(self, builder, sample_example):
        """Test building DSPy example without labels."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_example = MagicMock()
            mock_dspy.Example.return_value.with_inputs.return_value = mock_example

            result = builder.build_dspy_example(sample_example, include_labels=False)

            call_kwargs = mock_dspy.Example.call_args[1]
            assert 'expected_tools' not in call_kwargs
            assert 'expected_first_action' not in call_kwargs

    def test_build_batch(self, builder, sample_example):
        """Test building batch of examples."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_dspy.Example.return_value.with_inputs.return_value = MagicMock()

            examples = [sample_example, sample_example]
            result = builder.build_batch(examples)

            assert len(result) == 2

    def test_build_batch_with_errors(self, builder, sample_example):
        """Test batch building with some failures."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_dspy.Example.return_value.with_inputs.return_value = MagicMock()

            # First call succeeds, second fails
            def side_effect(*args, **kwargs):
                if mock_dspy.Example.call_count == 2:
                    raise ValueError("Test error")
                return MagicMock()

            mock_dspy.Example.side_effect = side_effect

            with patch('src.data.example_builder.logger') as mock_logger:
                result = builder.build_batch([sample_example, sample_example])
                assert len(result) == 1
                mock_logger.error.assert_called_once()


class TestSplitExamples:
    """Tests for split_examples function."""

    def test_split_examples_basic(self):
        """Test basic split."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            examples = [MagicMock() for _ in range(10)]

            train, val, test = split_examples(examples, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42)

            assert len(train) == 7
            assert len(val) == 1  # 10 * 0.15 = 1.5, truncated to 1
            assert len(test) == 2  # Remainder

    def test_split_examples_invalid_splits(self):
        """Test with invalid split proportions."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            with pytest.raises(ValueError, match="Splits must sum to 1.0"):
                split_examples([MagicMock()], train_split=0.5, val_split=0.3, test_split=0.3)

    def test_split_examples_stratify(self):
        """Test stratified split."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            # Create examples with different agent names
            examples = []
            for i in range(10):
                ex = MagicMock()
                ex.agent_name = "build" if i < 7 else "plan"
                examples.append(ex)

            train, val, test = split_examples(
                examples,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15,
                random_seed=42,
                stratify_by="agent_name"
            )

            assert len(train) > 0
            assert len(val) > 0
            assert len(test) > 0


class TestSaveAndLoadExamples:
    """Tests for save_examples and load_examples."""

    def test_save_examples(self, tmp_path):
        """Test saving examples to file."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            examples = [MagicMock(), MagicMock()]
            for ex in examples:
                ex.toDict.return_value = {"task": "test"}

            file_path = tmp_path / "examples.json"
            save_examples(examples, str(file_path))

            assert file_path.exists()
            data = json.loads(file_path.read_text())
            assert len(data) == 2

    def test_load_examples(self, tmp_path):
        """Test loading examples from file."""
        with patch('src.data.example_builder.dspy') as mock_dspy:
            mock_dspy.Example.return_value = MagicMock()

            # Create test file
            file_path = tmp_path / "examples.json"
            file_path.write_text(json.dumps([{"task": "test"}]))

            result = load_examples(str(file_path))

            assert len(result) == 1
            mock_dspy.Example.assert_called_once_with(task="test")