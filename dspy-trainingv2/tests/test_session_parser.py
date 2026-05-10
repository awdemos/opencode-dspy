"""
Test Session Parser - Unit tests for session_parser.py

Tests the parsing logic without requiring DSPy or actual session files.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data.session_parser import (
    ContextInfo,
    ToolAction,
    Message,
    Evaluation,
    Outcome,
    AgentConfig,
    SessionExample,
    SessionParser,
    load_and_parse_sessions,
)


class TestSessionParser:
    """Tests for SessionParser class."""

    @pytest.fixture
    def parser(self):
        """Create a default parser."""
        return SessionParser()

    @pytest.fixture
    def sample_session_data(self):
        """Create sample session data for testing."""
        return {
            "session": "test-session-123",
            "examples": [
                {
                    "input": {
                        "task": "Fix the bug in config.py",
                        "context": {
                            "workingDirectory": "/home/user/project",
                            "relevantFiles": ["config.py", "main.py"],
                            "lspDiagnostics": {
                                "errors": [{"file": "config.py", "line": 10}],
                                "warnings": []
                            },
                            "gitStatus": {
                                "branch": "main",
                                "uncommittedChanges": 2
                            },
                            "fileCount": 5
                        },
                        "conversationHistory": [
                            {
                                "role": "user",
                                "content": "Fix the bug",
                                "timestamp": "2024-01-01T00:00:00Z"
                            }
                        ]
                    },
                    "output": {
                        "response": "Fixed the bug by updating config.py"
                    },
                    "actions": [
                        {
                            "step": 1,
                            "tool": "read",
                            "callID": "call-1",
                            "args": {"filePath": "config.py"},
                            "timestamp": "2024-01-01T00:00:01Z",
                            "result": "file contents...",
                            "success": True
                        }
                    ],
                    "outcome": {
                        "success": True,
                        "taskCompleted": True,
                        "metrics": {
                            "timeToCompletion": 5.2,
                            "toolCallCount": 3,
                            "lspErrorsCleared": True,
                            "filesModified": 1
                        },
                        "evaluation": {
                            "correctness": 0.95,
                            "efficiency": 0.8,
                            "minimalEdits": 0.9
                        }
                    },
                    "agent": {
                        "name": "build",
                        "model": "gpt-4",
                        "temperature": 0.0,
                        "promptTokens": 100,
                        "completionTokens": 50
                    },
                    "metadata": {"source": "test"}
                }
            ]
        }

    def test_parse_context(self, parser):
        """Test parsing context data."""
        context_data = {
            "workingDirectory": "/home/user/project",
            "relevantFiles": ["config.py", "main.py"],
            "lspDiagnostics": {"errors": [], "warnings": []},
            "gitStatus": {"branch": "main", "uncommittedChanges": 0},
            "fileCount": 5
        }

        context = parser.parse_context(context_data)

        assert isinstance(context, ContextInfo)
        assert context.working_directory == "/home/user/project"
        assert context.relevant_files == ["config.py", "main.py"]
        assert context.file_count == 5

    def test_parse_context_missing_fields(self, parser):
        """Test parsing context with missing fields."""
        context_data = {}

        context = parser.parse_context(context_data)

        assert context.working_directory == ""
        assert context.relevant_files == []
        assert context.lsp_diagnostics == {}
        assert context.git_status is None
        assert context.file_count == 0

    def test_parse_actions(self, parser):
        """Test parsing tool actions."""
        actions_data = [
            {
                "step": 1,
                "tool": "read",
                "callID": "call-1",
                "args": {"filePath": "test.py"},
                "timestamp": "2024-01-01T00:00:00Z",
                "result": "content",
                "success": True
            },
            {
                "step": 2,
                "tool": "edit",
                "callID": "call-2",
                "args": {"filePath": "test.py"},
                "timestamp": "2024-01-01T00:00:01Z"
            }
        ]

        actions = parser.parse_actions(actions_data)

        assert len(actions) == 2
        assert isinstance(actions[0], ToolAction)
        assert actions[0].tool == "read"
        assert actions[0].success is True
        assert actions[1].tool == "edit"
        assert actions[1].result is None

    def test_parse_conversation_history(self, parser):
        """Test parsing conversation history."""
        history_data = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
            {"role": "assistant", "content": "Hi", "timestamp": "2024-01-01T00:00:01Z"}
        ]

        messages = parser.parse_conversation_history(history_data)

        assert len(messages) == 2
        assert isinstance(messages[0], Message)
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_parse_outcome(self, parser):
        """Test parsing outcome data."""
        outcome_data = {
            "success": True,
            "taskCompleted": True,
            "metrics": {
                "timeToCompletion": 10.5,
                "toolCallCount": 5,
                "lspErrorsCleared": True,
                "filesModified": 2
            },
            "evaluation": {
                "correctness": 0.9,
                "efficiency": 0.7,
                "minimalEdits": 0.8
            }
        }

        outcome = parser.parse_outcome(outcome_data)

        assert isinstance(outcome, Outcome)
        assert outcome.success is True
        assert outcome.correctness == 0.9
        assert outcome.efficiency == 0.7
        assert outcome.time_to_completion == 10.5
        assert outcome.tool_call_count == 5

    def test_parse_agent_config(self, parser):
        """Test parsing agent configuration."""
        agent_data = {
            "name": "build",
            "model": "gpt-4",
            "temperature": 0.0,
            "promptTokens": 100,
            "completionTokens": 50
        }

        config = parser.parse_agent_config(agent_data)

        assert isinstance(config, AgentConfig)
        assert config.name == "build"
        assert config.model == "gpt-4"
        assert config.temperature == 0.0
        assert config.prompt_tokens == 100

    def test_parse_example(self, parser, sample_session_data):
        """Test parsing a complete example."""
        example_data = sample_session_data["examples"][0]
        session_id = sample_session_data["session"]

        example = parser.parse_example(example_data, session_id)

        assert isinstance(example, SessionExample)
        assert example.session_id == "test-session-123"
        assert example.task == "Fix the bug in config.py"
        assert len(example.actions) == 1
        assert example.actions[0].tool == "read"
        assert example.outcome.correctness == 0.95
        assert example.agent_config.name == "build"

    def test_parse_example_invalid(self, parser):
        """Test parsing invalid example data."""
        with patch('src.data.session_parser.logger') as mock_logger:
            example = parser.parse_example({}, "test")
            assert example is not None  # Empty dict creates defaults
            assert example.session_id == "test"
            assert example.task == ""

    def test_filter_by_quality(self, parser):
        """Test quality filtering."""
        examples = [
            SessionExample(
                session_id="1",
                task="task1",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(True, True, 0.9, 0.8, 0.7, 0.0, 0),
                agent_config=AgentConfig("", "", 0.0)
            ),
            SessionExample(
                session_id="2",
                task="task2",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(True, True, 0.5, 0.4, 0.3, 0.0, 0),
                agent_config=AgentConfig("", "", 0.0)
            )
        ]

        parser.min_correctness = 0.7
        filtered = parser.filter_by_quality(examples)

        assert len(filtered) == 1
        assert filtered[0].session_id == "1"

    def test_filter_successful(self, parser):
        """Test filtering successful examples."""
        examples = [
            SessionExample(
                session_id="1",
                task="task1",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(True, True, 1.0, 1.0, 1.0, 0.0, 0),
                agent_config=AgentConfig("", "", 0.0)
            ),
            SessionExample(
                session_id="2",
                task="task2",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(False, False, 0.0, 0.0, 0.0, 0.0, 0),
                agent_config=AgentConfig("", "", 0.0)
            )
        ]

        successful = parser.filter_successful(examples)

        assert len(successful) == 1
        assert successful[0].session_id == "1"

    def test_filter_by_agent(self, parser):
        """Test filtering by agent name."""
        examples = [
            SessionExample(
                session_id="1",
                task="task1",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(True, True, 1.0, 1.0, 1.0, 0.0, 0),
                agent_config=AgentConfig("build", "gpt-4", 0.0)
            ),
            SessionExample(
                session_id="2",
                task="task2",
                context=ContextInfo("", [], {}, None, 0),
                conversation_history=[],
                actions=[],
                final_response="",
                outcome=Outcome(True, True, 1.0, 1.0, 1.0, 0.0, 0),
                agent_config=AgentConfig("plan", "gpt-4", 0.0)
            )
        ]

        filtered = parser.filter_by_agent(examples, "build")

        assert len(filtered) == 1
        assert filtered[0].session_id == "1"

    def test_parse_sessions(self, parser, sample_session_data):
        """Test parsing multiple sessions."""
        sessions = [sample_session_data]

        examples = parser.parse_sessions(sessions)

        assert len(examples) == 1
        assert examples[0].session_id == "test-session-123"

    def test_load_session_file(self, parser, tmp_path):
        """Test loading a session file."""
        session_file = tmp_path / "test.json"
        session_file.write_text(json.dumps({"examples": []}))

        result = parser.load_session_file(session_file)

        assert result == {"examples": []}

    def test_load_session_file_invalid_json(self, parser, tmp_path):
        """Test loading invalid JSON."""
        session_file = tmp_path / "invalid.json"
        session_file.write_text("not json")

        with patch('src.data.session_parser.logger') as mock_logger:
            result = parser.load_session_file(session_file)
            assert result is None
            mock_logger.error.assert_called_once()

    def test_load_session_file_not_dict(self, parser, tmp_path):
        """Test loading JSON that's not a dict."""
        session_file = tmp_path / "list.json"
        session_file.write_text(json.dumps([1, 2, 3]))

        with patch('src.data.session_parser.logger') as mock_logger:
            result = parser.load_session_file(session_file)
            assert result is None
            mock_logger.warning.assert_called_once()

    def test_load_sessions_from_directory(self, parser, tmp_path):
        """Test loading sessions from directory."""
        # Create test files
        (tmp_path / "session1.json").write_text(json.dumps({"examples": []}))
        (tmp_path / "session2.json").write_text(json.dumps({"examples": []}))
        (tmp_path / "not_json.txt").write_text("text")

        sessions = parser.load_sessions_from_directory(tmp_path)

        assert len(sessions) == 2


class TestLoadAndParseSessions:
    """Tests for load_and_parse_sessions convenience function."""

    def test_load_and_parse_sessions(self, tmp_path):
        """Test the convenience function."""
        # Create a session file
        session_data = {
            "session": "test",
            "examples": [
                {
                    "input": {
                        "task": "task1",
                        "context": {
                            "workingDirectory": "/test",
                            "relevantFiles": [],
                            "lspDiagnostics": {},
                            "fileCount": 0
                        },
                        "conversationHistory": []
                    },
                    "output": {"response": "done"},
                    "actions": [],
                    "outcome": {
                        "success": True,
                        "taskCompleted": True,
                        "metrics": {"timeToCompletion": 0, "toolCallCount": 0},
                        "evaluation": {"correctness": 0.9, "efficiency": 0.8, "minimalEdits": 0.7}
                    },
                    "agent": {"name": "build", "model": "gpt-4", "temperature": 0.0},
                    "metadata": {}
                }
            ]
        }
        (tmp_path / "test.json").write_text(json.dumps(session_data))

        examples = load_and_parse_sessions(
            directory=tmp_path,
            min_correctness=0.5,
            require_success=True
        )

        assert len(examples) == 1
        assert examples[0].outcome.correctness == 0.9

    def test_load_and_parse_sessions_empty_directory(self, tmp_path):
        """Test with empty directory."""
        examples = load_and_parse_sessions(
            directory=tmp_path,
            min_correctness=0.0,
            require_success=False
        )

        assert len(examples) == 0

    def test_load_and_parse_sessions_with_filters(self, tmp_path):
        """Test with agent filter."""
        # Create sessions with different agents
        for i, agent_name in enumerate(["build", "plan", "build"]):
            session_data = {
                "session": f"test-{i}",
                "examples": [
                    {
                        "input": {
                            "task": f"task{i}",
                            "context": {
                                "workingDirectory": "/test",
                                "relevantFiles": [],
                                "lspDiagnostics": {},
                                "fileCount": 0
                            },
                            "conversationHistory": []
                        },
                        "output": {"response": "done"},
                        "actions": [],
                        "outcome": {
                            "success": True,
                            "taskCompleted": True,
                            "metrics": {"timeToCompletion": 0, "toolCallCount": 0},
                            "evaluation": {"correctness": 0.9, "efficiency": 0.8, "minimalEdits": 0.7}
                        },
                        "agent": {"name": agent_name, "model": "gpt-4", "temperature": 0.0},
                        "metadata": {}
                    }
                ]
            }
            (tmp_path / f"session{i}.json").write_text(json.dumps(session_data))

        examples = load_and_parse_sessions(
            directory=tmp_path,
            min_correctness=0.0,
            require_success=False,
            agent_filter="build"
        )

        assert len(examples) == 2
        assert all(ex.agent_config.name == "build" for ex in examples)