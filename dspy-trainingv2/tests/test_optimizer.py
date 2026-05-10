"""
Test Optimizer Helpers - Unit tests for optimizer.py helper functions

Tests the pure helper functions without requiring DSPy LM instances.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.optimization.optimizer import (
    extract_score_value,
    configure_dspy_lm,
    _bypass_dspy_cache,
    _restore_temperature,
)


class TestExtractScoreValue:
    """Tests for extract_score_value function."""

    def test_float_input(self):
        """Test with float input."""
        assert extract_score_value(0.75) == 0.75

    def test_int_input(self):
        """Test with int input."""
        assert extract_score_value(42) == 42.0

    def test_object_with_score_attribute(self):
        """Test with object having score attribute."""
        obj = MagicMock()
        obj.score = 0.85
        assert extract_score_value(obj) == 0.85

    def test_dict_with_score_key(self):
        """Test with dict containing score key."""
        assert extract_score_value({"score": 0.9}) == 0.9

    def test_nested_dict_with_score(self):
        """Test with nested dict containing score."""
        assert extract_score_value({"score": {"score": 0.8}}) == 0.8

    def test_unconvertible_value(self):
        """Test with unconvertible value."""
        with patch('src.optimization.optimizer.logger') as mock_logger:
            result = extract_score_value("not a number")
            assert result == 0.0
            mock_logger.warning.assert_called_once()

    def test_none_value(self):
        """Test with None value."""
        with patch('src.optimization.optimizer.logger') as mock_logger:
            result = extract_score_value(None)
            assert result == 0.0
            mock_logger.warning.assert_called_once()


class TestBypassDspyCache:
    """Tests for _bypass_dspy_cache function."""

    def test_bypass_zero_temperature(self):
        """Test bypassing when temperature is 0."""
        lm = MagicMock()
        lm.kwargs = {"temperature": 0.0}

        original = _bypass_dspy_cache(lm, "Test message")

        assert original == 0.0
        assert lm.kwargs["temperature"] == 0.001

    def test_no_bypass_nonzero_temperature(self):
        """Test no bypass when temperature is non-zero."""
        lm = MagicMock()
        lm.kwargs = {"temperature": 0.5}

        original = _bypass_dspy_cache(lm)

        assert original == 0.5
        assert lm.kwargs["temperature"] == 0.5  # Unchanged

    def test_missing_temperature(self):
        """Test with missing temperature key."""
        lm = MagicMock()
        lm.kwargs = {}

        original = _bypass_dspy_cache(lm)

        assert original == 0.0  # Default
        assert lm.kwargs["temperature"] == 0.001


class TestRestoreTemperature:
    """Tests for _restore_temperature function."""

    def test_restore_zero(self):
        """Test restoring to zero."""
        lm = MagicMock()
        lm.kwargs = {"temperature": 0.001}

        _restore_temperature(lm, 0.0)

        assert lm.kwargs["temperature"] == 0.0

    def test_restore_nonzero(self):
        """Test that non-zero original temp is not modified."""
        lm = MagicMock()
        lm.kwargs = {"temperature": 0.001}

        _restore_temperature(lm, 0.7)

        assert lm.kwargs["temperature"] == 0.001


class TestConfigureDspyLm:
    """Tests for configure_dspy_lm function."""

    def test_ollama_openai_compatible(self):
        """Test Ollama with OpenAI-compatible endpoint."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            result = configure_dspy_lm(
                model="qwen2.5-coder:32b",
                provider="ollama",
                api_base="http://localhost:11434/v1"
            )

            mock_dspy.LM.assert_called_once()
            call_kwargs = mock_dspy.LM.call_args[1]
            assert call_kwargs["model"] == "openai/qwen2.5-coder:32b"
            assert call_kwargs["api_base"] == "http://localhost:11434/v1"
            assert call_kwargs["api_key"] == "ollama-no-key-required"

    def test_ollama_native(self):
        """Test Ollama with native API."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            result = configure_dspy_lm(
                model="qwen2.5-coder:32b",
                provider="ollama",
                api_base="http://localhost:11434"
            )

            call_kwargs = mock_dspy.LM.call_args[1]
            assert call_kwargs["model"] == "ollama_chat/qwen2.5-coder:32b"

    def test_openai_standard(self):
        """Test standard OpenAI configuration."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            result = configure_dspy_lm(
                model="gpt-4o",
                provider="openai",
                api_key="test-key"
            )

            call_kwargs = mock_dspy.LM.call_args[1]
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["api_key"] == "test-key"

    def test_openai_custom_endpoint(self):
        """Test OpenAI with custom endpoint."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            result = configure_dspy_lm(
                model="custom-model",
                provider="openai",
                api_base="https://api.custom.com/v1"
            )

            call_kwargs = mock_dspy.LM.call_args[1]
            assert call_kwargs["model"] == "openai/custom-model"
            assert call_kwargs["api_base"] == "https://api.custom.com/v1"

    def test_anthropic(self):
        """Test Anthropic configuration."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            result = configure_dspy_lm(
                model="claude-sonnet-4-5",
                provider="anthropic"
            )

            call_kwargs = mock_dspy.LM.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-5"

    def test_openai_compatible(self):
        """Test OpenAI-compatible provider."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            with patch('src.optimization.optimizer.logger') as mock_logger:
                result = configure_dspy_lm(
                    model="custom-model",
                    provider="openai-compatible"
                )

                call_kwargs = mock_dspy.LM.call_args[1]
                assert call_kwargs["model"] == "custom-model"
                mock_logger.warning.assert_called_once()

    def test_unknown_provider(self):
        """Test unknown provider warning."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_lm = MagicMock()
            mock_dspy.LM.return_value = mock_lm

            with patch('src.optimization.optimizer.logger') as mock_logger:
                result = configure_dspy_lm(
                    model="test-model",
                    provider="unknown"
                )

                mock_logger.warning.assert_called_once()

    def test_lm_creation_failure(self):
        """Test handling of LM creation failure."""
        with patch('src.optimization.optimizer.dspy') as mock_dspy:
            mock_dspy.LM.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                configure_dspy_lm(
                    model="test-model",
                    provider="openai"
                )

    def test_dspy_not_installed(self):
        """Test when DSPy is not installed."""
        with patch('src.optimization.optimizer.dspy', None):
            with pytest.raises(ImportError):
                configure_dspy_lm(model="test", provider="openai")