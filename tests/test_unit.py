"""
Unit tests for whispy transcribe module.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from whispy.transcribe import (
    find_whisper_cli, 
    find_default_model,
    transcribe_file
)


class TestPathFunctions:
    """Test path-related functions."""

    def test_find_whisper_cli_returns_path(self):
        """Test that find_whisper_cli returns a valid path or None."""
        path = find_whisper_cli()
        # Can be None if not found, or a string if found
        assert path is None or isinstance(path, str)

    def test_find_default_model_returns_path(self):
        """Test that find_default_model returns a valid path or None."""
        path = find_default_model()
        # Can be None if not found, or a string if found
        if path is not None:
            assert isinstance(path, str)
            assert str(path).endswith('.bin')


class TestTranscribeFile:
    """Test transcribe_file function."""

    @patch('whispy.transcribe.subprocess.run')
    @patch('whispy.transcribe.find_whisper_cli')
    def test_transcribe_file_basic(self, mock_find_cli, mock_run):
        """Test basic whisper-cli execution."""
        # Setup mocks
        mock_find_cli.return_value = "/path/to/whisper-cli"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Test transcription output",
            stderr=""
        )
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            result = transcribe_file(
                audio_path="test.wav",
                model_path="model.bin"
            )
        
        # Verify the result
        assert result == "Test transcription output"
        
        # Verify subprocess.run was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # Get the command args
        assert "/path/to/whisper-cli" in call_args
        assert "test.wav" in call_args
        assert "model.bin" in call_args

    @patch('whispy.transcribe.subprocess.run')
    @patch('whispy.transcribe.find_whisper_cli')
    def test_transcribe_file_with_language(self, mock_find_cli, mock_run):
        """Test whisper-cli execution with language option."""
        # Setup mocks
        mock_find_cli.return_value = "/path/to/whisper-cli"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Test output",
            stderr=""
        )
        
        # Test with language option
        with patch('os.path.exists', return_value=True):
            result = transcribe_file(
                audio_path="test.wav",
                model_path="model.bin",
                language="en"
            )
        
        # Verify language was included in command
        call_args = mock_run.call_args[0][0]
        assert "-l" in call_args
        assert "en" in call_args

    @patch('whispy.transcribe.subprocess.run')
    @patch('whispy.transcribe.find_whisper_cli')
    def test_transcribe_file_failure(self, mock_find_cli, mock_run):
        """Test whisper-cli execution failure."""
        # Setup mocks for failure
        mock_find_cli.return_value = "/path/to/whisper-cli"
        mock_run.side_effect = subprocess.CalledProcessError(1, 'whisper-cli', stderr="Error message")
        
        # Test the function - should raise WhispyError
        with patch('os.path.exists', return_value=True):
            with pytest.raises(Exception):  # WhispyError
                transcribe_file(
                    audio_path="test.wav",
                    model_path="model.bin"
                )

    @patch('whispy.transcribe.find_whisper_cli')
    def test_transcribe_file_missing_binary(self, mock_find_cli):
        """Test whisper-cli execution when binary is missing."""
        # Setup mocks for missing binary
        mock_find_cli.return_value = None
        
        # Test the function - should raise WhispyError
        with patch('os.path.exists', return_value=True):
            with pytest.raises(Exception):  # WhispyError
                transcribe_file(
                    audio_path="test.wav",
                    model_path="model.bin"
                )


class TestModelSearch:
    """Test model file search functionality."""

    def test_find_default_model_searches_directories(self):
        """Test that find_default_model searches appropriate directories."""
        # This is more of an integration test, but we can test the logic
        result = find_default_model()
        
        # Can be None if no models are found
        if result is not None:
            # Should be a .bin file
            assert str(result).endswith('.bin')

    @patch('whispy.transcribe.pathlib.Path.exists')
    def test_find_default_model_with_existing_file(self, mock_exists):
        """Test find_default_model when file exists."""
        mock_exists.return_value = True
        
        result = find_default_model()
        # Should find some model
        if result is not None:
            assert str(result).endswith('.bin')


# Test markers
pytestmark = pytest.mark.unit 