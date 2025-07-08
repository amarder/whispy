"""
Tests for whispy CLI functionality.
"""

import os
import tempfile
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch
from whispy.cli import app
from whispy.transcribe import find_whisper_cli, find_default_model
from whispy import WhispyError


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_audio_file():
    """Path to the sample audio file for testing."""
    # Use the JFK sample from whisper.cpp if available
    sample_path = Path("whisper.cpp/samples/jfk.wav")
    if sample_path.exists():
        return str(sample_path)
    
    # Fall back to a local copy if it exists
    sample_path = Path("whisper.cpp/samples/jfk.mp3")
    if sample_path.exists():
        return str(sample_path)
    
    # Skip tests if no sample file is available
    pytest.skip("No sample audio file available for testing")


class TestCLICommands:
    """Test basic CLI commands."""

    def test_help_command(self, runner):
        """Test the help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "whispy" in result.stdout.lower()
        assert "transcribe" in result.stdout.lower()
        assert "version" in result.stdout.lower()
        assert "info" in result.stdout.lower()

    def test_version_command(self, runner):
        """Test the version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "whispy version" in result.stdout.lower()
        assert "0.1.0" in result.stdout

    def test_info_command(self, runner):
        """Test the info command."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "whispy version" in result.stdout.lower()
        assert "whisper-cli" in result.stdout.lower()

    def test_transcribe_help(self, runner):
        """Test the transcribe command help."""
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.stdout.lower()
        assert "audio_file" in result.stdout.lower()
        assert "model" in result.stdout.lower()


class TestTranscribeCommand:
    """Test the transcribe command functionality."""

    def test_transcribe_with_sample_file(self, runner, sample_audio_file):
        """Test transcription with the sample audio file."""
        result = runner.invoke(app, ["transcribe", sample_audio_file])
        
        # Check that command completed successfully
        assert result.exit_code == 0
        
        # Check that output contains expected text (JFK quote)
        output_lower = result.stdout.lower()
        assert any(word in output_lower for word in [
            "fellow", "americans", "ask", "country", "what", "you", "can", "do"
        ])

    def test_transcribe_nonexistent_file(self, runner):
        """Test transcription with a nonexistent file."""
        result = runner.invoke(app, ["transcribe", "nonexistent_file.wav"])
        assert result.exit_code != 0

    def test_transcribe_with_verbose_flag(self, runner, sample_audio_file):
        """Test transcription with verbose output."""
        result = runner.invoke(app, ["transcribe", sample_audio_file, "--verbose"])
        
        # Should still complete successfully
        assert result.exit_code == 0

    def test_transcribe_with_output_file(self, runner, sample_audio_file):
        """Test transcription with output file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_file = f.name
        
        try:
            result = runner.invoke(app, ["transcribe", sample_audio_file, "--output", output_file])
            
            # Check that command completed successfully
            assert result.exit_code == 0
            
            # Check that output file was created and contains text
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                content = f.read().lower()
                assert len(content.strip()) > 0
                assert any(word in content for word in [
                    "fellow", "americans", "ask", "country"
                ])
        finally:
            # Clean up
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_transcribe_with_language_option(self, runner, sample_audio_file):
        """Test transcription with language option."""
        result = runner.invoke(app, ["transcribe", sample_audio_file, "--language", "en"])
        
        # Should complete successfully
        assert result.exit_code == 0


class TestSystemRequirements:
    """Test system requirements and setup."""

    def test_whisper_cli_path_exists(self):
        """Test that whisper-cli binary path can be found."""
        whisper_cli_path = find_whisper_cli()
        
        # Can be None if not found, or a string if found
        assert whisper_cli_path is None or isinstance(whisper_cli_path, str)

    def test_default_model_path(self):
        """Test that default model path is reasonable."""
        model_path = find_default_model()
        
        # Can be None if not found
        if model_path is not None:
            assert isinstance(model_path, str)
            # Should be a .bin file
            assert str(model_path).endswith('.bin')

    def test_build_command(self, runner):
        """Test the build command."""
        result = runner.invoke(app, ["build"])
        
        # Command should complete (success or failure depends on system setup)
        # We mainly check that it doesn't crash
        assert result.exit_code in [0, 1]  # Allow both success and failure


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_transcribe_empty_arguments(self, runner):
        """Test transcribe command with no arguments."""
        result = runner.invoke(app, ["transcribe"])
        assert result.exit_code != 0

    def test_transcribe_invalid_file_extension(self, runner):
        """Test transcribe with invalid file extension."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not an audio file")
            temp_file = f.name
        
        try:
            result = runner.invoke(app, ["transcribe", temp_file])
            # Should fail gracefully
            assert result.exit_code != 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_transcribe_with_invalid_model_path(self, runner, sample_audio_file):
        """Test transcribe with invalid model path."""
        result = runner.invoke(app, ["transcribe", sample_audio_file, "--model", "nonexistent_model.bin"])
        
        # Should fail gracefully
        assert result.exit_code != 0


class TestRecordAndTranscribeCommand:
    """Test the record-and-transcribe command."""

    def test_record_and_transcribe_help(self, runner):
        """Test the record-and-transcribe command help."""
        result = runner.invoke(app, ["record-and-transcribe", "--help"])
        assert result.exit_code == 0
        assert "record audio from microphone" in result.stdout.lower()
        assert "--test-mic" in result.stdout
        assert "--save-audio" in result.stdout

    @patch('whispy.cli.test_microphone')
    def test_record_and_transcribe_test_mic_success(self, mock_test_mic, runner):
        """Test record-and-transcribe with successful mic test."""
        mock_test_mic.return_value = True
        
        result = runner.invoke(app, ["record-and-transcribe", "--test-mic"])
        
        assert result.exit_code == 0
        assert "microphone test passed" in result.stdout.lower()
        mock_test_mic.assert_called_once()

    @patch('whispy.cli.test_microphone')
    def test_record_and_transcribe_test_mic_failure(self, mock_test_mic, runner):
        """Test record-and-transcribe with failed mic test."""
        mock_test_mic.return_value = False
        
        result = runner.invoke(app, ["record-and-transcribe", "--test-mic"])
        
        assert result.exit_code == 1
        assert "microphone test failed" in result.stdout.lower()
        mock_test_mic.assert_called_once()

    @patch('whispy.cli.check_audio_devices')
    @patch('whispy.cli.record_audio_until_interrupt')
    @patch('whispy.cli.transcribe_audio')
    @patch('whispy.cli.pathlib.Path.unlink')
    def test_record_and_transcribe_basic(
        self, 
        mock_unlink,
        mock_transcribe_audio,
        mock_record_audio,
        mock_check_devices,
        runner
    ):
        """Test basic record-and-transcribe functionality."""
        # Setup mocks
        mock_check_devices.return_value = {
            'default_input_info': {'name': 'Test Microphone'}
        }
        mock_record_audio.return_value = "/tmp/test_recording.wav"
        
        result = runner.invoke(app, ["record-and-transcribe", "--verbose"])
        
        assert result.exit_code == 0
        mock_check_devices.assert_called_once()
        mock_record_audio.assert_called_once()
        mock_transcribe_audio.assert_called_once()
        mock_unlink.assert_called_once()  # Cleanup temporary file

    @patch('whispy.cli.check_audio_devices')
    @patch('whispy.cli.record_audio_until_interrupt')
    @patch('whispy.cli.transcribe_audio')
    def test_record_and_transcribe_save_audio(
        self,
        mock_transcribe_audio,
        mock_record_audio,
        mock_check_devices,
        runner
    ):
        """Test record-and-transcribe with save audio option."""
        mock_check_devices.return_value = {
            'default_input_info': {'name': 'Test Microphone'}
        }
        mock_record_audio.return_value = "/tmp/test_recording.wav"
        
        result = runner.invoke(app, [
            "record-and-transcribe", 
            "--save-audio", "my_recording.wav"
        ])
        
        assert result.exit_code == 0
        # Should call record_audio with the specified output path
        mock_record_audio.assert_called_once_with(output_path="my_recording.wav")

    @patch('whispy.cli.check_audio_devices')
    @patch('whispy.cli.record_audio_until_interrupt')
    def test_record_and_transcribe_keyboard_interrupt(
        self,
        mock_record_audio,
        mock_check_devices,
        runner
    ):
        """Test record-and-transcribe with keyboard interrupt."""
        mock_check_devices.return_value = {
            'default_input_info': {'name': 'Test Microphone'}
        }
        mock_record_audio.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(app, ["record-and-transcribe"])
        
        assert result.exit_code == 130  # Standard Ctrl+C exit code
        assert "cancelled by user" in result.stdout.lower()

    @patch('whispy.cli.check_audio_devices')
    def test_record_and_transcribe_device_check_warning(
        self,
        mock_check_devices,
        runner
    ):
        """Test record-and-transcribe with device check warning."""
        mock_check_devices.side_effect = Exception("Device error")
        
        # Should continue with a warning
        with patch('whispy.cli.record_audio_until_interrupt') as mock_record:
            with patch('whispy.cli.transcribe_audio') as mock_transcribe:
                mock_record.return_value = "/tmp/test.wav"
                
                result = runner.invoke(app, ["record-and-transcribe"])
                
                assert result.exit_code == 0
                assert "warning" in result.stdout.lower()


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, runner, sample_audio_file):
        """Test a complete workflow from installation check to transcription."""
        # First check system info
        info_result = runner.invoke(app, ["info"])
        assert info_result.exit_code == 0
        
        # Check version
        version_result = runner.invoke(app, ["version"])
        assert version_result.exit_code == 0
        
        # Perform transcription
        transcribe_result = runner.invoke(app, ["transcribe", sample_audio_file])
        assert transcribe_result.exit_code == 0
        
        # Check that transcription produced reasonable output
        output = transcribe_result.stdout.strip()
        assert len(output) > 0
        # JFK quote should contain these words
        output_lower = output.lower()
        assert "ask" in output_lower
        assert "country" in output_lower


@pytest.mark.cli
class TestRealtimeCLI:
    """Test the realtime CLI command"""
    
    def test_realtime_help(self, runner):
        """Test realtime command help"""
        result = runner.invoke(app, ["realtime", "--help"])
        assert result.exit_code == 0
        assert "Real-time transcription from microphone" in result.stdout
        assert "--chunk-duration" in result.stdout
        assert "--overlap-duration" in result.stdout
        assert "--silence-threshold" in result.stdout
        assert "--show-chunks" in result.stdout
        assert "--test-setup" in result.stdout
    
    @patch('whispy.cli.test_realtime_setup')
    def test_realtime_test_setup_success(self, mock_test_setup, runner):
        """Test realtime --test-setup success"""
        mock_test_setup.return_value = True
        
        result = runner.invoke(app, ["realtime", "--test-setup"])
        assert result.exit_code == 0
        assert "Real-time transcription setup is ready!" in result.stdout
        mock_test_setup.assert_called_once()
    
    @patch('whispy.cli.test_realtime_setup')
    def test_realtime_test_setup_failure(self, mock_test_setup, runner):
        """Test realtime --test-setup failure"""
        mock_test_setup.return_value = False
        
        result = runner.invoke(app, ["realtime", "--test-setup"])
        assert result.exit_code == 1
        assert "Real-time transcription setup failed" in result.stdout
        mock_test_setup.assert_called_once()
    
    def test_realtime_invalid_chunk_duration(self, runner):
        """Test realtime with invalid chunk duration"""
        result = runner.invoke(app, ["realtime", "--chunk-duration", "0"])
        assert result.exit_code == 1
        assert "Chunk duration must be positive" in result.stdout
    
    def test_realtime_invalid_overlap_duration(self, runner):
        """Test realtime with invalid overlap duration"""
        result = runner.invoke(app, ["realtime", "--chunk-duration", "3.0", "--overlap-duration", "3.5"])
        assert result.exit_code == 1
        assert "Overlap duration must be between 0 and chunk duration" in result.stdout
    
    def test_realtime_invalid_silence_threshold(self, runner):
        """Test realtime with invalid silence threshold"""
        result = runner.invoke(app, ["realtime", "--silence-threshold", "1.5"])
        assert result.exit_code == 1
        assert "Silence threshold must be between 0 and 1" in result.stdout
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_success(self, mock_run_realtime, runner):
        """Test successful realtime transcription"""
        mock_run_realtime.return_value = "Hello world from realtime"
        
        result = runner.invoke(app, ["realtime"])
        assert result.exit_code == 0
        assert "Real-time transcription completed" in result.stdout
        mock_run_realtime.assert_called_once()
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_with_options(self, mock_run_realtime, runner):
        """Test realtime with custom options"""
        mock_run_realtime.return_value = "Custom realtime transcript"
        
        result = runner.invoke(app, [
            "realtime",
            "--model", "/path/to/model.bin",
            "--language", "en",
            "--chunk-duration", "2.5",
            "--overlap-duration", "0.5",
            "--silence-threshold", "0.02",
            "--output", "realtime_output.txt",
            "--show-chunks",
            "--verbose"
        ])
        assert result.exit_code == 0
        assert "Real-time transcription completed" in result.stdout
        
        # Check that the function was called with correct parameters
        mock_run_realtime.assert_called_once_with(
            model_path="/path/to/model.bin",
            language="en",
            chunk_duration=2.5,
            overlap_duration=0.5,
            silence_threshold=0.02,
            output_file="realtime_output.txt",
            verbose=True,
            show_chunks=True
        )
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_no_transcript(self, mock_run_realtime, runner):
        """Test realtime when no audio is transcribed"""
        mock_run_realtime.return_value = ""
        
        result = runner.invoke(app, ["realtime"])
        assert result.exit_code == 0
        assert "No audio was transcribed" in result.stdout
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_whispy_error(self, mock_run_realtime, runner):
        """Test realtime with WhispyError"""
        mock_run_realtime.side_effect = WhispyError("Realtime setup failed")
        
        result = runner.invoke(app, ["realtime"])
        assert result.exit_code == 1
        assert "Realtime setup failed" in result.stdout
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_keyboard_interrupt(self, mock_run_realtime, runner):
        """Test realtime with keyboard interrupt"""
        mock_run_realtime.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(app, ["realtime"])
        assert result.exit_code == 0
        assert "Real-time transcription interrupted" in result.stdout
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_unexpected_error(self, mock_run_realtime, runner):
        """Test realtime with unexpected error"""
        mock_run_realtime.side_effect = Exception("Unexpected error")
        
        result = runner.invoke(app, ["realtime"])
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout
    
    @patch('whispy.cli.run_realtime_transcription')
    def test_realtime_unexpected_error_verbose(self, mock_run_realtime, runner):
        """Test realtime with unexpected error in verbose mode"""
        mock_run_realtime.side_effect = Exception("Unexpected error")
        
        result = runner.invoke(app, ["realtime", "--verbose"])
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout
        # In verbose mode, traceback should be included
        assert "Traceback" in result.stdout or "Exception" in result.stdout


# Marks for different test categories
pytestmark = pytest.mark.cli 