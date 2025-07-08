"""
Tests for whispy recorder module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import numpy as np

from whispy.recorder import (
    AudioRecorder,
    record_audio_until_interrupt,
    check_audio_devices,
    test_microphone
)


class TestAudioRecorder:
    """Test AudioRecorder class."""

    def test_init(self):
        """Test AudioRecorder initialization."""
        recorder = AudioRecorder(sample_rate=22050, channels=2)
        
        assert recorder.sample_rate == 22050
        assert recorder.channels == 2
        assert recorder.recording is False
        assert recorder.audio_data == []
        assert recorder.stream is None

    @patch('whispy.recorder.sd.InputStream')
    def test_start_recording(self, mock_stream_class):
        """Test starting recording."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.start_recording()
        
        assert recorder.recording is True
        assert recorder.audio_data == []
        mock_stream_class.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        recorder = AudioRecorder()
        recorder.recording = True
        
        # Should not raise an error, just return
        recorder.start_recording()
        assert recorder.recording is True

    @patch('whispy.recorder.sd.InputStream')
    def test_start_recording_failure(self, mock_stream_class):
        """Test recording start failure."""
        mock_stream_class.side_effect = Exception("Audio device error")
        
        recorder = AudioRecorder()
        
        with pytest.raises(Exception):  # WhispyError
            recorder.start_recording()
        
        assert recorder.recording is False

    def test_stop_recording_not_started(self):
        """Test stopping recording when not started."""
        recorder = AudioRecorder()
        
        result = recorder.stop_recording()
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    @patch('whispy.recorder.sd.InputStream')
    def test_stop_recording_with_data(self, mock_stream_class):
        """Test stopping recording with audio data."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        recorder.recording = True
        recorder.stream = mock_stream
        
        # Add some fake audio data
        recorder.audio_data = [
            np.array([[0.1], [0.2]]),
            np.array([[0.3], [0.4]])
        ]
        
        result = recorder.stop_recording()
        
        assert recorder.recording is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert recorder.stream is None
        
        # Check that audio data was concatenated and flattened
        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # 2 samples * 2 chunks

    @patch('whispy.recorder.wavfile.write')
    def test_save_recording(self, mock_wavfile_write):
        """Test saving recording to file."""
        recorder = AudioRecorder()
        audio_data = np.array([0.1, 0.2, 0.3, 0.4])
        output_path = "test_output.wav"
        
        recorder.save_recording(audio_data, output_path)
        
        # Check that wavfile.write was called with correct parameters
        mock_wavfile_write.assert_called_once()
        call_args = mock_wavfile_write.call_args
        
        assert call_args[0][0] == output_path  # filename
        assert call_args[0][1] == recorder.sample_rate  # sample rate
        # Audio data should be converted to int16
        expected_audio = (audio_data * 32767).astype(np.int16)
        np.testing.assert_array_equal(call_args[0][2], expected_audio)

    def test_save_recording_empty_data(self):
        """Test saving empty recording."""
        recorder = AudioRecorder()
        
        with pytest.raises(Exception):  # WhispyError
            recorder.save_recording(np.array([]), "test.wav")

    @patch('whispy.recorder.wavfile.write')
    def test_save_recording_failure(self, mock_wavfile_write):
        """Test save recording failure."""
        mock_wavfile_write.side_effect = Exception("Write error")
        
        recorder = AudioRecorder()
        audio_data = np.array([0.1, 0.2])
        
        with pytest.raises(Exception):  # WhispyError
            recorder.save_recording(audio_data, "test.wav")


class TestRecordingFunctions:
    """Test recording utility functions."""

    @patch('whispy.recorder.AudioRecorder')
    @patch('whispy.recorder.signal.signal')
    @patch('whispy.recorder.tempfile.NamedTemporaryFile')
    def test_record_audio_until_interrupt(
        self, 
        mock_tempfile, 
        mock_signal, 
        mock_recorder_class
    ):
        """Test recording until interrupt."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test_recording.wav"
        mock_tempfile.return_value = mock_temp
        
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        mock_recorder.stop_recording.return_value = np.array([0.1, 0.2, 0.3])
        
        # Mock the recording process
        with patch('whispy.recorder.threading.Event') as mock_event_class:
            mock_event = MagicMock()
            mock_event_class.return_value = mock_event
            
            result = record_audio_until_interrupt()
            
            # Check that the process was set up correctly
            mock_recorder_class.assert_called_once()
            mock_recorder.start_recording.assert_called_once()
            mock_event.wait.assert_called_once()
            mock_recorder.stop_recording.assert_called_once()
            mock_recorder.save_recording.assert_called_once()
            
            assert result == "/tmp/test_recording.wav"

    @patch('whispy.recorder.sd.query_devices')
    @patch('whispy.recorder.sd.default')
    def test_check_audio_devices(self, mock_default, mock_query):
        """Test checking audio devices."""
        mock_devices = [
            {'name': 'Built-in Microphone', 'max_input_channels': 1},
            {'name': 'Built-in Output', 'max_output_channels': 2}
        ]
        mock_query.return_value = mock_devices
        mock_default.device = [0, 1]  # input, output
        
        result = check_audio_devices()
        
        assert 'devices' in result
        assert 'default_input' in result
        assert 'default_input_info' in result
        assert result['devices'] == mock_devices
        assert result['default_input'] == 0
        assert result['default_input_info'] == mock_devices[0]

    @patch('whispy.recorder.sd.query_devices')
    def test_check_audio_devices_failure(self, mock_query):
        """Test audio device check failure."""
        mock_query.side_effect = Exception("Device error")
        
        with pytest.raises(Exception):  # WhispyError
            check_audio_devices()

    @patch('whispy.recorder.AudioRecorder')
    @patch('time.sleep')
    def test_test_microphone_working(self, mock_sleep, mock_recorder_class):
        """Test microphone test when working."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        
        # Simulate audio with some signal
        audio_data = np.array([0.1, 0.2, 0.1, 0.2] * 1000)  # Repeating pattern
        mock_recorder.stop_recording.return_value = audio_data
        
        result = test_microphone()
        
        assert result is True
        mock_recorder.start_recording.assert_called_once()
        mock_sleep.assert_called_once_with(1)
        mock_recorder.stop_recording.assert_called_once()

    @patch('whispy.recorder.AudioRecorder')
    @patch('time.sleep')
    def test_test_microphone_silent(self, mock_sleep, mock_recorder_class):
        """Test microphone test with silence."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        
        # Simulate silence (very low amplitude)
        audio_data = np.array([0.0001, 0.0001, 0.0001] * 1000)
        mock_recorder.stop_recording.return_value = audio_data
        
        result = test_microphone()
        
        assert result is False

    @patch('whispy.recorder.AudioRecorder')
    def test_test_microphone_no_data(self, mock_recorder_class):
        """Test microphone test with no data."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        mock_recorder.stop_recording.return_value = np.array([])
        
        result = test_microphone()
        
        assert result is False

    @patch('whispy.recorder.AudioRecorder')
    def test_test_microphone_failure(self, mock_recorder_class):
        """Test microphone test failure."""
        mock_recorder_class.side_effect = Exception("Microphone error")
        
        result = test_microphone()
        
        assert result is False


# Test markers
pytestmark = pytest.mark.unit 