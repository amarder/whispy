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
)
from whispy import recorder


class TestAudioRecorder:
    """Test AudioRecorder class."""

    def test_init(self):
        """Test AudioRecorder initialization."""
        recorder = AudioRecorder(sample_rate=22050, channels=2)
        
        assert recorder.sample_rate == 22050
        assert recorder.channels == 2
        assert recorder.recording is False
        assert recorder.stream is None
        assert recorder.wav_file is None
        assert recorder.output_path is None
        assert recorder.frames_recorded == 0

    @patch('whispy.recorder.sd.InputStream')
    @patch('whispy.recorder.wave.open')
    def test_start_recording(self, mock_wave_open, mock_stream_class):
        """Test starting recording."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        mock_wav_file = MagicMock()
        mock_wave_open.return_value = mock_wav_file
        
        recorder = AudioRecorder()
        recorder.start_recording("/tmp/test.wav")
        
        assert recorder.recording is True
        assert recorder.output_path == "/tmp/test.wav"
        mock_wave_open.assert_called_once_with("/tmp/test.wav", 'wb')
        mock_stream_class.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        recorder = AudioRecorder()
        recorder.recording = True
        
        # Should not raise an error, just return
        recorder.start_recording("/tmp/test.wav")
        assert recorder.recording is True

    @patch('whispy.recorder.sd.InputStream')
    @patch('whispy.recorder.wave.open')
    def test_start_recording_failure(self, mock_wave_open, mock_stream_class):
        """Test recording start failure."""
        mock_wav_file = MagicMock()
        mock_wave_open.return_value = mock_wav_file
        mock_stream_class.side_effect = Exception("Audio device error")
        
        recorder = AudioRecorder()
        
        with pytest.raises(Exception):  # WhispyError
            recorder.start_recording("/tmp/test.wav")
        
        assert recorder.recording is False

    def test_stop_recording_not_started(self):
        """Test stopping recording when not started."""
        recorder = AudioRecorder()
        
        result = recorder.stop_recording()
        
        assert isinstance(result, str)
        assert result == ""

    def test_stop_recording_with_data(self):
        """Test stopping recording with audio data."""
        mock_stream = MagicMock()
        mock_wav_file = MagicMock()
        
        recorder = AudioRecorder()
        recorder.recording = True
        recorder.stream = mock_stream
        recorder.wav_file = mock_wav_file
        recorder.output_path = "/tmp/test.wav"
        recorder.frames_recorded = 1000  # Simulate some recorded frames
        
        result = recorder.stop_recording()
        
        assert recorder.recording is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert recorder.stream is None
        mock_wav_file.close.assert_called_once()
        
        # Check that result is the output path
        assert isinstance(result, str)
        assert result == "/tmp/test.wav"

    def test_get_recording_duration(self):
        """Test getting recording duration."""
        recorder = AudioRecorder()
        
        # Test with no recorded frames
        assert recorder.get_recording_duration() == 0.0
        
        # Test with some recorded frames
        recorder.frames_recorded = 16000  # 1 second at 16kHz
        assert recorder.get_recording_duration() == 1.0
        
        # Test with different sample rate
        recorder.sample_rate = 44100
        recorder.frames_recorded = 44100  # 1 second at 44.1kHz  
        assert recorder.get_recording_duration() == 1.0


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
        mock_recorder.stop_recording.return_value = "/tmp/test_recording.wav"
        mock_recorder.get_recording_duration.return_value = 5.0  # 5 seconds
        mock_recorder.peak_volume = 0.5  # Some volume
        
        # Mock the recording process
        with patch('whispy.recorder.threading.Event') as mock_event_class:
            mock_event = MagicMock()
            mock_event_class.return_value = mock_event
            
            result = record_audio_until_interrupt()
            
            # Check that the process was set up correctly
            mock_recorder_class.assert_called_once()
            mock_recorder.start_recording.assert_called_once_with("/tmp/test_recording.wav")
            mock_event.wait.assert_called_once()
            mock_recorder.stop_recording.assert_called_once()
            mock_recorder.get_recording_duration.assert_called_once()
            
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
    @patch('whispy.recorder.time.sleep')
    @patch('whispy.recorder.tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_test_microphone_working(self, mock_unlink, mock_tempfile, mock_sleep, mock_recorder_class):
        """Test microphone test when working."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        
        # Setup tempfile mock
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test_mic.wav"
        mock_tempfile.return_value = mock_temp
        
        # Simulate successful recording with audio signal
        mock_recorder.get_recording_duration.return_value = 1.0  # 1 second recorded
        mock_recorder.peak_volume = 0.1  # Above threshold (0.001)
        
        result = recorder.test_microphone()
        
        assert result is True
        mock_recorder_class.assert_called_once_with(show_volume=False)
        mock_recorder.start_recording.assert_called_once_with("/tmp/test_mic.wav")
        mock_sleep.assert_called_once_with(1)
        mock_recorder.stop_recording.assert_called_once()
        mock_unlink.assert_called_once()

    @patch('whispy.recorder.AudioRecorder')
    @patch('time.sleep')
    def test_test_microphone_silent(self, mock_sleep, mock_recorder_class):
        """Test microphone test with silence."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        
        # Simulate silence (very low amplitude)
        audio_data = np.array([0.0001, 0.0001, 0.0001] * 1000)
        mock_recorder.stop_recording.return_value = audio_data
        
        result = recorder.test_microphone()
        
        assert result is False

    @patch('whispy.recorder.AudioRecorder')
    def test_test_microphone_no_data(self, mock_recorder_class):
        """Test microphone test with no data."""
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder
        mock_recorder.stop_recording.return_value = np.array([])
        
        result = recorder.test_microphone()
        
        assert result is False

    @patch('whispy.recorder.AudioRecorder')
    def test_test_microphone_failure(self, mock_recorder_class):
        """Test microphone test failure."""
        mock_recorder_class.side_effect = Exception("Microphone error")
        
        result = recorder.test_microphone()
        
        assert result is False


# Test markers
pytestmark = pytest.mark.unit 