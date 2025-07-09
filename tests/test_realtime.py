"""
Tests for whispy.realtime module
"""

import threading
import time
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from whispy.realtime import RealtimeTranscriber, run_realtime_transcription
from whispy import realtime
from whispy import WhispyError


class TestRealtimeTranscriber:
    """Test the RealtimeTranscriber class"""
    
    def test_init_with_defaults(self):
        """Test initializing with default parameters"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                assert transcriber.model_path == '/path/to/model.bin'
                assert transcriber.language is None
                assert transcriber.chunk_duration == 3.0
                assert transcriber.overlap_duration == 1.0
                assert transcriber.silence_threshold == 0.01
                assert transcriber.sample_rate == 16000
                assert transcriber.verbose is False
                assert transcriber.is_running is False
    
    def test_init_with_custom_params(self):
        """Test initializing with custom parameters"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber(
                    model_path='/custom/model.bin',
                    language='en',
                    chunk_duration=2.5,
                    overlap_duration=0.5,
                    silence_threshold=0.02,
                    sample_rate=22050,
                    verbose=True
                )
                
                assert transcriber.model_path == '/custom/model.bin'
                assert transcriber.language == 'en'
                assert transcriber.chunk_duration == 2.5
                assert transcriber.overlap_duration == 0.5
                assert transcriber.silence_threshold == 0.02
                assert transcriber.sample_rate == 22050
                assert transcriber.verbose is True
    
    def test_init_no_whisper_cli(self):
        """Test initialization when whisper-cli is not found"""
        with patch('whispy.realtime.find_whisper_cli', return_value=None):
            with pytest.raises(WhispyError, match="whisper-cli not found"):
                RealtimeTranscriber()
    
    def test_init_no_model(self):
        """Test initialization when no model is found"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value=None):
                with pytest.raises(WhispyError, match="No model file found"):
                    RealtimeTranscriber()
    
    def test_detect_voice_activity(self):
        """Test voice activity detection"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber(silence_threshold=0.01)
                
                # Test silence (low amplitude)
                silent_audio = np.random.normal(0, 0.005, 1000)
                assert not transcriber._detect_voice_activity(silent_audio)
                
                # Test speech (high amplitude)
                speech_audio = np.random.normal(0, 0.05, 1000)
                assert transcriber._detect_voice_activity(speech_audio)
                
                # Test empty audio
                empty_audio = np.array([])
                assert not transcriber._detect_voice_activity(empty_audio)
    
    @patch('whispy.realtime.wavfile.write')
    @patch('tempfile.NamedTemporaryFile')
    def test_save_chunk_to_temp_file(self, mock_tempfile, mock_wavfile_write):
        """Test saving audio chunk to temporary file"""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = '/tmp/test_chunk.wav'
        mock_tempfile.return_value = mock_file
        
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                # Test audio chunk
                audio_chunk = np.random.normal(0, 0.1, 1000)
                temp_path = transcriber._save_chunk_to_temp_file(audio_chunk)
                
                assert temp_path == '/tmp/test_chunk.wav'
                mock_file.close.assert_called_once()
                mock_wavfile_write.assert_called_once()
    
    @patch('whispy.realtime.transcribe_file')
    @patch('os.unlink')
    def test_transcribe_chunk_success(self, mock_unlink, mock_transcribe_file):
        """Test successful chunk transcription"""
        mock_transcribe_file.return_value = "  Hello world  "
        
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                with patch.object(transcriber, '_save_chunk_to_temp_file', return_value='/tmp/test.wav'):
                    audio_chunk = np.random.normal(0, 0.1, 1000)
                    result = transcriber._transcribe_chunk(audio_chunk)
                    
                    assert result == "Hello world"
                    mock_transcribe_file.assert_called_once()
                    mock_unlink.assert_called_once_with('/tmp/test.wav')
    
    @patch('whispy.realtime.transcribe_file')
    def test_transcribe_chunk_error(self, mock_transcribe_file):
        """Test chunk transcription error handling"""
        mock_transcribe_file.side_effect = Exception("Transcription failed")
        
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                with patch.object(transcriber, '_save_chunk_to_temp_file', return_value='/tmp/test.wav'):
                    audio_chunk = np.random.normal(0, 0.1, 1000)
                    result = transcriber._transcribe_chunk(audio_chunk)
                    
                    assert result == ""
    
    @patch('whispy.realtime.sd.InputStream')
    @patch('threading.Thread')
    def test_start_stop(self, mock_thread, mock_input_stream):
        """Test starting and stopping transcription"""
        mock_stream = Mock()
        mock_input_stream.return_value = mock_stream
        
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                # Test start
                transcriber.start()
                assert transcriber.is_running is True
                mock_stream.start.assert_called_once()
                
                # Test stop
                transcriber.stop()
                assert transcriber.is_running is False
                mock_stream.stop.assert_called_once()
                mock_stream.close.assert_called_once()
    
    def test_get_transcript_history(self):
        """Test getting transcript history"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                # Add some history
                transcriber.transcript_history = [
                    {'timestamp': 123456, 'text': 'Hello', 'chunk': 0},
                    {'timestamp': 123457, 'text': 'World', 'chunk': 1}
                ]
                
                history = transcriber.get_transcript_history()
                assert len(history) == 2
                assert history[0]['text'] == 'Hello'
                assert history[1]['text'] == 'World'
    
    def test_get_full_transcript(self):
        """Test getting full transcript"""
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber()
                
                # Test empty history
                assert transcriber.get_full_transcript() == ""
                
                # Add some history
                transcriber.transcript_history = [
                    {'timestamp': 123456, 'text': 'Hello', 'chunk': 0},
                    {'timestamp': 123457, 'text': 'World', 'chunk': 1}
                ]
                
                full_transcript = transcriber.get_full_transcript()
                assert full_transcript == "Hello World"
                
                # Test custom separator
                full_transcript = transcriber.get_full_transcript(separator="\n")
                assert full_transcript == "Hello\nWorld"


class TestRunRealtimeTranscription:
    """Test the run_realtime_transcription function"""
    
    @patch('whispy.realtime.RealtimeTranscriber')
    @patch('signal.signal')
    def test_run_realtime_transcription_success(self, mock_signal, mock_transcriber_class):
        """Test successful real-time transcription"""
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.get_full_transcript.return_value = "Hello world"
        mock_transcriber_class.return_value = mock_transcriber
        
        # Mock signal handling
        interrupted = threading.Event()
        
        def mock_signal_handler(signum, handler):
            # Simulate immediate interrupt for testing
            interrupted.set()
            return Mock()
        
        mock_signal.side_effect = mock_signal_handler
        
        # Mock the interrupted event
        with patch('threading.Event') as mock_event:
            mock_event.return_value = interrupted
            
            result = run_realtime_transcription(
                model_path='/path/to/model.bin',
                language='en',
                verbose=True
            )
            
            assert result == "Hello world"
            mock_transcriber.start.assert_called_once()
            mock_transcriber.stop.assert_called_once()
    
    @patch('whispy.realtime.RealtimeTranscriber')
    @patch('signal.signal')
    def test_run_realtime_transcription_with_output_file(self, mock_signal, mock_transcriber_class):
        """Test real-time transcription with output file"""
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.get_full_transcript.return_value = "Hello world"
        mock_transcriber_class.return_value = mock_transcriber
        
        # Mock signal handling
        interrupted = threading.Event()
        mock_signal.return_value = Mock()
        
        with patch('threading.Event') as mock_event:
            mock_event.return_value = interrupted
            interrupted.set()  # Immediate interrupt
            
            with patch('pathlib.Path.write_text') as mock_write_text:
                result = run_realtime_transcription(
                    output_file='output.txt',
                    verbose=False
                )
                
                assert result == "Hello world"
                mock_write_text.assert_called_once_with("Hello world", encoding='utf-8')
    
    @patch('whispy.realtime.RealtimeTranscriber')
    def test_run_realtime_transcription_error(self, mock_transcriber_class):
        """Test real-time transcription error handling"""
        mock_transcriber_class.side_effect = WhispyError("Setup failed")
        
        with pytest.raises(WhispyError, match="Setup failed"):
            run_realtime_transcription()


class TestTestRealtimeSetup:
    """Test the test_realtime_setup function"""
    
    @patch('whispy.realtime.sd.query_devices')
    @patch('whispy.realtime.sd.default.device', [0, 1])
    @patch('whispy.realtime.find_default_model')
    @patch('whispy.realtime.find_whisper_cli')
    def test_test_realtime_setup_success(self, mock_find_cli, mock_find_model, mock_query_devices):
        """Test successful setup test"""
        mock_find_cli.return_value = '/usr/bin/whisper-cli'
        mock_find_model.return_value = '/path/to/model.bin'
        mock_query_devices.return_value = [
            {'name': 'Built-in Microphone', 'max_input_channels': 2}
        ]
        
        result = realtime.test_realtime_setup()
        assert result is True
    
    @patch('whispy.realtime.find_whisper_cli')
    def test_test_realtime_setup_no_cli(self, mock_find_cli):
        """Test setup test when whisper-cli is not found"""
        mock_find_cli.return_value = None
        
        result = realtime.test_realtime_setup()
        assert result is False
    
    @patch('whispy.realtime.find_whisper_cli')
    @patch('whispy.realtime.find_default_model')
    def test_test_realtime_setup_no_model(self, mock_find_model, mock_find_cli):
        """Test setup test when no model is found"""
        mock_find_cli.return_value = '/usr/bin/whisper-cli'
        mock_find_model.return_value = None
        
        result = realtime.test_realtime_setup()
        assert result is False
    
    @patch('whispy.realtime.find_whisper_cli')
    @patch('whispy.realtime.find_default_model')
    @patch('whispy.realtime.sd.query_devices')
    def test_test_realtime_setup_audio_error(self, mock_query_devices, mock_find_model, mock_find_cli):
        """Test setup test with audio device error"""
        mock_find_cli.return_value = '/usr/bin/whisper-cli'
        mock_find_model.return_value = '/path/to/model.bin'
        mock_query_devices.side_effect = Exception("Audio error")
        
        result = realtime.test_realtime_setup()
        assert result is False


@pytest.mark.integration
class TestRealtimeIntegration:
    """Integration tests for real-time transcription"""
    
    @patch('whispy.realtime.sd.InputStream')
    @patch('whispy.realtime.transcribe_file')
    def test_realtime_transcription_flow(self, mock_transcribe_file, mock_input_stream):
        """Test the complete real-time transcription flow"""
        # Mock audio stream
        mock_stream = Mock()
        mock_input_stream.return_value = mock_stream
        
        # Mock transcription
        mock_transcribe_file.return_value = "This is a test"
        
        with patch('whispy.realtime.find_whisper_cli', return_value='/usr/bin/whisper-cli'):
            with patch('whispy.realtime.find_default_model', return_value='/path/to/model.bin'):
                transcriber = RealtimeTranscriber(verbose=True)
                
                # Track transcript updates
                transcripts = []
                
                def on_transcript(text, is_final):
                    transcripts.append(text)
                
                transcriber.on_transcript = on_transcript
                
                # Start transcription
                transcriber.start()
                
                # Simulate audio input
                audio_data = np.random.normal(0, 0.1, 16000)  # 1 second of audio
                transcriber.audio_buffer.extend(audio_data)
                
                # Let it process briefly
                time.sleep(0.1)
                
                # Stop transcription
                transcriber.stop()
                
                # Check that stream was handled
                mock_stream.start.assert_called_once()
                mock_stream.stop.assert_called_once()
                mock_stream.close.assert_called_once() 