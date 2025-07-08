"""
Real-time transcription functionality for whispy.

This module provides real-time audio transcription from microphone input
using streaming audio chunks and continuous processing.
"""

import os
import queue
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List
import collections

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from . import WhispyError
from .transcribe import transcribe_file, find_whisper_cli, find_default_model


class RealtimeTranscriber:
    """
    Real-time audio transcription class.
    
    Continuously captures audio from microphone and transcribes it
    in overlapping chunks for smooth real-time transcription.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        language: Optional[str] = None,
        chunk_duration: float = 3.0,
        overlap_duration: float = 1.0,
        silence_threshold: float = 0.01,
        sample_rate: int = 16000,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        verbose: bool = False
    ):
        """
        Initialize the real-time transcriber.
        
        Args:
            model_path: Path to whisper model file
            language: Language code for transcription
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks in seconds  
            silence_threshold: RMS threshold below which audio is considered silence
            sample_rate: Audio sample rate (16kHz optimal for Whisper)
            on_transcript: Callback function for transcript updates
            verbose: Enable verbose output
        """
        self.model_path = model_path or find_default_model()
        self.language = language
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript
        self.verbose = verbose
        
        # Audio processing
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.audio_queue = queue.Queue()
        self.audio_buffer = collections.deque(maxlen=self.chunk_samples * 2)
        
        # Transcription state
        self.is_running = False
        self.transcription_thread = None
        self.audio_thread = None
        self.last_transcript = ""
        self.transcript_history = []
        
        # Check requirements
        if not find_whisper_cli():
            raise WhispyError("whisper-cli not found. Please build whisper.cpp first.")
        
        if not self.model_path:
            raise WhispyError("No model file found. Please download a model first.")
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback to capture microphone input."""
        if status:
            if self.verbose:
                print(f"Audio status: {status}")
        
        if self.is_running:
            # Convert to mono and add to buffer
            audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            self.audio_buffer.extend(audio_data)
    
    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Simple voice activity detection based on RMS energy."""
        if len(audio_chunk) == 0:
            return False
        
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.silence_threshold
    
    def _save_chunk_to_temp_file(self, audio_chunk: np.ndarray) -> str:
        """Save audio chunk to temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            prefix='whispy_chunk_'
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Convert to int16 and save
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        wavfile.write(temp_path, self.sample_rate, audio_int16)
        
        return temp_path
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk."""
        try:
            # Save chunk to temp file
            temp_file = self._save_chunk_to_temp_file(audio_chunk)
            
            # Transcribe using whisper
            transcript = transcribe_file(
                audio_path=temp_file,
                model_path=self.model_path,
                language=self.language
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
            
            return transcript.strip()
            
        except Exception as e:
            if self.verbose:
                print(f"Transcription error: {e}")
            return ""
    
    def _process_audio_chunks(self):
        """Background thread to process audio chunks for transcription."""
        chunk_counter = 0
        
        while self.is_running:
            try:
                # Wait for enough audio data
                if len(self.audio_buffer) < self.chunk_samples:
                    time.sleep(0.1)
                    continue
                
                # Extract chunk with overlap
                chunk_start = max(0, len(self.audio_buffer) - self.chunk_samples)
                audio_chunk = np.array(list(self.audio_buffer)[chunk_start:])
                
                # Check for voice activity
                if not self._detect_voice_activity(audio_chunk):
                    if self.verbose and chunk_counter % 10 == 0:
                        print("🔇 Listening... (silence detected)")
                    time.sleep(0.5)
                    chunk_counter += 1
                    continue
                
                if self.verbose:
                    print(f"🎤 Processing audio chunk {chunk_counter}...")
                
                # Transcribe chunk
                transcript = self._transcribe_chunk(audio_chunk)
                
                if transcript and transcript != self.last_transcript:
                    self.last_transcript = transcript
                    self.transcript_history.append({
                        'timestamp': time.time(),
                        'text': transcript,
                        'chunk': chunk_counter
                    })
                    
                    # Call callback if provided
                    if self.on_transcript:
                        is_final = False  # In streaming, no chunk is truly final
                        self.on_transcript(transcript, is_final)
                    
                    if self.verbose:
                        print(f"📝 Transcript {chunk_counter}: {transcript}")
                
                # Clear processed samples to avoid re-processing
                samples_to_remove = self.chunk_samples - self.overlap_samples
                for _ in range(min(samples_to_remove, len(self.audio_buffer))):
                    self.audio_buffer.popleft()
                
                chunk_counter += 1
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing audio: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start real-time transcription."""
        if self.is_running:
            return
        
        self.is_running = True
        self.audio_buffer.clear()
        self.transcript_history.clear()
        
        try:
            # Start audio stream
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                dtype=np.float32,
                blocksize=1024
            )
            self.audio_stream.start()
            
            # Start transcription thread
            self.transcription_thread = threading.Thread(
                target=self._process_audio_chunks,
                daemon=True
            )
            self.transcription_thread.start()
            
            if self.verbose:
                print("🎤 Real-time transcription started")
                print(f"📊 Chunk duration: {self.chunk_duration}s")
                print(f"📊 Overlap: {self.overlap_duration}s")
                print(f"📊 Sample rate: {self.sample_rate}Hz")
            
        except Exception as e:
            self.is_running = False
            raise WhispyError(f"Failed to start real-time transcription: {e}")
    
    def stop(self):
        """Stop real-time transcription."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Wait for transcription thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        if self.verbose:
            print("🛑 Real-time transcription stopped")
    
    def get_transcript_history(self) -> List[dict]:
        """Get the history of all transcribed chunks."""
        return self.transcript_history.copy()
    
    def get_full_transcript(self, separator: str = " ") -> str:
        """Get the full transcript by joining all chunks."""
        if not self.transcript_history:
            return ""
        
        # Simple approach: join all chunks
        # Could be improved with smart deduplication
        texts = [item['text'] for item in self.transcript_history]
        return separator.join(texts)


def run_realtime_transcription(
    model_path: Optional[str] = None,
    language: Optional[str] = None,
    chunk_duration: float = 3.0,
    overlap_duration: float = 1.0,
    silence_threshold: float = 0.01,
    output_file: Optional[str] = None,
    verbose: bool = False,
    show_chunks: bool = False
) -> str:
    """
    Run real-time transcription until interrupted.
    
    Args:
        model_path: Path to whisper model
        language: Language code
        chunk_duration: Duration of each processing chunk
        overlap_duration: Overlap between chunks
        silence_threshold: Voice activity detection threshold
        output_file: Optional file to save final transcript
        verbose: Enable verbose output
        show_chunks: Show individual chunk transcripts
        
    Returns:
        Final combined transcript
    """
    # Display callback for real-time output
    def on_transcript_update(text: str, is_final: bool):
        if show_chunks:
            chunk_indicator = "✓" if is_final else "📝"
            print(f"{chunk_indicator} {text}")
        else:
            # For continuous mode, just print the text
            print(f"📝 {text}")
    
    # Create transcriber
    transcriber = RealtimeTranscriber(
        model_path=model_path,
        language=language,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        silence_threshold=silence_threshold,
        on_transcript=on_transcript_update,
        verbose=verbose
    )
    
    # Set up signal handler for graceful shutdown
    interrupted = threading.Event()
    
    def signal_handler(signum, frame):
        print("\n🛑 Stopping real-time transcription...")
        interrupted.set()
    
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("🎤 Starting real-time transcription...")
        print("🔴 Speak into your microphone - Press Ctrl+C to stop")
        
        if not show_chunks:
            print("📋 Continuous transcript:")
            print("-" * 50)
        
        transcriber.start()
        
        # Wait for interrupt
        interrupted.wait()
        
        transcriber.stop()
        
        # Get final transcript
        final_transcript = transcriber.get_full_transcript()
        
        if not show_chunks and final_transcript:
            print("-" * 50)
            print("📄 Final transcript:")
            print(final_transcript)
        
        # Save to file if requested
        if output_file and final_transcript:
            try:
                Path(output_file).write_text(final_transcript, encoding='utf-8')
                print(f"💾 Transcript saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving transcript: {e}")
        
        return final_transcript
        
    except KeyboardInterrupt:
        print("\n🛑 Transcription interrupted")
        transcriber.stop()
        return transcriber.get_full_transcript()
        
    except Exception as e:
        transcriber.stop()
        raise WhispyError(f"Real-time transcription failed: {e}")
        
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def test_realtime_setup() -> bool:
    """Test if real-time transcription can be set up properly."""
    try:
        # Check whisper-cli
        if not find_whisper_cli():
            print("❌ whisper-cli not found")
            return False
        
        # Check model
        if not find_default_model():
            print("❌ No default model found")
            return False
        
        # Test audio devices
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            if default_input is None:
                print("❌ No default audio input device")
                return False
            
            print(f"✅ Audio input: {devices[default_input]['name']}")
            
        except Exception as e:
            print(f"❌ Audio device error: {e}")
            return False
        
        print("✅ Real-time transcription setup OK")
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False 