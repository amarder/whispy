"""
Audio recording functionality for whispy.

This module provides functions to record audio from the microphone
and save it to a file for transcription.
"""

import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import WhispyError


class AudioRecorder:
    """
    Audio recorder class for capturing microphone input.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, show_volume: bool = True):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz (16kHz is optimal for Whisper)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            show_volume: Whether to show real-time volume indicator
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.show_volume = show_volume
        self.current_volume = 0.0
        self.peak_volume = 0.0
        self.console = Console()
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Recording status: {status}")
        
        if self.recording:
            self.audio_data.append(indata.copy())
            
            # Calculate volume for display
            if self.show_volume:
                audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                rms = np.sqrt(np.mean(audio_data ** 2))
                self.current_volume = rms
                self.peak_volume = max(self.peak_volume, rms)
    
    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        if self.recording:
            return
        
        self.recording = True
        self.audio_data = []
        
        try:
            # Create audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            
        except Exception as e:
            self.recording = False
            raise WhispyError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the audio data.
        
        Returns:
            Audio data as numpy array
        """
        if not self.recording:
            return np.array([])
        
        self.recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.audio_data:
            return np.array([])
        
        # Concatenate all audio chunks
        audio_array = np.concatenate(self.audio_data, axis=0)
        
        # Convert to proper format for wav file
        if self.channels == 1:
            audio_array = audio_array.flatten()
        
        return audio_array
    
    def save_recording(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Path to save the WAV file
        """
        if len(audio_data) == 0:
            raise WhispyError("No audio data to save")
        
        try:
            # Convert float32 to int16 for WAV format
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV file
            wavfile.write(output_path, self.sample_rate, audio_int16)
            
        except Exception as e:
            raise WhispyError(f"Failed to save audio file: {e}")
    
    def _create_volume_display(self) -> Panel:
        """Create a visual volume indicator panel."""
        # Normalize volume to 0-100 scale
        volume_percent = min(100, self.current_volume * 1000)  # Scale factor for display
        peak_percent = min(100, self.peak_volume * 1000)
        
        # Create volume bar
        bar_length = 40
        filled_length = int(bar_length * volume_percent / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Color coding based on volume level
        if volume_percent < 10:
            volume_color = "dim white"
            status = "🔇 Very quiet"
        elif volume_percent < 30:
            volume_color = "green"
            status = "🔉 Good"
        elif volume_percent < 70:
            volume_color = "yellow"
            status = "🔊 Loud"
        else:
            volume_color = "red"
            status = "📢 Very loud"
        
        # Create display text
        volume_text = Text()
        volume_text.append(f"\nVolume: ", style="white")
        volume_text.append(f"{bar}\n", style="green")
        
        return Panel(
            volume_text,
            title="🎤 Recording Audio",
            border_style="blue",
            expand=False
        )


def record_audio_until_interrupt(
    sample_rate: int = 16000,
    channels: int = 1,
    output_path: Optional[str] = None,
    show_volume: bool = True
) -> str:
    """
    Record audio from microphone until interrupted (Ctrl+C).
    
    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        output_path: Path to save the recording (optional)
        show_volume: Whether to show real-time volume indicator
        
    Returns:
        Path to the saved audio file
    """
    recorder = AudioRecorder(sample_rate, channels, show_volume)
    console = Console()
    
    # Create output file if not specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            prefix='whispy_recording_'
        )
        output_path = temp_file.name
        temp_file.close()
    
    # Set up signal handler for graceful shutdown
    recording_stopped = threading.Event()
    
    def signal_handler(signum, frame):
        recording_stopped.set()
    
    # Register signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        console.print("🎤 [bold blue]Starting audio recording...[/bold blue]")
        console.print("🔴 [blue]Recording in progress - Press Ctrl+C to stop[/blue]")
        
        recorder.start_recording()
        
        # Show volume indicator if enabled
        if show_volume:
            try:
                with Live(recorder._create_volume_display(), refresh_per_second=10) as live:
                    while not recording_stopped.is_set():
                        live.update(recorder._create_volume_display())
                        time.sleep(0.1)
            except Exception as e:
                # Fallback to simple waiting if Live display fails (e.g., in tests)
                if "threads can only be started once" in str(e):
                    console.print("🎤 [yellow]Recording (volume display disabled in test mode)[/yellow]")
                    recording_stopped.wait()
                else:
                    raise
        else:
            # Wait for interrupt signal without volume display
            recording_stopped.wait()
        
        console.print("\n💾 [yellow]Saving recording...[/yellow]")
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            recorder.save_recording(audio_data, output_path)
            duration = len(audio_data) / sample_rate
            console.print(f"✅ [green]Recording saved:[/green] {output_path}")
            console.print(f"📊 [blue]Duration:[/blue] {duration:.2f} seconds")
            if show_volume and hasattr(recorder, 'peak_volume') and isinstance(recorder.peak_volume, (int, float)):
                peak_percent = min(100, recorder.peak_volume * 1000)
                console.print(f"📈 [cyan]Peak volume:[/cyan] {peak_percent:.1f}%")
            return output_path
        else:
            console.print("⚠️  [yellow]No audio data recorded[/yellow]")
            # Clean up empty file
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise WhispyError("No audio data was recorded")
            
    except KeyboardInterrupt:
        # This shouldn't happen with signal handler, but just in case
        console.print("\n🛑 [yellow]Recording interrupted[/yellow]")
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            recorder.save_recording(audio_data, output_path)
            return output_path
        else:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise WhispyError("Recording was interrupted with no audio data")
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise WhispyError(f"Recording failed: {e}")
        
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def check_audio_devices() -> dict:
    """
    Check available audio devices and return information.
    
    Returns:
        Dictionary with device information
    """
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        return {
            'devices': devices,
            'default_input': default_input,
            'default_input_info': devices[default_input] if default_input is not None else None
        }
    except Exception as e:
        raise WhispyError(f"Failed to query audio devices: {e}")


def test_microphone() -> bool:
    """
    Test if microphone is working by recording a short sample.
    
    Returns:
        True if microphone is working, False otherwise
    """
    try:
        print("🎤 Testing microphone...")
        recorder = AudioRecorder()
        
        recorder.start_recording()
        
        # Record for 1 second
        import time
        time.sleep(1)
        
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            # Check if there's actual audio (not just silence)
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0.001:  # Threshold for detecting audio
                print("✅ Microphone is working!")
                return True
            else:
                print("⚠️  Microphone detected but no audio signal")
                return False
        else:
            print("❌ No audio data captured")
            return False
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False 