import os
import pathlib
import subprocess
import json
import tempfile
import threading
import time
from typing import List, Optional, Union, Callable
from . import WhispyError


def find_whisper_cli() -> Optional[str]:
    """
    Find the whisper-cli binary in common locations
    
    Returns:
        Path to whisper-cli binary if found, None otherwise
    """
    # Common locations for whisper-cli
    common_locations = [
        "whisper.cpp/build/bin/whisper-cli",
        "whisper.cpp/build/whisper-cli", 
        "build/bin/whisper-cli",
        "build/whisper-cli",
        "/usr/local/bin/whisper-cli",
        "/opt/homebrew/bin/whisper-cli",
    ]
    
    # Check user directory first (for pip installs)
    user_home = pathlib.Path.home()
    user_locations = [
        user_home / ".whispy" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        user_home / ".whispy" / "whisper.cpp" / "build" / "whisper-cli",
    ]
    
    for cli_path in user_locations:
        if cli_path.exists() and cli_path.is_file():
            return str(cli_path)
    
    # Check current directory and parent directories
    current_dir = pathlib.Path.cwd()
    for _ in range(3):  # Check up to 3 parent directories
        for cli_path in common_locations:
            full_path = current_dir / cli_path
            if full_path.exists() and full_path.is_file():
                return str(full_path)
        current_dir = current_dir.parent
    
    # Check if whisper-cli is in PATH
    import shutil
    cli_in_path = shutil.which("whisper-cli")
    if cli_in_path:
        return cli_in_path
    
    return None


def transcribe_file(
    audio_path: str, 
    model_path: str, 
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> str:
    """
    Transcribe an audio file using whisper-cli
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the whisper model file
        language: Language code (optional)
        progress_callback: Callback function to report progress (0.0 to 1.0)
        
    Returns:
        Transcribed text
    """
    # Find whisper-cli binary
    whisper_cli = find_whisper_cli()
    if not whisper_cli:
        raise WhispyError(
            "whisper-cli binary not found. Please build whisper.cpp first:\n"
            "cd whisper.cpp && cmake -B build && cmake --build build -j --config Release"
        )
    
    # Validate inputs
    if not os.path.exists(audio_path):
        raise WhispyError(f"Audio file not found: {audio_path}")
    
    if not os.path.exists(model_path):
        raise WhispyError(f"Model file not found: {model_path}")
    
    # Build command
    cmd = [
        whisper_cli,
        "-m", model_path,
        "-f", audio_path,
        "--no-prints",  # Suppress debug output
        "--no-timestamps",  # Don't include timestamps in output
    ]
    
    # Add language if specified
    if language:
        cmd.extend(["-l", language])
    
    # Progress tracking setup
    start_time = time.time()
    process_completed = threading.Event()
    
    try:
        # Estimate processing time based on audio duration and file size
        audio_file_size = os.path.getsize(audio_path)
        
        def progress_tracker():
            """Track progress based on estimated duration."""
            # Rough estimation: ~2-4x real-time for transcription
            # Adjust this based on your system performance
            estimated_duration = max(5.0, audio_file_size / 1024 / 1024 * 8)  # Basic heuristic
            
            while not process_completed.is_set():
                elapsed = time.time() - start_time
                progress = min(0.95, elapsed / estimated_duration)  # Cap at 95% until completion
                
                if progress_callback:
                    progress_callback(progress)
                
                time.sleep(0.5)
        
        # Start progress tracking thread if callback provided
        progress_thread = None
        if progress_callback:
            progress_thread = threading.Thread(target=progress_tracker, daemon=True)
            progress_thread.start()
        
        # Run whisper-cli and capture stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Mark process as completed
        process_completed.set()
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
            if progress_thread:
                progress_thread.join(timeout=1.0)
        
        # Get transcript from stdout
        transcript = result.stdout.strip()
        if transcript:
            return transcript
        else:
            raise WhispyError("No transcription output found")
                
    except subprocess.CalledProcessError as e:
        process_completed.set()
        raise WhispyError(f"whisper-cli failed: {e.stderr}")
    except Exception as e:
        process_completed.set()
        raise WhispyError(f"Transcription failed: {e}")


def find_default_model() -> Optional[str]:
    """
    Try to find a default whisper model in common locations
    
    Returns:
        Path to model file if found, None otherwise
    """
    # Check user directory first (for pip installs)
    user_home = pathlib.Path.home()
    user_model_locations = [
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-base.en.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-base.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-small.en.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-small.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-tiny.en.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-tiny.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-medium.en.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-medium.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-large-v1.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-large-v2.bin",
        user_home / ".whispy" / "whisper.cpp" / "models" / "ggml-large-v3.bin",
    ]
    
    for model_path in user_model_locations:
        if model_path.exists():
            return str(model_path)
    
    # Common model locations (prioritize better models)
    common_locations = [
        # Local models directory
        "models/ggml-base.en.bin",
        "models/ggml-base.bin", 
        "models/ggml-small.en.bin",
        "models/ggml-small.bin",
        "models/ggml-tiny.en.bin",
        "models/ggml-tiny.bin",
        # whisper.cpp models directory
        "whisper.cpp/models/ggml-base.en.bin",
        "whisper.cpp/models/ggml-base.bin",
        "whisper.cpp/models/ggml-small.en.bin", 
        "whisper.cpp/models/ggml-small.bin",
        "whisper.cpp/models/ggml-tiny.en.bin",
        "whisper.cpp/models/ggml-tiny.bin",
        "whisper.cpp/models/ggml-medium.en.bin",
        "whisper.cpp/models/ggml-medium.bin",
        "whisper.cpp/models/ggml-large-v1.bin",
        "whisper.cpp/models/ggml-large-v2.bin",
        "whisper.cpp/models/ggml-large-v3.bin",
    ]
    
    # Check current directory and parent directories
    current_dir = pathlib.Path.cwd()
    for _ in range(3):  # Check up to 3 parent directories
        for model_name in common_locations:
            model_path = current_dir / model_name
            if model_path.exists():
                return str(model_path)
        current_dir = current_dir.parent
    
    return None


def clone_whisper_cpp() -> bool:
    """
    Clone whisper.cpp repository if it doesn't exist
    
    Returns:
        True if clone was successful, False otherwise
    """
    whisper_cpp_dir = pathlib.Path("whisper.cpp")
    if whisper_cpp_dir.exists() and any(whisper_cpp_dir.iterdir()):
        return True  # Already exists
    
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", str(whisper_cpp_dir)],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_whisper_cli() -> bool:
    """
    Try to build whisper-cli, cloning whisper.cpp if necessary
    
    Returns:
        True if build was successful, False otherwise
    """
    # First ensure whisper.cpp exists
    if not clone_whisper_cpp():
        return False
    
    whisper_cpp_dir = pathlib.Path("whisper.cpp")
    
    try:
        # Run cmake to configure
        subprocess.run(
            ["cmake", "-B", "build"],
            cwd=whisper_cpp_dir,
            check=True,
            capture_output=True
        )
        
        # Run cmake to build
        subprocess.run(
            ["cmake", "--build", "build", "-j", "--config", "Release"],
            cwd=whisper_cpp_dir,
            check=True,
            capture_output=True
        )
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False 