import ctypes
import os
import pathlib
import numpy as np
import librosa
from typing import List, Optional, Union
from . import get_library, WhispyError


class WhisperContext:
    """Wrapper for whisper_context from whisper.cpp"""
    
    def __init__(self, model_path: str):
        self.lib = get_library()
        self.model_path = model_path
        self.ctx = None
        self._setup_function_signatures()
        self._init_context()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for whisper.cpp API"""
        # whisper_context_default_params_by_ref
        self.lib.whisper_context_default_params_by_ref.argtypes = []
        self.lib.whisper_context_default_params_by_ref.restype = ctypes.c_void_p
        
        # whisper_init_from_file_with_params
        self.lib.whisper_init_from_file_with_params.argtypes = [
            ctypes.c_char_p,  # model path
            ctypes.c_void_p   # context params (pointer to struct)
        ]
        self.lib.whisper_init_from_file_with_params.restype = ctypes.c_void_p
        
        # whisper_full_default_params_by_ref
        self.lib.whisper_full_default_params_by_ref.argtypes = [ctypes.c_int]
        self.lib.whisper_full_default_params_by_ref.restype = ctypes.c_void_p
        
        # whisper_full
        self.lib.whisper_full.argtypes = [
            ctypes.c_void_p,  # context
            ctypes.c_void_p,  # params (pointer to struct)
            ctypes.POINTER(ctypes.c_float),  # samples
            ctypes.c_int      # n_samples
        ]
        self.lib.whisper_full.restype = ctypes.c_int
        
        # whisper_full_n_segments
        self.lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        self.lib.whisper_full_n_segments.restype = ctypes.c_int
        
        # whisper_full_get_segment_text
        self.lib.whisper_full_get_segment_text.argtypes = [
            ctypes.c_void_p,  # context
            ctypes.c_int      # segment index
        ]
        self.lib.whisper_full_get_segment_text.restype = ctypes.c_char_p
        
        # whisper_full_get_segment_t0
        self.lib.whisper_full_get_segment_t0.argtypes = [
            ctypes.c_void_p,  # context
            ctypes.c_int      # segment index
        ]
        self.lib.whisper_full_get_segment_t0.restype = ctypes.c_int64
        
        # whisper_full_get_segment_t1
        self.lib.whisper_full_get_segment_t1.argtypes = [
            ctypes.c_void_p,  # context
            ctypes.c_int      # segment index
        ]
        self.lib.whisper_full_get_segment_t1.restype = ctypes.c_int64
        
        # whisper_free
        self.lib.whisper_free.argtypes = [ctypes.c_void_p]
        self.lib.whisper_free.restype = None
        
        # whisper_free_context_params
        self.lib.whisper_free_context_params.argtypes = [ctypes.c_void_p]
        self.lib.whisper_free_context_params.restype = None
        
        # whisper_free_params
        self.lib.whisper_free_params.argtypes = [ctypes.c_void_p]
        self.lib.whisper_free_params.restype = None
    
    def _init_context(self):
        """Initialize the whisper context"""
        if not os.path.exists(self.model_path):
            raise WhispyError(f"Model file not found: {self.model_path}")
        
        # Get default context parameters
        self.ctx_params = self.lib.whisper_context_default_params_by_ref()
        
        # Initialize context from file
        self.ctx = self.lib.whisper_init_from_file_with_params(
            self.model_path.encode('utf-8'),
            self.ctx_params
        )
        
        if not self.ctx:
            # Clean up parameters on failure
            self.lib.whisper_free_context_params(self.ctx_params)
            raise WhispyError(f"Failed to initialize whisper context from {self.model_path}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[dict]:
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detect.
            
        Returns:
            List of transcription segments with text and timestamps
        """
        if not os.path.exists(audio_path):
            raise WhispyError(f"Audio file not found: {audio_path}")
        
        # Load audio using librosa
        try:
            # Load audio and resample to 16kHz (whisper requirement)
            audio, _ = librosa.load(audio_path, sr=16000)
            
            # Convert to float32 array for whisper
            audio_data = audio.astype(np.float32)
            
        except Exception as e:
            raise WhispyError(f"Failed to load audio file {audio_path}: {e}")
        
        # Get default full parameters
        # WHISPER_SAMPLING_GREEDY = 0
        params = self.lib.whisper_full_default_params_by_ref(0)
        
        # TODO: Set language if provided
        # For now, we'll use auto-detection
        
        # Convert numpy array to ctypes array
        audio_ctypes = (ctypes.c_float * len(audio_data))(*audio_data)
        
        # Run transcription
        result = self.lib.whisper_full(
            self.ctx,
            params,
            audio_ctypes,
            len(audio_data)
        )
        
        if result != 0:
            # Clean up params on failure
            self.lib.whisper_free_params(params)
            raise WhispyError(f"Whisper transcription failed with error code: {result}")
        
        # Get transcription results
        segments = []
        n_segments = self.lib.whisper_full_n_segments(self.ctx)
        
        for i in range(n_segments):
            # Get segment text
            text_ptr = self.lib.whisper_full_get_segment_text(self.ctx, i)
            if text_ptr:
                text = text_ptr.decode('utf-8')
            else:
                text = ""
            
            # Get timestamps (in centiseconds, convert to seconds)
            t0 = self.lib.whisper_full_get_segment_t0(self.ctx, i) / 100.0
            t1 = self.lib.whisper_full_get_segment_t1(self.ctx, i) / 100.0
            
            segments.append({
                'text': text,
                'start': t0,
                'end': t1
            })
        
        # Clean up params after successful transcription
        self.lib.whisper_free_params(params)
        
        return segments
    
    def __del__(self):
        """Clean up whisper context"""
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.whisper_free(self.ctx)
        if hasattr(self, 'ctx_params') and self.ctx_params:
            self.lib.whisper_free_context_params(self.ctx_params)


def transcribe_file(audio_path: str, model_path: str, language: Optional[str] = None) -> str:
    """
    Transcribe an audio file using whisper.cpp
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the whisper model file
        language: Language code (optional)
        
    Returns:
        Transcribed text
    """
    ctx = WhisperContext(model_path)
    segments = ctx.transcribe(audio_path, language)
    
    # Combine all segments into a single text
    full_text = " ".join(segment['text'].strip() for segment in segments if segment['text'].strip())
    
    return full_text


def find_default_model() -> Optional[str]:
    """
    Try to find a default whisper model in common locations
    
    Returns:
        Path to model file if found, None otherwise
    """
    # Common model locations
    common_locations = [
        "models/ggml-base.en.bin",
        "models/ggml-base.bin",
        "models/ggml-small.en.bin",
        "models/ggml-small.bin",
        "models/ggml-tiny.en.bin",
        "models/ggml-tiny.bin",
        "whisper.cpp/models/ggml-base.en.bin",
        "whisper.cpp/models/ggml-base.bin",
        "whisper.cpp/models/ggml-small.en.bin",
        "whisper.cpp/models/ggml-small.bin",
        "whisper.cpp/models/ggml-tiny.en.bin",
        "whisper.cpp/models/ggml-tiny.bin",
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