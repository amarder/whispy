# Whispy - Fast Speech Recognition CLI

A fast and efficient command-line interface for [whisper.cpp](https://github.com/ggerganov/whisper.cpp), providing automatic speech recognition with GPU acceleration.

## Features

- 🚀 **Fast transcription** using whisper.cpp with GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
- 🎯 **Simple CLI interface** for easy audio transcription
- 📁 **Multiple audio formats** supported (WAV, MP3, FLAC, OGG)
- 🌍 **Multi-language support** with automatic language detection
- 📝 **Flexible output** options (stdout, file)
- 🔧 **Auto-detection** of models and whisper-cli binary
- 🏗️ **Automatic building** of whisper.cpp if needed

## Installation

### Quick Install (Recommended)

Install directly from GitHub with automatic setup:

```bash
pip install git+https://github.com/amarder/whispy.git
```

This will automatically:
- Clone whisper.cpp to `~/.whispy/whisper.cpp`
- Build the whisper-cli binary with GPU acceleration
- Install the whispy CLI

### Manual Install

If you prefer to install manually:

#### Prerequisites

- Python 3.7+
- CMake 3.10+ (for building whisper.cpp)
- C++ compiler with C++17 support
- Git (for cloning whisper.cpp)

#### Steps

```bash
# Clone repository
git clone https://github.com/amarder/whispy.git
cd whispy

# Install whispy
pip install -e .

# Clone whisper.cpp if you don't have it
git clone https://github.com/ggerganov/whisper.cpp.git

# Build whisper-cli (or use: whispy build)
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
cd ..
```

### Download a model

After installation, download a model to use for transcription:

```bash
# For pip installs from GitHub
cd ~/.whispy/whisper.cpp
sh ./models/download-ggml-model.sh base.en

# For manual installs
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en

# Alternative: Download directly to models/
mkdir -p models
curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

## Usage

### Basic transcription

```bash
# Transcribe an audio file
whispy transcribe audio.wav

# Transcribe with explicit model
whispy transcribe audio.wav --model models/ggml-base.en.bin

# Transcribe with language specification
whispy transcribe audio.wav --language en

# Save transcript to file
whispy transcribe audio.wav --output transcript.txt

# Verbose output
whispy transcribe audio.wav --verbose
```

### System information

```bash
# Check system status
whispy info

# Show version
whispy version

# Build whisper-cli if needed
whispy build
```

### Supported audio formats

- WAV
- MP3  
- FLAC
- OGG

### Available models

Download models using whisper.cpp's script or directly:

- `tiny.en`, `tiny` - Fastest, least accurate
- `base.en`, `base` - Good balance of speed and accuracy
- `small.en`, `small` - Better accuracy
- `medium.en`, `medium` - High accuracy
- `large-v1`, `large-v2`, `large-v3` - Best accuracy, slower

## Examples

```bash
# Quick transcription with auto-detected model
whispy transcribe meeting.wav

# High-quality transcription
whispy transcribe interview.mp3 --model whisper.cpp/models/ggml-large-v3.bin

# Transcribe non-English audio
whispy transcribe spanish_audio.wav --language es

# Save results and show details
whispy transcribe podcast.mp3 --output transcript.txt --verbose
```

## Testing

Whispy includes a comprehensive test suite to ensure the CLI works correctly with different scenarios.

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run only unit tests
pytest tests/test_unit.py

# Run only CLI tests
pytest tests/test_cli.py

# Run tests with coverage
pytest --cov=whispy --cov-report=html

# Skip slow tests
pytest --fast
```

### Test Categories

- **Unit tests** (`tests/test_unit.py`): Test individual functions and modules
- **CLI tests** (`tests/test_cli.py`): Test command-line interface functionality
- **Integration tests**: Test full workflows with real audio files

### Using the Test Runner

```bash
# Use the convenience script
python run_tests.py --help

# Run unit tests only
python run_tests.py -t unit -v

# Run with coverage
python run_tests.py -c -v

# Run fast tests only
python run_tests.py -f
```

### Test Requirements

- pytest >= 7.0.0
- pytest-cov >= 4.0.0  
- pytest-mock >= 3.10.0
- Sample audio files (JFK sample from whisper.cpp)

### What's Tested

- ✅ CLI commands (help, version, info, transcribe)
- ✅ Audio file transcription with sample files
- ✅ Error handling for invalid files/models
- ✅ Output file generation
- ✅ Language options and verbose modes
- ✅ System requirements and binary detection
- ✅ Model file discovery and validation

## Development

### Project Structure

```
whispy/
├── whispy/
│   ├── __init__.py       # Package initialization
│   ├── cli.py           # Command-line interface
│   └── transcribe.py    # Core transcription logic
├── whisper.cpp/         # Git submodule (whisper.cpp source)
├── models/              # Model files directory
├── pyproject.toml       # Project configuration
└── README.md
```

### How it works

Whispy works as a wrapper around the `whisper-cli` binary from whisper.cpp:

1. **Auto-detection**: Finds whisper-cli binary and model files automatically
2. **Subprocess calls**: Runs whisper-cli as a subprocess for transcription
3. **Output parsing**: Captures and returns the transcribed text
4. **Performance**: Gets full GPU acceleration and optimizations from whisper.cpp

### Building from source

```bash
# Clone with whisper.cpp submodule
git clone --recursive https://github.com/your-username/whispy.git
cd whispy

# Install in development mode
pip install -e .

# Build whisper.cpp
whispy build
# OR manually:
# cd whisper.cpp && cmake -B build && cmake --build build -j --config Release
```

### Adding new features

The CLI is built with [Typer](https://typer.tiangolo.com/) and can be easily extended:

```python
@app.command()
def new_command():
    """Add a new command to the CLI"""
    console.print("New feature!")
```

## Performance

Whispy automatically uses the best available backend:

- **macOS**: Metal GPU acceleration  
- **Linux/Windows**: CUDA GPU acceleration (if available)
- **Fallback**: Optimized CPU with BLAS

Typical performance on Apple M1:
- ~10x faster than real-time for base.en model
- ~5x faster than real-time for large-v3 model

## Troubleshooting

### whisper-cli not found

```bash
# Check if whisper-cli exists
whispy info

# Build whisper-cli
whispy build

# Or build manually
cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
```

### No model found

```bash
# Download a model
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en

# Or specify model explicitly
whispy transcribe audio.wav --model /path/to/model.bin
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development setup

```bash
git clone --recursive https://github.com/your-username/whispy.git
cd whispy
pip install -e .
whispy build
```

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Fast C++ implementation of OpenAI's Whisper
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [Typer](https://typer.tiangolo.com/) - CLI framework 