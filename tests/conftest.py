"""
Pytest configuration and shared fixtures for whispy tests.
"""

import os
import pytest
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def whisper_cpp_dir(project_root):
    """Get the whisper.cpp directory path."""
    return project_root / "whisper.cpp"


@pytest.fixture(scope="session") 
def sample_audio_available(whisper_cpp_dir):
    """Check if sample audio files are available."""
    jfk_wav = whisper_cpp_dir / "samples" / "jfk.wav"
    jfk_mp3 = whisper_cpp_dir / "samples" / "jfk.mp3"
    
    return jfk_wav.exists() or jfk_mp3.exists()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Store original working directory
    original_cwd = os.getcwd()
    
    yield
    
    # Restore original working directory
    os.chdir(original_cwd)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add 'slow' marker to tests that likely take longer
        if any(keyword in item.name.lower() for keyword in ["transcribe", "build", "integration"]):
            item.add_marker(pytest.mark.slow)
        
        # Add 'integration' marker to integration tests
        if "integration" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)


def pytest_runtest_setup(item):
    """Setup before each test runs."""
    # Skip slow tests if --fast flag is used
    if item.config.getoption("--fast", default=False):
        if "slow" in [mark.name for mark in item.iter_markers()]:
            pytest.skip("Skipping slow test (--fast flag used)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true", 
        default=False,
        help="Run integration tests"
    ) 