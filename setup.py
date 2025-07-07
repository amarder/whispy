import os
import pathlib
import subprocess
import sys
import shutil
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

def setup_whisper_cpp():
    """Setup whisper.cpp submodule and build whisper-cli"""
    # Install to user directory so it persists after pip installation
    user_home = pathlib.Path.home()
    whispy_dir = user_home / ".whispy"
    whisper_cpp_path = whispy_dir / "whisper.cpp"
    
    # Create .whispy directory if it doesn't exist
    whispy_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Setting up whisper.cpp...")
    print("="*60)
    
    # Check if whisper.cpp exists and has content
    if not whisper_cpp_path.exists() or not any(whisper_cpp_path.iterdir()):
        print("📦 Cloning whisper.cpp...")
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/ggerganov/whisper.cpp.git",
                str(whisper_cpp_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓ Successfully cloned whisper.cpp")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone whisper.cpp: {e}")
            print("You can clone it manually after installation:")
            print("  git clone https://github.com/ggerganov/whisper.cpp.git")
            return False
    else:
        print("✓ whisper.cpp already exists")
    
    # Build whisper-cli
    print("🔨 Building whisper-cli...")
    build_dir = whisper_cpp_path / "build"
    
    try:
        # Configure with CMake
        subprocess.check_call([
            "cmake", "-B", str(build_dir), 
            "-DCMAKE_BUILD_TYPE=Release",
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=ON"
        ], cwd=whisper_cpp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Build
        subprocess.check_call([
            "cmake", "--build", str(build_dir), 
            "-j", "--config", "Release"
        ], cwd=whisper_cpp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check if whisper-cli was built
        whisper_cli_paths = [
            build_dir / "bin" / "whisper-cli",
            build_dir / "whisper-cli",
            build_dir / "examples" / "cli" / "whisper-cli"
        ]
        
        whisper_cli_built = any(path.exists() for path in whisper_cli_paths)
        
        if whisper_cli_built:
            print("✓ Successfully built whisper-cli")
        else:
            print("⚠ whisper-cli build completed but binary not found")
            
    except subprocess.CalledProcessError as e:
        print(f"⚠ Failed to build whisper-cli: {e}")
        print("You can build it manually after installation:")
        print("  cd whisper.cpp && cmake -B build && cmake --build build -j --config Release")
        return False
    except FileNotFoundError:
        print("❌ CMake not found. Please install CMake to build whisper-cli")
        print("You can build it manually after installation:")
        print("  cd whisper.cpp && cmake -B build && cmake --build build -j --config Release")
        return False
    
    print("🎉 Setup complete!")
    print(f"\nwhisper.cpp installed to: {whisper_cpp_path}")
    print("\nNext steps:")
    print(f"1. Download a model: cd {whisper_cpp_path} && sh ./models/download-ggml-model.sh base.en")
    print("2. Test: whispy transcribe audio.wav")
    print("="*60 + "\n")
    
    return True

class PostInstallCommand(install):
    """Post-installation command to set up whisper.cpp and build whisper-cli"""
    
    def run(self):
        # Run the normal installation first
        install.run(self)
        
        # Then setup whisper.cpp
        setup_whisper_cpp()

class PostDevelopCommand(develop):
    """Post-develop command to set up whisper.cpp and build whisper-cli"""
    
    def run(self):
        # Run the normal develop installation first
        develop.run(self)
        
        # Then setup whisper.cpp
        setup_whisper_cpp()

# The main setup configuration
setup(
    name="whispy",
    packages=["whispy"],
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
    zip_safe=True,
) 