import os
import pathlib
import subprocess
import sys
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

class CMakeBuild:
    """Helper class to build CMake projects"""
    
    def __init__(self, sourcedir="whisper.cpp"):
        self.sourcedir = pathlib.Path(sourcedir).resolve()
    
    def build(self, build_temp, build_lib):
        """Build the CMake project"""
        build_temp = pathlib.Path(build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp.absolute()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_METAL=OFF",
            "-DGGML_CUDA=OFF",
            "-DGGML_OPENCL=OFF",
            "-DGGML_VULKAN=OFF",
        ]

        # Platform-specific settings
        if sys.platform == "darwin":
            cmake_args.append("-DCMAKE_MACOSX_RPATH=ON")
        elif sys.platform == "win32":
            cmake_args.append("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=" + str(build_temp.absolute()))

        # Set the build type
        cfg = "Release"
        build_args = ["--config", cfg]
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        # Run CMake
        subprocess.check_call(["cmake", str(self.sourcedir)] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

        # Copy libraries to the package directory
        build_lib_path = pathlib.Path(build_lib) / "whispy"
        build_lib_path.mkdir(parents=True, exist_ok=True)
        
        # Look for the library files
        lib_patterns = ["libwhisper*.so", "libwhisper*.dylib", "whisper*.dll"]
        library_found = False
        
        for pattern in lib_patterns:
            for lib in build_temp.glob(pattern):
                dest_path = build_lib_path / lib.name
                shutil.copy2(str(lib), str(dest_path))
                library_found = True
                print(f"Copied {lib.name} to {dest_path}")
        
        if not library_found:
            # Also check subdirectories
            for pattern in lib_patterns:
                for lib in build_temp.rglob(pattern):
                    dest_path = build_lib_path / lib.name
                    shutil.copy2(str(lib), str(dest_path))
                    library_found = True
                    print(f"Copied {lib.name} to {dest_path}")
        
        if not library_found:
            raise RuntimeError("Could not find compiled whisper library")

class CustomBuildPy(build_py):
    """Custom build_py command that builds CMake project"""
    
    def run(self):
        # Build the CMake project
        cmake_build = CMakeBuild()
        cmake_build.build(self.build_temp, self.build_lib)
        
        # Run the normal build_py
        super().run()

class CustomDevelop(develop):
    """Custom develop command that builds CMake project"""
    
    def run(self):
        # Build the CMake project
        cmake_build = CMakeBuild()
        cmake_build.build("build/temp.cmake", "build/lib")
        
        # Run the normal develop
        super().run()

# The main setup configuration
setup(
    name="whispy",
    packages=["whispy"],
    cmdclass={
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
    },
    zip_safe=False,
) 