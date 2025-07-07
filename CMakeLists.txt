cmake_minimum_required(VERSION 3.10)
project(whispy)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build shared library by default
option(BUILD_SHARED_LIBS "Build shared library" ON)

# Disable problematic backends to avoid read-only file system issues
set(GGML_METAL OFF CACHE BOOL "Disable Metal backend" FORCE)
set(GGML_CUDA OFF CACHE BOOL "Disable CUDA backend" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "Disable OpenCL backend" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan backend" FORCE)

# Disable examples and tests to speed up build
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "Disable whisper examples" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "Disable whisper tests" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "Disable whisper server" FORCE)

# Set additional flags for compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

# Add the complete whisper.cpp directory
add_subdirectory(whisper.cpp)

# Install rules
install(TARGETS whisper
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 