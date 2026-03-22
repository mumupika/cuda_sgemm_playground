cmake_minimum_required(VERSION 3.20)

# set clang21 for our project
set(CMAKE_PREFIX_PATH "/home/linuxbrew/.linuxbrew/opt/llvm@21")
set(CMAKE_EXE_LINKER_FLAGS "-L/home/linuxbrew/.linuxbrew/opt/llvm@21/lib")
set(CMAKE_CXX_FLAGS "-I/home/linuxbrew/.linuxbrew/opt/llvm@21/include")
set(CMAKE_C_COMPILER "/home/linuxbrew/.linuxbrew/opt/llvm@21/bin/clang")
set(CMAKE_CXX_COMPILER "/home/linuxbrew/.linuxbrew/opt/llvm@21/bin/clang++")

project(gemm LANGUAGES C CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Git REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library" FORCE)
set(CUTLASS_NVCC_ARCHS "89" CACHE STRING "The SM architectures requested.")
set(CUTLASS_CUDA_USE_RESPONSE_FILE OFF FORCE)

# set --options-file off.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# warn all.
add_compile_options(-Wall)

# FetchContent for the ceiling line of the GEMM.
include(FetchContent)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cutlass)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.3.5
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/cutlass
)
# automatically check whether the cutlass has been populated.
FetchContent_MakeAvailable(cutlass)

# Add include directories.
include_directories(include third-party/include)

# set CXX standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set CUDA standard.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# cuda architectures.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# get the executable file.
add_library(
    components SHARED
    src/running.cu
    src/kernels.cu 
    src/cutlass.cu
    src/tools.cu
    src/checker.cu
)
add_executable(gemm_test gemm_test.cu)
add_executable(gemm_bench gemm_bench.cu)

target_link_libraries(
    components
    CUTLASS
    cutlass_tools_util_includes
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)

target_link_libraries(gemm_test
    PRIVATE 
        components
)
target_link_libraries(gemm_bench
    PRIVATE 
        components
)

# Get the target compile options.
target_compile_options(components PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)