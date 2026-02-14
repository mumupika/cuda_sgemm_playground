cmake_minimum_required(VERSION 3.20)

project(gemm LANGUAGES C CXX CUDA)

find_package(Git REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library" FORCE)
set(CUTLASS_NVCC_ARCHS "100" CACHE STRING "The SM architectures requested.")

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
set(CMAKE_CUDA_ARCHITECTURES 100)

# get the executable file.
add_executable(gemm gemm.cu src/kernels.cu src/cutlass.cu)
target_link_libraries(gemm 
    PRIVATE 
        CUTLASS
        cutlass_tools_util_includes
        cublas
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
)

# Get the target compile options.
target_compile_options(gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)