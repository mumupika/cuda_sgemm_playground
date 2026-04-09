#pragma once

/// CUDA RUNTIME.
#include "cuda_runtime.h"
#include "cublas_v2.h"

/// C headers.
#include <stdio.h>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                \
    {                                                        \
        cutlass::Status error = status;                      \
        if (error != cutlass::Status::kSuccess) {            \
            printf("Got cutlass error: %s at line: %d\n",    \
                   cutlassGetStatusString(error), __LINE__); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                  \
    {                                                       \
        cudaError_t error = status;                         \
        if (error != cudaSuccess) {                         \
            printf("Got bad cuda status: %s at line: %d\n", \
                   cudaGetErrorString(error), __LINE__);    \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }

/**
 * Panic Wrapper for unwinding CUBLAS runtime errors
 */
#define CUBLAS_CHECK(status)                                  \
    {                                                         \
        cublasStatus_t error = status;                        \
        if (error != CUBLAS_STATUS_SUCCESS) {                 \
            printf("Got bad cublas status: %s at line: %d\n", \
                   cublasGetStatusName(error), __LINE__);     \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    }

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer {
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() :
        _stream_id(0) {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0) {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop() {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis() {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/**
 * @brief Self defined functions as constexpr to avoid warning.
 * 
 */
namespace helper {
    template<typename T>
    __device__ constexpr auto max(T a, T b) {
        return (a < b ? b : a);
    }
    template<typename T>
    __device__ constexpr auto min(T a, T b) {
        return (a < b ? a : b);
    }
}

/**
 * @brief Memory Swizzle for the Bank conflict free implementation.
 * 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY mask
 *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 * 
 * Adapted from https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp#L43-L53
 *
 * @tparam B BBits is the number of bits in the mask
 * @tparam M MBase is the number of least-sig bits to keep constant
 * @tparam S SShift is the distance to shift the YYY mask (pos shifts YYY to the right, neg shifts YYY to the left)
 */
template <int B, int M, int S>
struct Swizzle {
    static constexpr int BBITS = B;
    static constexpr int MBASE = M;
    static constexpr int SSHIFT = S;
    static constexpr int BIT_MASK = (1 << BBITS) - 1;
    static constexpr int YYY_MASK = BIT_MASK << (MBASE + helper::max(0, SSHIFT));
    static constexpr int ZZZ_MASK = BIT_MASK << (MBASE - helper::min(0, SSHIFT));
    template <int b>
    __device__ constexpr int SHIFT_RIGHT(const int a) {
        if constexpr (b >= 0) {
            return a >> b;
        } else {
            return a << (-b);
        }
    }
    __device__ constexpr int GET_SWZ(int offset) {
        return offset ^ SHIFT_RIGHT<SSHIFT>((offset & YYY_MASK));
    }
};

/**
 * @brief Get the 2d offset with the width as matrix width.
 */
template <int Width>
__device__ int get_2d_offset(int col, int row) {
    return col * Width + row;
}