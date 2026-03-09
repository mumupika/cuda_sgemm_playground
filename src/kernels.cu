/**
 * @file kernels.cu
 * @author mumupika
 * @brief sgemm implementations.
 * @version 0.1
 * @date 2026-02-09
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cuda_runtime.h"

#include "launcher.h"
#include "helper.h"

/**
 * @brief The naive Gemm implementation.
 * @brief sgemm for Matrix A(M, K), B(K, N), C(M, N) has alpha * A @ B + beta * C for calculation.
 */
__global__ void sgemm_naive(
    int const M, int const N, int const K,
    float const alpha,
    float const *A, float const *B,
    float const beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void launch_sgemm_naive(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize, cudaStream_t stream) {
    /// Get the kernel.
    sgemm_naive<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

/**
 * @brief The Gemm implementation with thread access coalescing.
 * @brief sgemm for Matrix A(M, K), B(K, N), C(M, N) has alpha * A @ B + beta * C for calculation.
 */
__global__ void sgemm_coalescing(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C,
    int blockSize) {
    const int x = blockIdx.x * blockSize + (threadIdx.x / blockSize);
    const int y = blockIdx.y * blockSize + (threadIdx.x % blockSize);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void launch_sgemm_coalescing(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim, int const blockSize,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_coalescing<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C, blockSize);
}