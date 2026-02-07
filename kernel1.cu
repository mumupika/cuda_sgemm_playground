/**
 * @file kernel1.cu
 * @author mumupika
 * @brief Naive sgemm implementation.
 * @version 0.1
 * @date 2026-02-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "cuda_runtime.h"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}