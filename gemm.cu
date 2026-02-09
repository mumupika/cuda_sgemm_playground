/**
 * @file gemm.cu
 * @author mumupika
 * @brief This is the main execution point.
 * TODO: Add googleTest + googleBenchmark inside.
 *
 * @version 0.1
 * @date 2026-02-07
 *
 * @copyright Copyright (c) 2026
 *
 */

/// CUDA RUNTIME.
#include "cuda_runtime.h"

#include "helper.h"
#include "tools.h"
#include "launcher.cuh"

int main() {
    /// Get the device properties.
    GetProperties();

    /// Matrix Dimension.
    /// Which means: A (M, N) @ B (N, K) * alpha + beta * C (M, K);
    int M = 4096;
    int N = 4096;
    int K = 4096;

    /// host data.
    float *hA;
    float *hB;
    float *hC;

    /// malloc hA, hB, hC.
    hA = static_cast<float *>(std::malloc(sizeof(float) * M * N));
    hB = static_cast<float *>(std::malloc(sizeof(float) * N * K));
    hC = static_cast<float *>(std::malloc(sizeof(float) * M * K));

    /// device data.
    float *dA;
    float *dB;
    float *dC;

    /// Here is the GPU take in. cudaMalloc dA, dB, dC in `prepare_matrix`.
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * K));

    /// Bias.
    float alpha;
    float beta;

    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    check_data(hA, hB, hC, dA, dB, dC);

    /// free hA, hB, hC.
    std::free(hA);
    std::free(hB);
    std::free(hC);

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    /// launch the kernel from launcher.
    launch_sgemm_naive(M, N, K, dA, dB, dC, alpha, beta, gridDim, blockDim);

    /// Cudafree hA, hB, hC.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}