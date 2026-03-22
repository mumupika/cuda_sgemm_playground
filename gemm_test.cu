/**
 * @file gemm.cu
 * @author mumupika
 * @brief This is the main execution point. This is used for checking the kernel's correctivity.
 *
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
#include "running.h"

int main() {

    /// Matrix Dimension.
    /// Which means: A (M, K) @ B (K, N) * alpha + beta * C (M, N);
    int M = 512;
    int N = 128;
    int K = 256;
    bool check_result_flag = true;

    /// Get the device properties.
    GetProperties();

    /// warm up the cuda.
    cudaFree(0);

    /// host data.
    float *hA;
    float *hB;
    float *hC;

    /// malloc hA, hB, hC.
    hA = static_cast<float *>(std::malloc(sizeof(float) * M * K));
    hB = static_cast<float *>(std::malloc(sizeof(float) * K * N));
    hC = static_cast<float *>(std::malloc(sizeof(float) * M * N));

    /// device data.
    float *dA;
    float *dB;
    float *dC;

    /// Here is the GPU take in. cudaMalloc dA, dB, dC in `prepare_matrix`.
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));

    /// Bias.
    float alpha;
    float beta;

    prepare_matrix(M, N, K, hA, hB, hC, alpha, beta);

    /// execute naive sgemm.
    run_kernel<1>(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_kernel<2>(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_kernel<3>(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_kernel<4>(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_kernel<5>(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_cutlass(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    run_cublas(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);

    /// free hA, hB, hC.
    std::free(hA);
    std::free(hB);
    std::free(hC);

    /// Cudafree hA, hB, hC.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}