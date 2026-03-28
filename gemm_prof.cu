/**
 * @file gemm.cu
 * @author mumupika
 * @brief This is the main execution point. This is used for profile.
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
#include "gemm_utils.cuh"

template <int KernelId>
void testKernel(
    int M, int N, int K,
    float *hA, float *hB, float *hC,
    float const alpha, float const beta,
    char const *name,
    bool check_result_flag) {
    /// device data.
    float *dA;
    float *dB;
    float *dC;

    /// Here is the GPU take in. cudaMalloc dA, dB, dC in `prepare_matrix`.
    size_t pitchA;
    size_t pitchB;
    size_t pitchC;
    CUDA_CHECK(cudaMallocPitch(&dA, &pitchA, sizeof(float) * K, M));
    CUDA_CHECK(cudaMallocPitch(&dB, &pitchB, sizeof(float) * N, K));
    CUDA_CHECK(cudaMallocPitch(&dC, &pitchC, sizeof(float) * N, M));
    int ldA = pitchA / sizeof(float);
    int ldB = pitchB / sizeof(float);
    int ldC = pitchC / sizeof(float);

    auto kernel = getKernel<KernelId>();
    printf("================================================================\n");
    printf("Started test %s.\n", name);
    kernel(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    printf("================================================================\n");

    /// Cudafree hA, hB, hC.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main() {
    /// Matrix Dimension.
    /// Which means: A (M, K) @ B (K, N) * alpha + beta * C (M, N);
    int M = 4096;
    int N = 4096;
    int K = 4096;
    bool check_result_flag = false;

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

    /// Bias.
    float alpha;
    float beta;

    prepare_matrix(M, N, K, hA, hB, hC, alpha, beta);

    printf("M = %d, K = %d, N = %d Kernel test:\n", M, K, N);

    constexpr int kernelId = 9;
    testKernel<kernelId>(M, N, K, hA, hB, hC, alpha, beta, get_kernel_name<kernelId>().c_str(), check_result_flag);

    /// free hA, hB, hC.
    std::free(hA);
    std::free(hB);
    std::free(hC);
    return 0;
}