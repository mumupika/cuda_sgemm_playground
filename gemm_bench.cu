/**
 * @file gemm_bench.cu
 * @author mumupika
 * @brief This is for the benchmark.
 * @version 0.1
 * @date 2026-03-22
 *
 * @copyright Copyright (c) 2026
 *
 */

/// CUDA RUNTIME.
#include "cuda_runtime.h"

#include "helper.h"
#include "tools.h"
#include "running.h"
#include <string>

template <size_t i>
std::string get_kernel_name() {
    if constexpr (i == 0) {
        return "cublas";
    } else if constexpr (i == 1) {
        return "cutlass";
    } else {
        return "kernel " + std::to_string(i - 1);
    }
}

template <int KernelId>
auto getKernel() {
    if constexpr (KernelId == 0) {
        return run_cublas;
    } else if constexpr (KernelId == 1) {
        return run_cutlass;
    } else if constexpr (KernelId >= 2 && KernelId < KERNEL_NUMS + 2) {
        return run_kernel<KernelId - 1>;
    }
}

template <int KernelId>
void benchKernel(
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
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));

    auto kernel = getKernel<KernelId>();

    printf("================================================================\n");
    printf("This is the first warm up.\n");
    printf("================================================================\n");
    kernel(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
    printf("================================================================\n");
    cudaDeviceSynchronize();

    /// Now started real bench.
    float time;
    double elapsed_milis = 0.0;
    for (int i = 0; i < 10; i++) {
        time = kernel(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta, check_result_flag);
        elapsed_milis += time;
        cudaDeviceSynchronize();
    }
    elapsed_milis /= 10;
    long long total_flop_cnt = 2LL * M * N * K;
    double calc = total_flop_cnt / elapsed_milis * (int)(1e3) / (int)(1e9);
    printf("%s average elapsed time: %f ms, Calculate capability: %lf GFlops/s.\n", name, elapsed_milis, calc);

    /// Cudafree hA, hB, hC.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

template <size_t... Is>
void call_bench(
    int M, int N, int K,
    float *hA, float *hB, float *hC,
    float const alpha, float const beta,
    bool check_result_flag,
    std::index_sequence<Is...>) {
    (benchKernel<Is>(M, N, K, hA, hB, hC, alpha, beta, get_kernel_name<Is>().c_str(), check_result_flag), ...);
}

int main() {
    int M = 512, N = 512, K = 512;
    bool check_result_flag = false;
    for (int i = 0; i < 5; i++) {
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

        printf("M = %d, K = %d, N = %d Kernel bench:\n", M, K, N);

        call_bench(M, N, K, hA, hB, hC, alpha, beta, check_result_flag, std::make_index_sequence<KERNEL_NUMS + 2>{});

        /// free hA, hB, hC.
        std::free(hA);
        std::free(hB);
        std::free(hC);

        M *= 2;
        N *= 2;
        K *= 2;
        printf("================================================================\n\n\n");
    }

    return 0;
}