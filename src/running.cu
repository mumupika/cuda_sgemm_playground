#include "checker.h"
#include "launcher.h"
#include "running.h"
#include "tools.h"

void run_kernel1(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag) {
    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    if (check_result_flag) {
        check_data(hA, hB, hC, dA, dB, dC);
    }

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_naive(M, N, K, dA, dB, dC, alpha, beta, gridDim, blockDim);
    time.stop();

    printf("Kernel 1: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
}

void run_kernel2(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag) {
    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    if (check_result_flag) {
        check_data(hA, hB, hC, dA, dB, dC);
    }

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    int const blockSize = 32;
    dim3 blockDim(blockSize * blockSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_coalescing(M, N, K, dA, dB, dC, alpha, beta, gridDim, blockDim, blockSize);
    time.stop();

    printf("Kernel 2: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
}

void run_kernel3(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag) {
    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    if (check_result_flag) {
        check_data(hA, hB, hC, dA, dB, dC);
    }

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    int const tileSize = 32;
    dim3 blockDim(tileSize, tileSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_smem(M, N, K, dA, dB, dC, alpha, beta, gridDim, blockDim, tileSize);
    time.stop();

    printf("Kernel 3: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
}

void run_cutlass(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag) {
    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    if (check_result_flag) {
        check_data(hA, hB, hC, dA, dB, dC);
    }

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    CutlassSgemmNN(M, N, K, alpha, dA, dB, beta, dC);
    time.stop();

    printf("Cutlass basic example: GPU executed elapsed: %f ms\n", time.elapsed_millis());
    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        float *reference = static_cast<float *>(std::malloc(sizeof(float) * M * N));
        get_cpu_result(M, N, K, hA, hB, hC, reference, alpha, beta);
        check_cpu_result(M, N, reference, dC);
        printf("==========================================================\n");
        std::free(reference);
    }
}

void run_cublas(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag) {
    prepare_matrix(M, N, K, hA, hB, hC, dA, dB, dC, alpha, beta);
    if (check_result_flag) {
        check_data(hA, hB, hC, dA, dB, dC);
    }

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    CublasLauncher(M, N, K, alpha, dA, dB, beta, dC);
    time.stop();

    printf("Cublas basic example: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        float *reference = static_cast<float *>(std::malloc(sizeof(float) * M * N));
        get_cpu_result(M, N, K, hA, hB, hC, reference, alpha, beta);
        check_cpu_result(M, N, reference, dC);
        printf("==========================================================\n");
        std::free(reference);
    }
}