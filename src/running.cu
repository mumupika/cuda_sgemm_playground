#include "checker.h"
#include "launcher.h"
#include "running.h"
#include "tools.h"
#include "launcher_templates.h"

template <>
float run_kernel<1>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    /// Create blocks and grids to map the datas for calculation.
    constexpr int blockSize = 32;
    dim3 gridDim(CEIL_DIV(M, blockSize), CEIL_DIV(N, blockSize), 1);
    dim3 blockDim(blockSize, blockSize, 1);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_naive(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim);
    time.stop();

    printf("Kernel 1: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cublas result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<2>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    /// Create blocks and grids to map the datas for calculation.
    constexpr int blockSize = 32;
    dim3 gridDim(CEIL_DIV(N, blockSize), CEIL_DIV(M, blockSize), 1);
    dim3 blockDim(blockSize, blockSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_coalescing(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim);
    time.stop();

    printf("Kernel 2: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<3>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    /// Create blocks and grids to map the datas for calculation.
    constexpr int blockSize = 32;
    dim3 gridDim(CEIL_DIV(M, blockSize), CEIL_DIV(N, blockSize), 1);
    dim3 blockDim(blockSize * blockSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    launch_sgemm_coalescing2(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, blockSize);
    time.stop();

    printf("Kernel 3: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<4>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    /// Create blocks and grids to map the datas for calculation.
    constexpr int tileSize = 32;
    dim3 gridDim(CEIL_DIV(N, tileSize), CEIL_DIV(M, tileSize), 1);
    dim3 blockDim(tileSize, tileSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(float);
    launch_sgemm_smem(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, sharedMemSize);
    time.stop();

    printf("Kernel 4: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<5>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    /// Create blocks and grids to map the datas for calculation.
    constexpr int tileSize = 32;
    dim3 gridDim(CEIL_DIV(N, tileSize), CEIL_DIV(M, tileSize), 1);
    dim3 blockDim(tileSize, tileSize);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(float);
    launch_sgemm_smem_opt(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, tileSize, sharedMemSize);
    time.stop();

    printf("Kernel 5: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<6>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockDim(BN / TN, BM / TM);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    size_t sharedMemSize = sizeof(float) * (BM * BK + BK * BN);
    launch_sgemm_reg_blocking<BM, BN, BK, TM, TN>(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, sharedMemSize);
    time.stop();

    printf("Kernel 6: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<7>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockDim(BN / TN, BM / TM);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    size_t sharedMemSize = sizeof(float) * (BM * (BK + 1) + BK * (BN + 1));
    launch_sgemm_reg_block_opt<BM, BN, BK, TM, TN>(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, sharedMemSize);
    time.stop();

    printf("Kernel 7: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

template <>
float run_kernel<8>(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

    /// Create blocks and grids to map the datas for calculation.
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockDim(BN / TN, BM / TM);

    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    size_t sharedMemSize = sizeof(float) * (BM * (BK + 1) + BK * (BN + 1));
    launch_sgemm_vec_load<BM, BN, BK, TM, TN>(M, N, K, ldA, ldB, ldC, dA, dB, dC, alpha, beta, gridDim, blockDim, sharedMemSize);
    time.stop();

    printf("Kernel 8: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        check_cublas_result(M, N, K, hA, hB, hC, dC, alpha, beta);
        printf("==========================================================\n");
    }
    return time.elapsed_millis();
}

float run_cutlass(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);
    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    CutlassSgemmNN(M, N, K, ldA, ldB, ldC, alpha, dA, dB, beta, dC);
    time.stop();

    printf("Cutlass basic example: GPU executed elapsed: %f ms\n", time.elapsed_millis());
    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        float *reference = static_cast<float *>(std::malloc(sizeof(float) * M * N));
        get_cpu_result(M, N, K, hA, hB, hC, reference, alpha, beta);
        check_cpu_result(M, N, ldC, reference, dC);
        printf("==========================================================\n");
        std::free(reference);
    }
    return time.elapsed_millis();
}

float run_cublas(
    int const M, int const N, int const K,
    int const ldA, int const ldB, int const ldC,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag) {
    memHtoD(M, N, K, ldA, ldB, ldC, hA, hB, hC, dA, dB, dC);
    /// launch the kernel from launcher.
    GpuTimer time{};
    time.start();
    CublasLauncher(M, N, K, ldA, ldB, ldC, alpha, dA, dB, beta, dC);
    time.stop();

    printf("Cublas basic example: GPU executed elapsed: %f ms\n", time.elapsed_millis());

    /// Check the data's correctivity.
    if (check_result_flag) {
        printf("==========================================================\n");
        printf("Check with cpu result enabled. Checking...\n");
        float *reference = static_cast<float *>(std::malloc(sizeof(float) * M * N));
        get_cpu_result(M, N, K, hA, hB, hC, reference, alpha, beta);
        check_cpu_result(M, N, ldC, reference, dC);
        printf("==========================================================\n");
        std::free(reference);
    }
    return time.elapsed_millis();
}