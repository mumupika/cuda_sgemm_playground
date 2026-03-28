#include "checker.h"
#include "launcher.h"

void check_data(
    const float *hA, const float *hB, const float *hC, // host.
    const float *dA, const float *dB, const float *dC  // device.
) {
    /// Check the data.
    /// Malloc chA, chB, chC.
    float *chA, *chB, *chC;
    chA = static_cast<float *>(std::malloc(sizeof(float) * 10));
    chB = static_cast<float *>(std::malloc(sizeof(float) * 10));
    chC = static_cast<float *>(std::malloc(sizeof(float) * 10));
    CUDA_CHECK(cudaMemcpy(chA, dA, sizeof(float) * 10, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(chB, dB, sizeof(float) * 10, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(chC, dC, sizeof(float) * 10, cudaMemcpyDeviceToHost));

    printf("hA: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hA[i]);
    }
    printf(" ...\n");
    printf("dA: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", chA[i]);
    }
    printf(" ...\n");
    printf("hB: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hB[i]);
    }
    printf(" ...\n");
    printf("dB: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", chB[i]);
    }
    printf(" ...\n");
    printf("hC: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hC[i]);
    }
    printf(" ...\n");
    printf("dC: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", chC[i]);
    }
    printf(" ...\n");

    /// Free them after out of scope.
    std::free(chA);
    std::free(chB);
    std::free(chC);
}

void check_cpu_result(
    int const M, int const N, // Dimensions;
    int const ldC,            // C's leading dim;
    float const *reference,   // Host reference data;
    float const *dC           // Device Data;
) {
    float *result = static_cast<float *>(std::malloc(sizeof(float) * M * N));

    /// copy the data back.
    // CUDA_CHECK(cudaMemcpy(result, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(result, N * sizeof(float), dC, ldC * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));

    /// Check parity.
    bool pass = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (static_cast<float>(std::abs(reference[i * N + j] - result[i * N + j])) > 5e-3) {
                printf("Result error at (%d, %d):\t\n", i, j);
                printf("Expected result: %f\n", reference[i * N + j]);
                printf("Got: %f\n", result[i * N + j]);
                printf("Miss: %lf\n", static_cast<float>(std::abs(reference[i * N + j] - result[i * N + j])));
                pass = false;
            }
        }
    }

    std::free(result);
    if (pass == true) {
        printf("Congratulations, passed!\n");
    } else {
        printf("Failed to pass!\n");
    }
}

void check_cutlass_result(
    int const M, int const N, int const K,
    float const *hA, float const *hB, float const *hC, float const *old_dC,
    float const alpha, float const beta) {
    /// device data malloc.
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

    /// copy datas.
    /// Now copy the data from host to GPU.
    CUDA_CHECK(cudaMemcpy2D(dA, ldA * sizeof(float), hA, K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dB, ldB * sizeof(float), hB, N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dC, ldC * sizeof(float), hC, N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice));

    /// Real cutlass calculation.
    CUDA_CHECK(CutlassSgemmNN(M, N, K, ldA, ldB, ldC, alpha, dA, dB, beta, dC));

    // host_data malloc.
    float *cutlass_hC = static_cast<float *>(std::malloc(sizeof(float) * M * N));
    float *kernel_hC = static_cast<float *>(std::malloc(sizeof(float) * M * N));

    CUDA_CHECK(cudaMemcpy2D(cutlass_hC, N * sizeof(float), dC, ldC * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(kernel_hC, N * sizeof(float), old_dC, ldC * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));

    /// Check parity with GPU result.
    bool pass = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (static_cast<float>(std::abs(cutlass_hC[i * N + j] - kernel_hC[i * N + j])) > 5e-3) {
                printf("Result error at (%d, %d):\t\n", i, j);
                printf("cutlass result: %f\n", cutlass_hC[i * N + j]);
                printf("kernel: %f\n", kernel_hC[i * N + j]);
                printf("Miss: %lf\n", static_cast<float>(std::abs(cutlass_hC[i * N + j] - kernel_hC[i * N + j])));
                pass = false;
            }
        }
    }

    // Free host data.
    std::free(cutlass_hC);
    std::free(kernel_hC);

    // Free device data.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (pass == true) {
        printf("Congratulations, passed!\n");
    } else {
        printf("Failed to pass!\n");
    }
}

void check_cublas_result(
    int const M, int const N, int const K,
    float const *hA, float const *hB, float const *hC, float const *old_dC,
    float const alpha, float const beta) {
    /// device data malloc.
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

    /// copy datas.
    /// Now copy the data from host to GPU.
    CUDA_CHECK(cudaMemcpy2D(dA, ldA * sizeof(float), hA, K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dB, ldB * sizeof(float), hB, N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dC, ldC * sizeof(float), hC, N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice));

    /// Real cutlass calculation.
    CUBLAS_CHECK(CublasLauncher(M, N, K, ldA, ldB, ldC, alpha, dA, dB, beta, dC));

    // host_data malloc.
    float *cublas_hC = static_cast<float *>(std::malloc(sizeof(float) * M * N));
    float *kernel_hC = static_cast<float *>(std::malloc(sizeof(float) * M * N));

    CUDA_CHECK(cudaMemcpy2D(cublas_hC, N * sizeof(float), dC, ldC * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(kernel_hC, N * sizeof(float), old_dC, ldC * sizeof(float), N * sizeof(float), M, cudaMemcpyDeviceToHost));

    /// Check parity with GPU result.
    bool pass = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (static_cast<float>(std::abs(cublas_hC[i * N + j] - kernel_hC[i * N + j])) > 5e-3) {
                printf("Result error at (%d, %d):\t\n", i, j);
                printf("cublas result: %f\n", cublas_hC[i * N + j]);
                printf("kernel: %f\n", kernel_hC[i * N + j]);
                printf("Miss: %lf\n", static_cast<float>(std::abs(cublas_hC[i * N + j] - kernel_hC[i * N + j])));
                pass = false;
                goto finalize;
            }
        }
    }

finalize:
    // Free host data.
    std::free(cublas_hC);
    std::free(kernel_hC);

    // Free device data.
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (pass == true) {
        printf("Congratulations, passed!\n");
    } else {
        printf("Failed to pass!\n");
    }
}