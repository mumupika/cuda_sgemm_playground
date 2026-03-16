/// CUDA RUNTIME.
#include "cuda_runtime.h"

/// C headers.
#include <stdio.h>

/// C++ headers.
#include <random>
#include <cstdlib>
#include <chrono>

#include "helper.h"
#include "tools.h"

void GetProperties() {
    printf("============================= Properties =============================\n");
    cudaDeviceProp *prop = static_cast<cudaDeviceProp *>(std::malloc(sizeof(cudaDeviceProp)));
    cudaGetDeviceProperties(prop, 0);
    printf("Device: %s\n", prop->name);
    printf("Total global memory: %ld Bytes\n", prop->totalGlobalMem);
    printf("Total Constant memory: %ld Bytes\n", prop->totalConstMem);
    printf("Shared mem per Block: %ld Bytes\n", prop->sharedMemPerBlock);
    printf("Regs per block: %d\n", prop->regsPerBlock);
    printf("WarpSize: %d\n", prop->warpSize);
    printf("maxThreadsPerBlock: %d\n", prop->maxThreadsPerBlock);
    printf("maxThreadsDim: (%d, %d, %d)\n", prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
    printf("maxGridSize: (%d, %d, %d)\n", prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
    printf("max Concurrent kers: %d\n", prop->concurrentKernels);
    printf("async engine cnt: %d\n", prop->asyncEngineCount);
    printf("=====================================================================\n\n\n\n");
    std::free(prop);
}

void prepare_matrix(
    int M, int N, int K,             // Dimensions;
    float *hA, float *hB, float *hC, // Host data;
    float *dA, float *dB, float *dC, // Device Data;
    float &alpha, float &beta        // bias.
) {
    /// Prepare the data. generate uniformed float.
    std::random_device r;
    std::mt19937_64 e(r());
    std::uniform_real_distribution<float> uniform_dist(-9.0, 9.0);
    for (size_t i = 0; i < M * K; i++) {
        hA[i] = uniform_dist(e);
    }
    for (size_t i = 0; i < K * N; i++) {
        hB[i] = uniform_dist(e);
    }
    for (size_t i = 0; i < M * N; i++) {
        hC[i] = uniform_dist(e);
    }
    alpha = uniform_dist(e);
    beta = uniform_dist(e);

    /// Now copy the data from host to GPU.
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice));
}

void get_cpu_result(
    int const M, int const N, int const K,             // Dimensions;
    float const *hA, float const *hB, float const *hC, // Host data;
    float *reference,                                  // Device Data;
    float const alpha, float const beta                // bias.
) {
    const auto start_time{std::chrono::steady_clock::now()};
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += hA[i * K + k] * hB[j + k * N];
            }
            reference[i * N + j] = alpha * sum + beta * hC[i * N + j];
        }
    }
    const auto end_time{std::chrono::steady_clock::now()};
    std::chrono::duration<double> duration = end_time - start_time;
    const auto elapse = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(duration);
    printf("CPU time elapsed: %lf ms\n", elapse.count());
}
