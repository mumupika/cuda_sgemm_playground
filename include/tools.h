#pragma once

/// CUDA RUNTIME.
#include "cuda_runtime.h"

/// C headers.
#include <cstdio>
#include <stdio.h>

/// C++ headers.
#include <random>
#include <cstdlib>

#include "helper.h"

void GetProperties() {
    printf("============================= Properties =============================\n");
    cudaDeviceProp *prop = static_cast<cudaDeviceProp *>(std::malloc(sizeof(cudaDeviceProp)));
    cudaGetDeviceProperties(prop, 0);
    printf("Device: %s\n", prop -> name);
    printf("Total global memory: %ld Bytes\n", prop -> totalGlobalMem);
    printf("Total Constant memory: %ld Bytes\n", prop -> totalConstMem);
    printf("Shared mem per Block: %ld Bytes\n", prop -> sharedMemPerBlock);
    printf("Regs per block: %d\n", prop -> regsPerBlock);
    printf("WarpSize: %d\n", prop -> warpSize);
    printf("maxThreadsPerBlock: %d\n", prop -> maxThreadsPerBlock);
    printf("maxThreadsDim: (%d, %d, %d)\n", prop -> maxThreadsDim[0], prop -> maxThreadsDim[1], prop -> maxThreadsDim[2]);
    printf("maxGridSize: (%d, %d, %d)\n", prop -> maxGridSize[0], prop -> maxGridSize[1], prop -> maxGridSize[2]);
    printf("max Concurrent kers: %d\n", prop -> concurrentKernels);
    printf("async engine cnt: %d\n", prop -> asyncEngineCount);
    printf("=====================================================================\n");
    std::free(prop);
}

void prepare_matrix(
    int M, int N, int K,                    // Dimensions;
    float *hA, float *hB, float *hC,        // Host data;
    float *dA, float *dB, float *dC,     // Device Data;
    float &alpha, float &beta               // bias.
) {
    /// Prepare the data. generate uniformed float.
    std::random_device r;
    std::mt19937_64 e(r());
    std::uniform_real_distribution<float> uniform_dist(-9.0, 9.0);
    for (size_t i = 0; i < M * N; i++) {
        hA[i] = uniform_dist(e);
        hB[i] = uniform_dist(e);
        hC[i] = uniform_dist(e);
    }
    alpha = uniform_dist(e);
    beta = uniform_dist(e);

    /// Now copy the data from host to GPU.
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(float) * N * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeof(float) * M * K, cudaMemcpyHostToDevice));
}

void check_data(
    const float *hA, const float *hB, const float *hC,      // host.
    const float *dA, const float *dB, const float *dC       // device.
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

void check_result(
    int const M, int const N, int const K,                    // Dimensions;
    float const *hA, float const *hB, float const *hC,        // Host data;
    float const *dC,                                          // Device Data;
    float const alpha, float const beta                       // bias.
) {
    float *result = static_cast<float *>(std::malloc(sizeof(float) * M * N));

    /// copy the data back.
    CUDA_CHECK(cudaMemcpy(result, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    /// CPU calculation.
    float *reference = static_cast<float *>(std::malloc(sizeof(float) * M * N));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += hA[i * K + k] * hB[j + k * N];
            }
            reference[i * N + j] = alpha * sum + beta * hC[i * N + j];
        }
    }

    /// Check parity.
    bool pass = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (static_cast<double>(std::abs(reference[i * N + j] - result[i * N + j]) / std::abs(reference[i * N + j])) > 1e-3) {
                printf("Result error at (%d, %d):\t\n", i, j);
                printf("Expected result: %f\n", reference[i * N + j]);
                printf("Got: %f\n", result[i * N + j]);
                printf("Miss: %lf\n", static_cast<double>(std::abs(reference[i * N + j] - result[i * N + j]) / std::abs(reference[i * N + j])));
                pass = false;
                goto error;
            }
        }
    }

error:
    std::free(result);
    std::free(reference);
    if (pass == true) {
        printf("Congratulations, passed!\n");
    } else {
        printf("Failed to pass!\n");
    }
}