#pragma once

/// CUDA RUNTIME.
#include "cuda_runtime.h"

/// C headers.
#include <stdio.h>

/// C++ headers.
#include <random>
#include <cstdlib>

#include "helper.h"

/**
 * @brief Get the device properties.
 *
 */
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
    printf("=====================================================================\n");
    std::free(prop);
}

/**
 * @brief Prepare datas for copy to device and sgemm.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA Pointer to matrix A on the host.
 * @param hB Pointer to matrix B on the host.
 * @param hC Pointer to matrix C on the host.
 * @param dA Pointer to matrix A on the device.
 * @param dB Pointer to matrix B on the device.
 * @param dC Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 */
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