#pragma once

/// CUDA RUNTIME.
#include "cuda_runtime.h"

/// C headers.
#include <stdio.h>

/// C++ headers.
#include <cstdlib>

#include "helper.h"

/**
 * @brief Check for the data's copy is completed.
 *
 * @param hA host data A.
 * @param hB host data B.
 * @param hC host data C.
 * @param dA device data A.
 * @param dB device data B.
 * @param dC device data C.
 */
void check_data(
    const float *hA, const float *hB, const float *hC, // host.
    const float *dA, const float *dB, const float *dC  // device.
);

/**
 * @brief Check for the Calculation correctivity by CPU brute force.
 *
 * @param M size of A is (M, K), B is (K, N), C is (M, N).
 * @param N size of A is (M, K), B is (K, N), C is (M, N).
 * @param hA Host data A.
 * @param hB Host data B.
 * @param hC Host data C.
 * @param reference Host reference data C.
 * @param dC Device Data C.
 * @param alpha alpha.
 * @param beta beta.
 */
void check_cpu_result(
    int const M, int const N, // Dimensions;
    float const *reference,   // Host reference data;
    float const *dC           // Device Data;
);

/**
 * @brief Implemented in src/cutlass.cu.
 *
 * @param M size of A is (M, K), B is (K, N), C is (M, N).
 * @param N size of A is (M, K), B is (K, N), C is (M, N).
 * @param K size of A is (M, K), B is (K, N), C is (M, N).
 * @param alpha Scalar alpha.
 * @param A device data A.
 * @param B device data B.
 * @param beta Scalar beta.
 * @param C device data C.
 * @return cudaError_t
 */
cudaError_t CutlassSgemmNN(
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C);

/**
 * @brief Check the correctivity with the cutlass result.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA host data A.
 * @param hB host data B.
 * @param hC host data C.
 * @param old_dC the previous kernel calculated result.
 * @param reference the CPU reference result.
 * @param alpha Scalar bias alpha.
 * @param beta Scalar bias beta.
 */
void check_cutlass_result(
    int const M, int const N, int const K,
    float const *hA, float const *hB, float const *hC, float const *old_dC,
    float const alpha, float const beta);

/**
 * @brief Check the correctivity with the cublas result.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA host data A.
 * @param hB host data B.
 * @param hC host data C.
 * @param old_dC the previous kernel calculated result.
 * @param reference the CPU reference result.
 * @param alpha Scalar bias alpha.
 * @param beta Scalar bias beta.
 */
void check_cublas_result(
    int const M, int const N, int const K,
    float const *hA, float const *hB, float const *hC, float const *old_dC,
    float const alpha, float const beta);