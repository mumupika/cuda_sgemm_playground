#pragma once

#include "cublas_v2.h"

/**
 * @brief The launcher of the naive sgemm kernel.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param C Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 * @param gridDim The grid dimensions for the kernel launch.
 * @param blockDim The block dimensions for the kernel launch.
 * @param sharedMemSize The amount of shared memory per block.
 * @param stream The CUDA stream for the kernel launch.
 */
void launch_sgemm_naive(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0);

/**
 * @brief The launcher of the naive sgemm kernel with coalescing deal.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param C Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 * @param gridDim The grid dimensions for the kernel launch.
 * @param blockDim The block dimensions for the kernel launch.
 * @param blockSize The blockSize for coalescing.
 * @param sharedMemSize The amount of shared memory per block.
 * @param stream The CUDA stream for the kernel launch.
 */
void launch_sgemm_coalescing(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0);

/**
 * @brief The launcher of the naive sgemm kernel with coalescing deal.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param C Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 * @param gridDim The grid dimensions for the kernel launch.
 * @param blockDim The block dimensions for the kernel launch.
 * @param blockSize The blockSize for coalescing.
 * @param sharedMemSize The amount of shared memory per block.
 * @param stream The CUDA stream for the kernel launch.
 */
void launch_sgemm_coalescing2(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim, int const blockSize,
    size_t sharedMemSize = 0, cudaStream_t stream = 0);

/**
 * @brief To launch a cublas kernel.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param alpha Scalar alpha.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param beta Scalar beta.
 * @param C Pointer to matrix C on the device.
 * @return cudaError_t
 */
cublasStatus_t CublasLauncher(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha, const float *A, const float *B,
    float beta, float *C);

/**
 * @brief The launcher of the naive sgemm kernel with shared memory tiling load.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param C Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 * @param gridDim The grid dimensions for the kernel launch.
 * @param blockDim The block dimensions for the kernel launch.
 * @param sharedMemSize The amount of shared memory per block.
 * @param stream The CUDA stream for the kernel launch.
 */
void launch_sgemm_smem(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0);

/**
 * @brief The launcher of shared memory tiling load with reducing bank conflict.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param A Pointer to matrix A on the device.
 * @param B Pointer to matrix B on the device.
 * @param C Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 * @param gridDim The grid dimensions for the kernel launch.
 * @param blockDim The block dimensions for the kernel launch.
 * @param tileSize The 2 * tileSize * tileSize for shared memory.
 * @param sharedMemSize The amount of shared memory per block.
 * @param stream The CUDA stream for the kernel launch.
 */
void launch_sgemm_smem_opt(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim, int const tileSize,
    size_t sharedMemSize = 0, cudaStream_t stream = 0);