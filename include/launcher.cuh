#pragma once

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
    const float *A, const float *B, float *C, 
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0
);