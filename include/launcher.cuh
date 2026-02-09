#pragma once

/**
 * @brief The launcher of the kernel.
 * 
 * @param sharedMemSize 
 * @param stream 
 */
void launch_sgemm_naive(
    int M, int N, int K,
    const float *A, const float *B, float *C, 
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0
);