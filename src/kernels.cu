/**
 * @file kernels.cu
 * @author mumupika
 * @brief sgemm implementations.
 * @version 0.1
 * @date 2026-02-09
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cuda_runtime.h"

#include "launcher.h"
#include "helper.h"
#include "cublas_v2.h"

/**
 * @brief The naive Gemm implementation.
 * @brief sgemm for Matrix A(M, K), B(K, N), C(M, N) has alpha * A @ B + beta * C for calculation.
 */
__global__ void sgemm_naive(
    int const M, int const N, int const K,
    float const alpha,
    float const *A, float const *B,
    float const beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void launch_sgemm_naive(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize, cudaStream_t stream) {
    /// Get the kernel.
    sgemm_naive<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

/**
 * @brief The Gemm implementation with global memory coalescing.
 * @brief sgemm for Matrix A(M, K), B(K, N), C(M, N) has alpha * A @ B + beta * C for calculation.
 */
__global__ void sgemm_coalescing(
    int const M, int const N, int const K,
    float const alpha,
    float const *A, float const *B,
    float const beta, float *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

void launch_sgemm_coalescing(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_coalescing<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

/**
 * @brief The Gemm implementation with thread access coalescing.
 * @brief sgemm for Matrix A(M, K), B(K, N), C(M, N) has alpha * A @ B + beta * C for calculation.
 */
__global__ void sgemm_coalescing2(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C,
    int blockSize) {
    const int x = blockIdx.x * blockSize + (threadIdx.x / blockSize);
    const int y = blockIdx.y * blockSize + (threadIdx.x % blockSize);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = \alpha * (A @ B) + \beta * C;
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void launch_sgemm_coalescing2(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim, int const blockSize,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_coalescing2<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C, blockSize);
}

__global__ void sgemm_smem(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    // Statically assigned 2 dim shared memory.
    constexpr int tileSize = 32;
    __shared__ float As[tileSize][tileSize];
    __shared__ float Bs[tileSize][tileSize];
    // the threadidx inside this block, the blockIdx in global grid.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Global row and Global col.
    int row_c = by * tileSize + ty;
    int col_c = bx * tileSize + tx;
    float sum = 0;
    // Load the block into the Shared Memory.
    for (int k_out = 0; k_out < K; k_out += tileSize) {
        int row_a = by * tileSize + ty;
        int col_a = k_out + tx;
        if (row_a < M && col_a < K) {
            As[ty][tx] = A[row_a * K + col_a];
        } else {
            As[ty][tx] = 0;
        }

        int row_b = k_out + ty;
        int col_b = bx * tileSize + tx;
        if (row_b < K && col_b < N) {
            Bs[ty][tx] = B[row_b * N + col_b];
        } else {
            Bs[ty][tx] = 0;
        }
        __syncthreads();
        for (int k_in = 0; k_in < tileSize; k_in++) {
            sum += As[ty][k_in] * Bs[k_in][tx];
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = alpha * sum + beta * C[row_c * N + col_c];
    }
}

void launch_sgemm_smem(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_smem<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

__global__ void sgemm_smem_opt(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C,
    int tileSize) {
    // Shared memory has the same location when dynamically allocated during launching.
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = &smem[tileSize * tileSize];
    // the threadidx inside this block, the blockIdx in global grid.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Global row and Global col.
    int row_c = by * tileSize + ty;
    int col_c = bx * tileSize + tx;
    float sum = 0;
    // Load the block into the Shared Memory.
    for (int k_out = 0; k_out < K; k_out += tileSize) {
        int row_a = by * tileSize + ty;
        int col_a = k_out + tx;
        if (row_a < M && col_a < K) {
            As[ty * tileSize + tx] = A[row_a * K + col_a];
        } else {
            As[ty * tileSize + tx] = 0;
        }

        int row_b = k_out + ty;
        int col_b = bx * tileSize + tx;
        if (row_b < K && col_b < N) {
            Bs[ty * tileSize + tx] = B[row_b * N + col_b];
        } else {
            Bs[ty * tileSize + tx] = 0;
        }
        __syncthreads();
        for (int k_in = 0; k_in < tileSize; k_in++) {
            sum += As[ty * tileSize + k_in] * Bs[k_in * tileSize + tx];
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = alpha * sum + beta * C[row_c * N + col_c];
    }
}

void launch_sgemm_smem_opt(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim, int const tileSize,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_smem_opt<<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C, tileSize);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_reg_blocking(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    // The shared memories.
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    // thread in current block.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    // The block_row and col in C.
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    // Thread output block coords.
    const int output_row = block_row + ty * TM;
    const int output_col = block_col + tx * TN;
    // The sum of storing.
    float sum[TM][TN]{0.0};
    // Here we started load and calculate.
    for (int k_out = 0; k_out < K; k_out += BK) {
        // k0 is going to load A followed by col, load B folloed by row.
        // Load A followd by col.
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            // The share memory coords.
            int share_row = idx / BK;
            int share_col = idx % BK;
            // The global memory coords.
            int global_row = block_row + share_row;
            int global_col = k_out + share_col;
            if (global_row < M && global_col < K) {
                As[share_row][share_col] = A[global_row * K + global_col];
            } else {
                As[share_row][share_col] = 0.0f;
            }
        }
        // load B followed by row.
        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            // The share memory coords.
            int share_row = idx / BN;
            int share_col = idx % BN;
            // The global memory coords.
            int global_row = k_out + share_row;
            int global_col = block_col + share_col;
            if (global_row < K && global_col < N) {
                Bs[share_row][share_col] = B[global_row * N + global_col];
            } else {
                Bs[share_row][share_col] = 0.0f;
            }
        }
        __syncthreads();
// Compute now.
#pragma unroll
        for (int k_in = 0; k_in < BK; k_in++) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    sum[i][j] = As[ty * TM + i][k_in] * Bs[k_in][tx * TN + j];
                }
            }
        }
        __syncthreads();
    }
// Store back to C.
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = output_row + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                int col = output_col + j;
                if (col < N) {
                    C[row * N + col] = alpha * sum[i][j] + beta * C[row * N + col];
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_sgemm_reg_blocking(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize, cudaStream_t stream) {
    sgemm_reg_blocking<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

cublasStatus_t CublasLauncher(
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));
    CUBLAS_CHECK(cublasDestroy(handle));
    return CUBLAS_STATUS_SUCCESS;
}