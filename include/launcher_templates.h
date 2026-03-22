#pragma once
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
                    sum[i][j] += As[ty * TM + i][k_in] * Bs[k_in][tx * TN + j];
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

/**
 * @brief The kernel for sGemm optimized for compute intensity.
 *
 * @tparam BM The tile of As row.
 * @tparam BN The tile of Bs col.
 * @tparam BK The tile of As col and Bs row.
 * @tparam TM The threads calculated row num.
 * @tparam TN The threads calculated col num.
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
template <int BM, int BN, int BK, int TM, int TN>
void launch_sgemm_reg_blocking(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_reg_blocking<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}