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

/**
 * @brief trying to remove the 2-way bank confilicts in smem.
 *
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_reg_block_opt(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    extern __shared__ float smem[]; // One dim shared memory for dynamic allocation.
    // Padding As with BK + 1. So As has (BM, BK + 1);
    // Padding Bs with BN + 1. So Bs has (BK, BN + 1);
    float *As = smem;
    float *Bs = &smem[BM * (BK + 1)];
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
    float sum[TM * TN]{0.0f};
    for (int k_out = 0; k_out < K; k_out += BK) {
        // Coalescing load for global memory.
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            // The share memory coords.
            int share_row = idx / BK;
            int share_col = idx % BK;
            // The global memory coords.
            int global_row = block_row + share_row;
            int global_col = k_out + share_col;
            if (global_row < M && global_col < K) {
                As[share_row * (BK + 1) + share_col] = A[global_row * K + global_col];
            } else {
                As[share_row * (BK + 1) + share_col] = 0.0f;
            }
        }
        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            // The share memory coords.
            int share_row = idx / BN;
            int share_col = idx % BN;
            // The global memory coords.
            int global_row = k_out + share_row;
            int global_col = block_col + share_col;
            if (global_row < K && global_col < N) {
                Bs[share_row * (BN + 1) + share_col] = B[global_row * N + global_col];
            } else {
                Bs[share_row * (BN + 1) + share_col] = 0.0f;
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
                    sum[i * TN + j] += As[(ty * TM + i) * (BK + 1) + k_in] * Bs[k_in * (BN + 1) + tx * TN + j];
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
                    C[row * N + col] = alpha * sum[i * TN + j] + beta * C[row * N + col];
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
void launch_sgemm_reg_block_opt(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_reg_block_opt<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vec_load(
    int M, int N, int K,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = &smem[BM * BK];
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
    float sum[TM * TN]{0.0f};
    for (int k_out = 0; k_out < K; k_out += BK) {
        for (int idx = tid * 4; idx < BM * BK; idx += num_threads * 4) {
            int share_row = idx / BK;
            int share_col = idx % BK;
            int global_row = block_row + share_row;
            int global_col = k_out + share_col;
            float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_row < M) {
                if (global_col + 3 < K && ((global_row * K + global_col) & 3) == 0) {
                    // vectorized load.
                    float4 global_A = *reinterpret_cast<const float4 *>(&A[global_row * K + global_col]);
                    vals[0] = global_A.x;
                    vals[1] = global_A.y;
                    vals[2] = global_A.z;
                    vals[3] = global_A.w;
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (global_col + i < K) {
                            vals[i] = A[global_row * K + global_col + i];
                        }
                    }
                }
            }
            As[share_row * BK + share_col + 0] = vals[0];
            As[share_row * BK + share_col + 1] = vals[1];
            As[share_row * BK + share_col + 2] = vals[2];
            As[share_row * BK + share_col + 3] = vals[3];
        }

        for (int idx = tid * 4; idx < BK * BN; idx += num_threads * 4) {
            int share_row = idx / BN;
            int share_col = idx % BN;
            int global_row = k_out + share_row;
            int global_col = block_col + share_col;
            float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_row < K) {
                if (global_col + 3 < N && ((global_row * N + global_col) & 3) == 0) {
                    // vectorized load.
                    float4 global_B = *reinterpret_cast<const float4 *>(&B[global_row * N + global_col]);
                    vals[0] = global_B.x;
                    vals[1] = global_B.y;
                    vals[2] = global_B.z;
                    vals[3] = global_B.w;
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (global_col + i < N) {
                            vals[i] = B[global_row * N + global_col + i];
                        }
                    }
                }
            }
            Bs[share_row * BN + share_col + 0] = vals[0];
            Bs[share_row * BN + share_col + 1] = vals[1];
            Bs[share_row * BN + share_col + 2] = vals[2];
            Bs[share_row * BN + share_col + 3] = vals[3];
        }
        __syncthreads();
#pragma unroll
        for (int k_in = 0; k_in < BK; k_in++) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    sum[i * TN + j] += As[(ty * TM + i) * BK + k_in] * Bs[k_in * BN + tx * TN + j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = output_row + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int col = output_col + j;
                if (col + 3 < N && ((row * N + col) & 3) == 0) {
                    float4 tempC;
                    tempC = *reinterpret_cast<float4 *>(&C[row * N + col]);
                    tempC.x = alpha * sum[i * TN + j] + beta * tempC.x;
                    tempC.y = alpha * sum[i * TN + j + 1] + beta * tempC.y;
                    tempC.z = alpha * sum[i * TN + j + 2] + beta * tempC.z;
                    tempC.w = alpha * sum[i * TN + j + 3] + beta * tempC.w;
                    *reinterpret_cast<float4 *>(&C[row * N + col]) = tempC;
                } else {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (col + k < N) {
                            C[row * N + col + k] = alpha * sum[i * TN + j + k] + beta * C[row * N + col + k];
                        }
                    }
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
void launch_sgemm_vec_load(
    int M, int N, int K,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_vec_load<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, alpha, A, B, beta, C);
}