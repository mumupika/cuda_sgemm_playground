#pragma once
#include "helper.h"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_reg_blocking(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
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
                As[share_row][share_col] = A[global_row * ldA + global_col];
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
                Bs[share_row][share_col] = B[global_row * ldB + global_col];
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
                    C[row * ldC + col] = alpha * sum[i][j] + beta * C[row * ldC + col];
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
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_reg_blocking<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief trying to remove the 2-way bank confilicts in smem.
 *
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_reg_block_opt(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
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
                As[share_row * (BK + 1) + share_col] = A[global_row * ldA + global_col];
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
                Bs[share_row * (BN + 1) + share_col] = B[global_row * ldB + global_col];
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
                    C[row * ldC + col] = alpha * sum[i * TN + j] + beta * C[row * ldC + col];
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
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_reg_block_opt<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vec_load(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    extern __shared__ float smem[];
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
        for (int idx = tid * 4; idx < BM * BK; idx += num_threads * 4) {
            int share_row = idx / BK;
            int share_col = idx % BK;
            int global_row = block_row + share_row;
            int global_col = k_out + share_col;
            float4 global_A = *reinterpret_cast<const float4 *>(&A[global_row * ldA + global_col]);
            As[share_row * (BK + 1) + share_col] = global_A.x;
            As[share_row * (BK + 1) + share_col + 1] = global_A.y;
            As[share_row * (BK + 1) + share_col + 2] = global_A.z;
            As[share_row * (BK + 1) + share_col + 3] = global_A.w;
        }
        for (int idx = tid * 4; idx < BK * BN; idx += num_threads * 4) {
            int share_row = idx / BN;
            int share_col = idx % BN;
            int global_row = k_out + share_row;
            int global_col = block_col + share_col;
            float4 global_B = *reinterpret_cast<const float4 *>(&B[global_row * ldB + global_col]);
            Bs[share_row * (BN + 1) + share_col] = global_B.x;
            Bs[share_row * (BN + 1) + share_col + 1] = global_B.y;
            Bs[share_row * (BN + 1) + share_col + 2] = global_B.z;
            Bs[share_row * (BN + 1) + share_col + 3] = global_B.w;
        }
        __syncthreads();
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
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = output_row + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int col = output_col + j;
                float4 tempC = *reinterpret_cast<float4 *>(&C[row * ldC + col]);
                tempC.x = alpha * sum[i * TN + j] + beta * tempC.x;
                tempC.y = alpha * sum[i * TN + j + 1] + beta * tempC.y;
                tempC.z = alpha * sum[i * TN + j + 2] + beta * tempC.z;
                tempC.w = alpha * sum[i * TN + j + 3] + beta * tempC.w;
                *reinterpret_cast<float4 *>(&C[row * ldC + col]) = tempC;
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
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_vec_load<BM, BN, BK, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemm_warp_tiling(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = &smem[BM * (BK + 1)];

    // We should calculate the base address for all hierarchies.
    // First the block. From execute model -> memory model.
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    // Then the warp.
    // First we get the thread in the block.
    const int ty = threadIdx.y; // thread row.
    const int tx = threadIdx.x; // thread col.
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    // Then we can get the warp id.
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    // Get the warp in blocks.
    const int num_warp_cols = BN / WN;
    const int warp_idy = warp_id / num_warp_cols;
    const int warp_idx = warp_id % num_warp_cols;
    // Get the lane in warp.
    const int num_lane_cols = WN / TN;
    const int lane_idy = lane_id / num_lane_cols;
    const int lane_idx = lane_id % num_lane_cols;

    // The registers file.
    float regA[TM];
    float regB[TN];
    float sum[TM * TN]{0.0};

    // The output related hierarchy.
    const int block_row = by * BM;
    const int block_col = bx * BN;
    const int warp_row_in_block = warp_idy * WM;
    const int warp_col_in_block = warp_idx * WN;
    const int lane_row_in_warp = lane_idy * TM;
    const int lane_col_in_warp = lane_idx * TN;

    // BlockTile: Load from GMEM -> SMEM;
    for (int kb = 0; kb < K; kb += BK) {
        for (int idx = tid * 4; idx < BM * BK; idx += num_threads * 4) {
            int share_row = idx / BK;
            int share_col = idx % BK;
            int global_row = block_row + share_row;
            int global_col = kb + share_col;
            float4 global_A = *reinterpret_cast<const float4 *>(&A[global_row * ldA + global_col]);
            As[share_row * (BK + 1) + share_col] = global_A.x;
            As[share_row * (BK + 1) + (share_col + 1)] = global_A.y;
            As[share_row * (BK + 1) + (share_col + 2)] = global_A.z;
            As[share_row * (BK + 1) + (share_col + 3)] = global_A.w;
        }
        for (int idx = tid * 4; idx < BK * BN; idx += num_threads * 4) {
            int share_row = idx / BN;
            int share_col = idx % BN;
            int global_row = kb + share_row;
            int global_col = block_col + share_col;
            float4 global_B = *reinterpret_cast<const float4 *>(&B[global_row * ldB + global_col]);
            Bs[share_row * BN + share_col] = global_B.x;
            Bs[share_row * BN + share_col + 1] = global_B.y;
            Bs[share_row * BN + share_col + 2] = global_B.z;
            Bs[share_row * BN + share_col + 3] = global_B.w;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < BK; kt++) {
// Load As to register.
#pragma unroll
            for (int ki = 0; ki < TM; ki++) {
                regA[ki] = As[(warp_row_in_block + lane_row_in_warp + ki) * (BK + 1) + kt];
            }
// Load Bs to register.
#pragma unroll
            for (int ki = 0; ki < TN; ki += 4) {
                *reinterpret_cast<float4 *>(&regB[ki]) = *reinterpret_cast<const float4 *>(&Bs[kt * BN + (warp_col_in_block + lane_col_in_warp + ki)]);
            }
// Now we Calculate and store in sum.
#pragma unroll
            for (int ki = 0; ki < TM; ki++) {
#pragma unroll
                for (int kj = 0; kj < TN; kj++) {
                    sum[ki * TN + kj] += regA[ki] * regB[kj];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = block_row + warp_row_in_block + lane_row_in_warp + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int col = block_col + warp_col_in_block + lane_col_in_warp + j;
                float4 tempC = *reinterpret_cast<float4 *>(&C[row * ldC + col]);
                tempC.x = alpha * sum[i * TN + j] + beta * tempC.x;
                tempC.y = alpha * sum[i * TN + j + 1] + beta * tempC.y;
                tempC.z = alpha * sum[i * TN + j + 2] + beta * tempC.z;
                tempC.w = alpha * sum[i * TN + j + 3] + beta * tempC.w;
                *reinterpret_cast<float4 *>(&C[row * ldC + col]) = tempC;
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
void launch_sgemm_warp_tiling(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_warp_tiling<BM, BN, BK, WM, WN, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void __launch_bounds__(256, 2) sgemm_warp_tiling_swizzle(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    // swizzle Calculation. 
    #define GET_A(col, row) ((col) * BM + ((row) ^ ((col) & ~3)))       // Swizzle<3, 2, 7> swz;
    
    // Shared memory Allocation. No padding needed.
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = &smem[BM * BK];

    // The registers file.
    float regA[TM];
    float regB[TN];
    float sum[TM * TN]{0.0};
    
    // The values that need to use in runtime.
    RuntimeHelper<BM, BN, BK, WM, WN, TM, TN> rh{};

    // BlockTile: Load from GMEM -> SMEM;
    for (int kb = 0; kb < K; kb += BK) {
        for (int idx = rh.tid * 4; idx < BM * BK; idx += rh.num_threads * 4) {
            int share_row = idx / BK;
            int share_col = idx % BK;
            int global_row = rh.block_row + share_row;
            int global_col = kb + share_col;
            float4 global_A = *reinterpret_cast<const float4 *>(&A[global_row * ldA + global_col]);
            As[GET_A(share_col, share_row)] = global_A.x;
            As[GET_A(share_col + 1, share_row)] = global_A.y;
            As[GET_A(share_col + 2, share_row)] = global_A.z;
            As[GET_A(share_col + 3, share_row)] = global_A.w;
        }
        for (int idx = rh.tid * 4; idx < BK * BN; idx += rh.num_threads * 4) {
            int share_row = idx / BN;
            int share_col = idx % BN;
            int global_row = kb + share_row;
            int global_col = rh.block_col + share_col;
            float4 global_B = *reinterpret_cast<const float4 *>(&B[global_row * ldB + global_col]);
            *reinterpret_cast<float4 *>(&Bs[share_row * BN + share_col]) = global_B;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < BK; kt++) {
#pragma unroll
            for (int ki = 0; ki < TM; ki += 4) {
                *reinterpret_cast<float4 *>(&regA[ki]) = *reinterpret_cast<const float4 *>(&As[GET_A(kt, rh.row_A + ki)]);
            }
#pragma unroll
            for (int ki = 0; ki < TN; ki += 4) {
                *reinterpret_cast<float4 *>(&regB[ki]) = *reinterpret_cast<const float4 *>(&Bs[kt * BN + rh.col_B + ki]);
            }
// Now we Calculate and store in sum.
#pragma unroll
            for (int ki = 0; ki < TM; ki++) {
#pragma unroll
                for (int kj = 0; kj < TN; kj++) {
                    sum[ki * TN + kj] += regA[ki] * regB[kj];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        if ((rh.res_row + i) < M) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                float4 tempC = *reinterpret_cast<float4 *>(&C[(rh.res_row + i) * ldC + (rh.res_col + j)]);
                tempC.x = alpha * sum[i * TN + j] + beta * tempC.x;
                tempC.y = alpha * sum[i * TN + j + 1] + beta * tempC.y;
                tempC.z = alpha * sum[i * TN + j + 2] + beta * tempC.z;
                tempC.w = alpha * sum[i * TN + j + 3] + beta * tempC.w;
                *reinterpret_cast<float4 *>(&C[(rh.res_row + i) * ldC + (rh.res_col + j)]) = tempC;
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
void launch_sgemm_warp_tiling_swizzle(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_warp_tiling_swizzle<BM, BN, BK, WM, WN, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void load_gmem_to_smem(
    const int smem, const int kb,
    const int ldA, const int ldB,
    float **As_buf, float **Bs_buf,
    const float *A, const float *B,
    RuntimeHelper<BM, BN, BK, WM, WN, TM, TN> &rh) {
// Swizzle<3, 2, 6> swz;
#define GET_OFFSET(rown, row, col) (row) * (rown) + (col)
#define GET_SWZ(num) ((((num) >> 9) << 2) ^ (num))

    for (int idx = rh.tid * 4; idx < BM * BK; idx += rh.num_threads * 4) {
        int share_row = idx / BK;
        int share_col = idx % BK;
        int global_row = rh.block_row + share_row;
        int global_col = kb + share_col;
        float4 global_A = __ldg(reinterpret_cast<const float4 *>(&A[global_row * ldA + global_col]));
        As_buf[smem][GET_SWZ(GET_OFFSET(BM, share_col, share_row))] = global_A.x;
        As_buf[smem][GET_SWZ(GET_OFFSET(BM, share_col + 1, share_row))] = global_A.y;
        As_buf[smem][GET_SWZ(GET_OFFSET(BM, share_col + 2, share_row))] = global_A.z;
        As_buf[smem][GET_SWZ(GET_OFFSET(BM, share_col + 3, share_row))] = global_A.w;
    }
    for (int idx = rh.tid * 4; idx < BK * BN; idx += rh.num_threads * 4) {
        int share_row = idx / BN;
        int share_col = idx % BN;
        int global_row = kb + share_row;
        int global_col = rh.block_col + share_col;
        *reinterpret_cast<float4 *>(&Bs_buf[smem][share_row * BN + share_col]) = __ldg(reinterpret_cast<const float4 *>(&B[global_row * ldB + global_col]));
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void load_smem_to_reg(
    const int reg, const int smem, const int kt,
    float (*regA)[TM], float **As_buf,
    float (*regB)[TN], float **Bs_buf,
    RuntimeHelper<BM, BN, BK, WM, WN, TM, TN> &rh) {
// Swizzle<3, 2, 6> swz;
#define GET_OFFSET(rown, row, col) (row) * (rown) + (col)
#define GET_SWZ(num) ((((num) >> 9) << 2) ^ (num))

#pragma unroll
    for (int ki = 0; ki < TM; ki += 4) {
        *reinterpret_cast<float4 *>(&regA[reg][ki]) = *reinterpret_cast<const float4 *>(&As_buf[smem][GET_SWZ(GET_OFFSET(BM, kt, rh.row_A + ki))]);
    }
#pragma unroll
    for (int ki = 0; ki < TN; ki += 4) {
        *reinterpret_cast<float4 *>(&regB[reg][ki]) = *reinterpret_cast<const float4 *>(&Bs_buf[smem][kt * BN + rh.col_B + ki]);
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void compute_result(
    int reg, float *sum, float (*regA)[TM], float (*regB)[TN]) {
#pragma unroll
    for (int ki = 0; ki < TM; ki++) {
#pragma unroll
        for (int kj = 0; kj < TN; kj++) {
            sum[ki * TN + kj] += regA[reg][ki] * regB[reg][kj];
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void store_back_result(
    const int M, const int ldC,
    const float alpha, const float beta,
    float *sum, float *C,
    RuntimeHelper<BM, BN, BK, WM, WN, TM, TN> &rh) {
#pragma unroll
    for (int i = 0; i < TM; i++) {
        if ((rh.res_row + i) < M) {
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                float4 tempC = *reinterpret_cast<float4 *>(&C[(rh.res_row + i) * ldC + (rh.res_col + j)]);
                tempC.x = alpha * sum[i * TN + j] + beta * tempC.x;
                tempC.y = alpha * sum[i * TN + j + 1] + beta * tempC.y;
                tempC.z = alpha * sum[i * TN + j + 2] + beta * tempC.z;
                tempC.w = alpha * sum[i * TN + j + 3] + beta * tempC.w;
                *reinterpret_cast<float4 *>(&C[(rh.res_row + i) * ldC + (rh.res_col + j)]) = tempC;
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void __launch_bounds__(256, 2) sgemm_dbo_warp_tiling_swizzle(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha,
    const float *A, const float *B,
    float beta, float *C) {
    // Double buffer smem.
    extern __shared__ float smem[];
    float *As_buf[2];
    float *Bs_buf[2];
    As_buf[0] = smem;
    As_buf[1] = &smem[BM * BK];
    Bs_buf[0] = &smem[2 * BM * BK];
    Bs_buf[1] = &smem[2 * BM * BK + BK * BN];

    // The registers file.
    float regA[2][TM];
    float regB[2][TN];
    float sum[TM * TN]{0.0};

    // Calculate the runtime parameters.
    RuntimeHelper<BM, BN, BK, WM, WN, TM, TN> rh{};

    // === smem Prologue
    bool cur_smem = 0;
    load_gmem_to_smem<BM, BN, BK, WM, WN, TM, TN>(cur_smem, 0, ldA, ldB, As_buf, Bs_buf, A, B, rh);
    __syncthreads();

    // === smem DBO
    for (int kb = BK; kb < K; kb += BK) {
        bool next_smem = !cur_smem;
        load_gmem_to_smem<BM, BN, BK, WM, WN, TM, TN>(next_smem, kb, ldA, ldB, As_buf, Bs_buf, A, B, rh);

        // ===== reg prologue.
        bool cur_reg = 0;
        load_smem_to_reg<BM, BN, BK, WM, WN, TM, TN>(cur_reg, cur_smem, 0, regA, As_buf, regB, Bs_buf, rh);
        // ===== reg DBO.
#pragma unroll
        for (int kt = 1; kt < BK; kt++) {
            bool next_reg = !cur_reg;
            load_smem_to_reg<BM, BN, BK, WM, WN, TM, TN>(next_reg, cur_smem, kt, regA, As_buf, regB, Bs_buf, rh);
            compute_result<BM, BN, BK, WM, WN, TM, TN>(cur_reg, sum, regA, regB);
            cur_reg = next_reg;
        }
        compute_result<BM, BN, BK, WM, WN, TM, TN>(cur_reg, sum, regA, regB);
        __syncthreads();
        cur_smem = next_smem;
    }

    // === smem epilogue.
    // ===== reg prologue.
    bool cur_reg = 0;
    load_smem_to_reg<BM, BN, BK, WM, WN, TM, TN>(cur_reg, cur_smem, 0, regA, As_buf, regB, Bs_buf, rh);
    // ===== reg DBO.
#pragma unroll
    for (int kt = 1; kt < BK; kt++) {
        bool next_reg = !cur_reg;
        load_smem_to_reg<BM, BN, BK, WM, WN, TM, TN>(next_reg, cur_smem, kt, regA, As_buf, regB, Bs_buf, rh);
        compute_result<BM, BN, BK, WM, WN, TM, TN>(cur_reg, sum, regA, regB);
        cur_reg = next_reg;
    }
    compute_result<BM, BN, BK, WM, WN, TM, TN>(cur_reg, sum, regA, regB);
    // === write back.
    store_back_result<BM, BN, BK, WM, WN, TM, TN>(M, ldC, alpha, beta, sum, C, rh);
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
void launch_sgemm_dbo_warp_tiling_swizzle(
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    dim3 gridDim, dim3 blockDim,
    size_t sharedMemSize = 0, cudaStream_t stream = 0) {
    sgemm_dbo_warp_tiling_swizzle<BM, BN, BK, WM, WN, TM, TN><<<gridDim, blockDim, sharedMemSize, stream>>>(M, N, K, ldA, ldB, ldC, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}