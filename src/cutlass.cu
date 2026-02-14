
/// third-party: cutlass.

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

cudaError_t CutlassSgemmNN(
    int M, int N, int K,
    float alpha, const float *A, const float *B,
    float beta, float *C) {
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float,
                                                    RowMajor,
                                                    float,
                                                    RowMajor,
                                                    float,
                                                    RowMajor>;
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {alpha, beta});

    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
