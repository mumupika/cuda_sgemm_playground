#pragma once

#include "helper.h"
#define KERNEL_NUMS 5

template <int KernelId>
GpuTimer run_kernel(int const M, int const N, int const K,
                    float *hA, float *hB, float *hC,
                    float *dA, float *dB, float *dC,
                    float &alpha, float &beta,
                    bool const check_result_flag) {
    static_assert(KernelId >= 1 && KernelId <= KERNEL_NUMS, "Invalid kernel ID");
    return GpuTimer();
}

#define KERNEL(id)                             \
    template <>                                \
    GpuTimer run_kernel<id>(                   \
        int const M, int const N, int const K, \
        float *hA, float *hB, float *hC,       \
        float *dA, float *dB, float *dC,       \
        float &alpha, float &beta,             \
        bool const check_result_flag);

#define KERNEL_GEN \
    KERNEL(1)      \
    KERNEL(2)      \
    KERNEL(3)      \
    KERNEL(4)      \
    KERNEL(5)

KERNEL_GEN

#undef KERNEL_GEN
#undef KERNEL
#undef KERNEL_NUMS

GpuTimer run_cutlass(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

GpuTimer run_cublas(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);