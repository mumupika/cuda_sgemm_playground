#pragma once

#include "helper.h"
#define KERNEL_NUMS 5

template <int KernelId>
std::enable_if_t<((KernelId >= 0) && (KernelId < KERNEL_NUMS + 1)), float>
run_kernel(int const M, int const N, int const K,
           float *hA, float *hB, float *hC,
           float *dA, float *dB, float *dC,
           float const alpha, float const beta,
           bool const check_result_flag) {
    return float();
}

#define KERNEL(id)                             \
    template <>                                \
    float run_kernel<id>(                   \
        int const M, int const N, int const K, \
        float *hA, float *hB, float *hC,       \
        float *dA, float *dB, float *dC,       \
        float const alpha, float const beta,   \
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

float run_cutlass(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag);

float run_cublas(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float const alpha, float const beta,
    bool const check_result_flag);