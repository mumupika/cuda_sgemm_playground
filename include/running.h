#pragma once

#define KERNEL_NUMS 5

template <int KernelId>
void run_kernel(int const M, int const N, int const K,
                float *hA, float *hB, float *hC,
                float *dA, float *dB, float *dC,
                float &alpha, float &beta,
                bool const check_result_flag) {
    static_assert(KernelId >= 1 && KernelId <= KERNEL_NUMS, "Invalid kernel ID");
}

#define KERNEL(id)                             \
    template <>                                \
    void run_kernel<id>(                       \
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

void run_cutlass(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

void run_cublas(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);