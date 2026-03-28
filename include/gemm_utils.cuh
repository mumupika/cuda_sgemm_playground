#pragma once
#include <string>
#include <running.h>

template <size_t i>
std::string get_kernel_name() {
    if constexpr (i == 0) {
        return "cublas";
    } else if constexpr (i >= 1 && i <= KERNEL_NUMS) {
        return "kernel " + std::to_string(i);
    } else if constexpr (i == KERNEL_NUMS + 1) {
        return "cutlass";
    }
}

template <int KernelId>
auto getKernel() {
    if constexpr (KernelId == 0) {
        return run_cublas;
    } else if constexpr (KernelId >= 1 && KernelId <= KERNEL_NUMS) {
        return run_kernel<KernelId>;
    } else if constexpr (KernelId == KERNEL_NUMS + 1) {
        return run_cutlass;
    }
}