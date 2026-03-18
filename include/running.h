#pragma once

void run_kernel1(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

void run_kernel2(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

void run_kernel3(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

void run_kernel4(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

void run_kernel5(
    int const M, int const N, int const K,
    float *hA, float *hB, float *hC,
    float *dA, float *dB, float *dC,
    float &alpha, float &beta,
    bool const check_result_flag);

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