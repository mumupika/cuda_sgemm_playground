#pragma once

/**
 * @brief Get the device properties.
 *
 */
void GetProperties();

/**
 * @brief Prepare datas for copy to device and sgemm.
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA Pointer to matrix A on the host.
 * @param hB Pointer to matrix B on the host.
 * @param hC Pointer to matrix C on the host.
 * @param dA Pointer to matrix A on the device.
 * @param dB Pointer to matrix B on the device.
 * @param dC Pointer to matrix C on the device.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 */
void prepare_matrix(
    int M, int N, int K,             // Dimensions;
    float *hA, float *hB, float *hC, // Host data;
    float *dA, float *dB, float *dC, // Device Data;
    float &alpha, float &beta        // bias.
);

/**
 * @brief Memory copy from host to device.
 * 
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA Pointer to matrix A on the host.
 * @param hB Pointer to matrix B on the host.
 * @param hC Pointer to matrix C on the host.
 * @param dA Pointer to matrix A on the device.
 * @param dB Pointer to matrix B on the device.
 * @param dC Pointer to matrix C on the device.
 */
void memHtoD(
    int M, int N, int K,             // Dimensions;
    float *hA, float *hB, float *hC, // Host data;
    float *dA, float *dB, float *dC  // Device Data;
);

/**
 * @brief Get the cpu result object
 *
 * @param M The number of rows of matrix A and C.
 * @param N The number of columns of matrix B and C.
 * @param K The number of columns of A and rows of B.
 * @param hA Pointer to matrix A on the host.
 * @param hB Pointer to matrix B on the host.
 * @param hC Pointer to matrix C on the host.
 * @param reference The referencing pointer for parity check.
 * @param alpha Scalar alpha.
 * @param beta Scalar beta.
 */
void get_cpu_result(
    int const M, int const N, int const K,             // Dimensions;
    float const *hA, float const *hB, float const *hC, // Host data;
    float *reference,                                  // Device Data;
    float const alpha, float const beta                // bias.
);