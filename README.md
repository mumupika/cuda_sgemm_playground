# Optimising GEMM

This is a simple project for optimising the gemm with comparison to CUBLAS and CUTLASS. (Learning...)

## References:

[SIBOEHM: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

[HAMZA: Worklog: Optimising GEMM on NVIDIA H100 for cuBLAS-like Performance (WIP)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/)

## License:

Personal usage only.

## Experiment Result

```txt
============================= Properties =============================
Device: NVIDIA GeForce RTX 5090
Total global memory: 33668988928 Bytes
Total Constant memory: 65536 Bytes
Shared mem per Block: 49152 Bytes
Regs per block: 65536
WarpSize: 32
maxThreadsPerBlock: 1024
maxThreadsDim: (1024, 1024, 64)
maxGridSize: (2147483647, 65535, 65535)
max Concurrent kers: 1
async engine cnt: 2
=====================================================================

M = 4096, N = 4096, K = 4096
kernel 1 average elapsed time: 203.515846 ms, Calculate capability: 675.323106 GFlops/s.
kernel 2 average elapsed time: 25.308681 ms, Calculate capability: 5430.506310 GFlops/s.
kernel 3 average elapsed time: 26.328311 ms, Calculate capability: 5220.196451 GFlops/s.
kernel 4 average elapsed time: 28.976618 ms, Calculate capability: 4743.098534 GFlops/s.
kernel 5 average elapsed time: 29.711123 ms, Calculate capability: 4625.841716 GFlops/s.
```
