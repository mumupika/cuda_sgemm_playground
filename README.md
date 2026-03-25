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
Device: NVIDIA L40
Total global memory: 47676129280 Bytes
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
cublas average elapsed time: 3.151971 ms, Calculate capability: 43604.127676 GFlops/s.
cutlass average elapsed time: 3.260141 ms, Calculate capability: 42157.367083 GFlops/s.

kernel 1 average elapsed time: 203.513190 ms, Calculate capability: 675.331922 GFlops/s.
kernel 2 average elapsed time: 25.753898 ms, Calculate capability: 5336.627304 GFlops/s.
kernel 3 average elapsed time: 26.784688 ms, Calculate capability: 5131.250866 GFlops/s.
kernel 4 average elapsed time: 28.972742 ms, Calculate capability: 4743.732964 GFlops/s.
kernel 5 average elapsed time: 29.728295 ms, Calculate capability: 4623.169795 GFlops/s.
kernel 6 average elapsed time: 8.317194 ms, Calculate capability: 16524.678689 GFlops/s.
kernel 7 average elapsed time: 4.162794 ms, Calculate capability: 33016.038520 GFlops/s.
```

## Roadmaps

- [x] Kernel 1: naive sgemm kernel
- [x] Kernel 2: gmem coalesced (opt)
- [x] Kernel 3: one dim gmem coalesced (opt)
- [x] Kernel 4: static 2-dim smem using (degrade)
- [x] Kernel 5: dynamic 1-dim smem using (degrade)
- [x] Kernel 6: increase arithmetic intensity (opt)
- [x] Kernel 7: padding smem + arithmetic intensity to avoid bank conflict (opt)
