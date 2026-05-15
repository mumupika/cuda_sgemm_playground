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
cublas average elapsed time: 3.146611 ms, Calculate capability: 43678.403915 GFlops/s.
cutlass average elapsed time: 3.265565 ms, Calculate capability: 42087.345350 GFlops/s.

kernel 1 average elapsed time: 203.514256 ms, Calculate capability: 675.328382 GFlops/s.
kernel 2 average elapsed time: 25.736381 ms, Calculate capability: 5340.259561 GFlops/s.
kernel 3 average elapsed time: 26.679219 ms, Calculate capability: 5151.535779 GFlops/s.
kernel 4 average elapsed time: 28.980531 ms, Calculate capability: 4742.458031 GFlops/s.
kernel 5 average elapsed time: 29.712617 ms, Calculate capability: 4625.609117 GFlops/s.
kernel 6 average elapsed time: 7.619616 ms, Calculate capability: 18037.516970 GFlops/s.
kernel 7 average elapsed time: 5.152272 ms, Calculate capability: 26675.407247 GFlops/s.
kernel 8 average elapsed time: 3.964093 ms, Calculate capability: 34670.972730 GFlops/s.
kernel 9 average elapsed time: 3.859088 ms, Calculate capability: 35614.361157 GFlops/s.
kernel 10 average elapsed time: 3.620208 ms, Calculate capability: 37964.380342 GFlops/s.
kernel 11 average elapsed time: 4.209661 ms, Calculate capability: 32648.462912 GFlops/s.
```

## Roadmaps

✅ Improved
❌ Degrade

- [x] Kernel 1: naive sgemm kernel (baseline)
- [x] Kernel 2: gmem coalesced ✅
- [x] Kernel 3: one dim gmem coalesced ✅
- [x] Kernel 4: static 2-dim smem using ❌
- [x] Kernel 5: dynamic 1-dim smem using ❌
- [x] Kernel 6: increase arithmetic intensity ✅
- [x] Kernel 7: padding smem + arithmetic intensity to avoid bank conflict ✅
- [x] kernel 8: vectorised loading. ✅
- [x] Kernel 9: single warp tiling. ✅
- [x] Kernel 10: single warp tiling + swizzle with XOR macro. ✅
- [x] Kernel 11: Double Buffer Optimization. ❌
