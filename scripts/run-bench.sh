cd ../
if [ ! -d "logs" ]; then
    mkdir -p logs
fi
cd build/bin
./gemm_bench 2>&1 | tee ../../logs/device_bench.log