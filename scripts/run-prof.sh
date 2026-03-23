cd ../
if [ ! -d "logs" ]; then
    mkdir -p logs
fi
cd build/bin
sudo /usr/local/cuda/bin/ncu --set full --target-processes all ./gemm_prof 2>&1 | tee ../../logs/device_prof.log