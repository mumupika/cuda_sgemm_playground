cd ../
if [ ! -d "logs" ]; then
    mkdir -p logs
fi
cd build/bin
sudo /usr/local/cuda/bin/ncu --set full -o ../../logs/device_report -f --target-processes all ./gemm_prof 2>&1 | tee ../../logs/device_prof.log
sudo /usr/local/cuda/bin/ncu --set full -f --target-processes all ./gemm_prof 2>&1 | tee ../../logs/device_prof.log