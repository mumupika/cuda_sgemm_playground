cd ../
if [ ! -d "logs" ]; then
    mkdir -p logs
fi
cd build

# compute-sanitizer check.
compute-sanitizer --tool=memcheck ./gemm 2>&1 | tee ../logs/device_leak_check.log
# valgrind check.
valgrind --leak-check=full ./gemm 2>&1 | tee ../logs/leak.log