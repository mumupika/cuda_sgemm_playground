cd ../
if [ ! -d "logs" ]; then
    mkdir -p logs
fi
cd build/bin

# compute-sanitizer check.
compute-sanitizer --tool=memcheck ./gemm_test 2>&1 | tee ../../logs/device_leak_check.log
# valgrind check.
valgrind --leak-check=full ./gemm_test 2>&1 | tee ../../logs/leak.log