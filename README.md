```bash
sudo apt install libpcl-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel

# Run KdTree benchmark with random data
./build/benchmark/kdtree_test -N 1000000 -Q 1000000 --iter 2

# Run ICP benchmark
./build/benchmark/icp_test --repeat 1 --maxiter 1000 -s assets/source.ply -t assets/target.ply
```