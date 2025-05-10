```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel
./build/benchmark/kdtree_test -N 1000000 -Q 1000000 --iter 2
```