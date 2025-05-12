### Build docker image
```bash
# For CUDA 11.4
docker build -t test -f Dockerfile.cuda114 .

# For CUDA 12.1
docker build -t test -f Dockerfile.cuda121 .
```

### Run tests
```bash
docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace test bash

# Inside the container, run
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release --parallel

# Run KdTree benchmark with random data
./build/benchmark/kdtree_test -N 1000000 -Q 1000000 --repeat 2

# Run ICP benchmark
./build/benchmark/icp_test --repeat 1 --maxiter 1000 -s assets/source.ply -t assets/target.ply
```