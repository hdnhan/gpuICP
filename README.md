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
./build/benchmark/icp_test \
    --repeat 1 \
    --maxiter 1000 \
    -s assets/source.ply \
    -t assets/target.ply
```

### Run on g4ad.xlarge (AMD instance)
| GPU KdTree | PCL KdTree |
|------------|------------|
| 83ms       | 2433ms     |

| GPU ICP | PCL ICP |
|---------|---------|
| 39ms    | 1173ms  |

### Run on Kaggle P100
https://www.kaggle.com/code/hdnhan28/benchmark-gpu-icp-vs-pcl-icp

| GPU KdTree | PCL KdTree |
|----------|--------------|
| 61ms     | 4105ms       |

| GPU ICP | PCL ICP |
|-------|-----------|
| 6ms   | 3006ms    |
