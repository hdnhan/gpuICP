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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run KdTree benchmark with random data
./build/benchmark/kdtree_test -N 1000000 -Q 1000000 --repeat 10

# Run ICP benchmark
./build/benchmark/icp_test \
    --repeat 10 \
    --maxiter 1000 \
    -s assets/source.ply \
    -t assets/target.ply
```

### Run on g4ad.xlarge (AMD instance)
| GPU KdTree | PCL KdTree |
|------------|------------|
| 86ms       | 2389ms     |

| GPU ICP | PCL ICP |
|---------|---------|
| 46ms    | 1187ms  |

### Run on g4dn.xlarge (NVIDIA instance)
| GPU KdTree | PCL KdTree |
|----------|--------------|
| 93ms     | 3411ms       |

| GPU ICP | PCL ICP |
|---------|---------|
| 35ms    | 1707ms  |

### Run on Kaggle P100
https://www.kaggle.com/code/hdnhan28/benchmark-gpu-icp-vs-pcl-icp
