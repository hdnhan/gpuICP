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
```bash
ubuntu@ip-172-31-30-56:~/gpuICP$ ./build/benchmark/kdtree_test -N 1000000 -Q 1000000 --repeat 2
[05/23/25 14:39:29.441] [info] N: 1000000, Q: 1000000, repeat: 2
[05/23/25 14:39:29.441] [info] Repeat: 1
[05/23/25 14:39:32.377] [info] PCL KdTree: 2404 ms
[05/23/25 14:39:32.707] [info] CUDA KdTree
[05/23/25 14:39:32.797] [info] Build: 89 ms
[05/23/25 14:39:32.828] [info] Queries: 31 ms
[05/23/25 14:39:32.828] [info] Total Latency: 120 ms
[05/23/25 14:39:32.829] [info] Max diff: 3.7252903e-09 at index 160
[05/23/25 14:39:32.837] [info] Repeat: 2
[05/23/25 14:39:35.706] [info] PCL KdTree: 2347 ms
[05/23/25 14:39:35.708] [info] CUDA KdTree
[05/23/25 14:39:35.775] [info] Build: 67 ms
[05/23/25 14:39:35.804] [info] Queries: 28 ms
[05/23/25 14:39:35.804] [info] Total Latency: 96 ms
[05/23/25 14:39:35.805] [info] Max diff: 3.7252903e-09 at index 0
[05/23/25 14:39:35.814] [info] Finished all repeats.


ubuntu@ip-172-31-30-56:~/gpuICP$ ./build/benchmark/icp_test \
    --repeat 1 \
    --maxiter 1000 \
    -s assets/source.ply \
    -t assets/target.ply
[05/23/25 14:40:05.447] [info] Source: assets/source.ply, Target: assets/target.ply, repeat: 1
[05/23/25 14:40:05.585] [info] Source size: 40256, Target size: 35947
[05/23/25 14:40:05.585] [info] Repeat: 1
[05/23/25 14:40:06.790] [info] PCL ICP: 1205 ms
[05/23/25 14:40:06.790] [info] PCL ICP converged, percent: 83.830986
[05/23/25 14:40:06.790] [info] PCL ICP result:
0.993880 0.089887 -0.064225 0.000305 
-0.088332 0.995736 0.026664 -0.000211 
0.066348 -0.020826 0.997581 -0.000629 
0.000000 0.000000 0.000000 1.000000 

[05/23/25 14:40:07.192] [info] CUDA ICP: 68 ms
[05/23/25 14:40:07.192] [info] CUDA ICP converged, percent: 100
[05/23/25 14:40:07.192] [info] CUDA ICP result:
0.993603 0.095283 -0.060659 0.000010 
-0.093322 0.995046 0.034374 -0.000171 
0.063633 -0.028492 0.997570 -0.000170 
0.000000 0.000000 0.000000 1.000000 
[05/23/25 14:40:07.193] [info] Finished all repeats.
```

### Run on Kaggle P100
https://www.kaggle.com/code/hdnhan28/benchmark-cuicp-vs-pcl-icp

| cuKdTree | PCL KdTree |
|----------|------------|
| 73ms     | 4105ms     |

| cuICP | PCL ICP |
|-------|---------|
| 37ms  | 3006ms  |
