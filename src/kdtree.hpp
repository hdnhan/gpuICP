#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <vector>

class KDTree {
  public:
    KDTree(std::vector<float3> target, cudaStream_t stream);
    ~KDTree() {}

    // if inlierThreshold is non-positive, it will be ignored
    std::vector<float> findAllNearestDistance(std::vector<float3> source, float inlierThreshold,
                                              cudaStream_t stream);

  private:
    uint32_t n_target;
    thrust::device_vector<float3> d_target;
};
