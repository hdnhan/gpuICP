#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <vector>

class ICP; // Forward declaration of ICP class

class KDTree {
    friend class ICP; // Allow ICP class to access private members of KDTree
  public:
    // Build the KDTree
    // target: the target point cloud
    // stream: the CUDA stream to use for building the KDTree
    void buildTree(std::vector<float3> const &target, cudaStream_t stream);

    // Find the nearest neighbor for each point in source
    // source: the source point cloud
    // inlierThreshold: the maximum distance to consider a point as an inlier, if inlierThreshold is
    // non-positive, it will be ignored stream: the CUDA stream to use for finding nearest neighbors Returns a
    // vector of distances to the nearest neighbor for each point in source
    std::vector<float> findAllNearestDistance(std::vector<float3> const &source, float inlierThreshold,
                                              cudaStream_t stream);

  private:
    void findCorrespondences(float3 const *d_source, uint32_t n_source, float inlierThreshold,
                             uint32_t *inlier, float *dsx, float *dsy, float *dsz, float *dtx, float *dty,
                             float *dtz, cudaStream_t stream);

  private:
    uint32_t n_target;
    thrust::device_vector<float3> d_target;
};
