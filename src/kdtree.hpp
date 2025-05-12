#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <vector>

class ICP; // Forward declaration of ICP class

class KDTree {
    friend class ICP; // Allow ICP class to access private members of KDTree
  public:
    /*
    Build the KDTree for the target point cloud
    @param target: The target point cloud to build the KDTree for
    @param stream: CUDA stream to use for the computation
    */
    void buildTree(std::vector<float3> const &target, cudaStream_t stream);

    /*
    Find the nearest neighbor for each point in source
    @param source: The source point cloud
    @param inlierThreshold: The maximum distance to consider a point as an inlier, if inlierThreshold is
                           non-positive, it will be ignored
    @param stream: CUDA stream to use for finding nearest neighbors
    @return: A vector of distances to the nearest neighbor for each point in source, if not found, the
    distance will be set to -1.0f
    */
    std::vector<float> findAllNearestDistance(std::vector<float3> const &source, float inlierThreshold,
                                              cudaStream_t stream);

  private:
    /*
    This class is used for ICP purpose only, it is not a general KDTree class
    @param d_source: The source point cloud
    @param n_source: The number of points in the source point cloud
    @param inlierThreshold: The maximum distance to consider a point as an inlier, if inlierThreshold is
                           non-positive, it will be ignored
    @param inlier: The output array to store the inliers (n_source: not inlier, <n_source: inlier)
    @param dsrc, dtar: The output arrays to store the source and target point coordinates, if not found the
                          closest point will be set to 0.0f
    @param stream: CUDA stream to use for finding nearest neighbors
    */
    void findCorrespondences(float3 const *d_source, uint32_t n_source, float inlierThreshold,
                             uint32_t *inlier, float3 *dsrc, float3 *dtar, cudaStream_t stream);

  private:
    uint32_t n_target;
    thrust::device_vector<float3> d_target;
};
