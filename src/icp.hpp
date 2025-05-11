#pragma once

#include "kdtree.hpp"

class ICP {
  public:
    /*
    To build KDTree for the target point cloud
    @param target: The target point cloud to build the KDTree for
    @param stream: CUDA stream to use for the computation
     */
    void setTarget(std::vector<float3> const &target, cudaStream_t stream);

    /*
    Align the source point cloud to the target point cloud
    @param source: The source point cloud to align
    @param maxCorrespondenceDistance: Maximum distance for a point to be considered a correspondence
    @param maximumIterations: Maximum number of iterations to run
    @param transformationEpsilon: Minimum change in transformation to be considered converged
    @param euclideanFitnessEpsilon: Minimum change in fitness score to be considered converged
    @param Rt: Initial guess for the transformation matrix (4x4), it will also be the final transformation
    @param stream: CUDA stream to use for the computation
    @return: A tuple containing:
        - bool: Whether the algorithm converged
        - float: Percentage of points in the source point cloud that are within maxCorrespondenceDistance
    */
    std::tuple<bool, float> align(std::vector<float3> const &source, float maxCorrespondenceDistance,
                                  int maximumIterations, float transformationEpsilon,
                                  float euclideanFitnessEpsilon, std::vector<float> &Rt, cudaStream_t stream);

  private:
    KDTree kdtree;
};