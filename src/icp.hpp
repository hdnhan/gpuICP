#pragma once

#include "kdtree.hpp"

class ICP {
  public:
    void setTarget(std::vector<float3> const &target, cudaStream_t stream);
    void align(std::vector<float3> const &source, float maxCorrespondenceDistance, int maximumIterations,
               float transformationEpsilon, float euclideanFitnessEpsilon, cudaStream_t stream);

  private:
    KDTree kdtree;
};