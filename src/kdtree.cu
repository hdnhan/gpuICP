#include "kdtree.hpp"
#include <thrust/iterator/zip_iterator.h> // make_zip_iterator
#include <thrust/sort.h>                  // stable_sort
#include <thrust/tuple.h>                 // make_tuple

/*
The final tree structure should be like this:

level 0:            0
                 /     \
level 1:       1         2
             /  \       /  \
level 2:    3    4     5    6
           / \  / \   / \  / \
level 3:  7  8 9 10 11 12 13 14

As we construct the tree by each level from top to bottom,
treeIDs at level 2 should be: [0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
IDRange[0] = [0, 1)
IDRange[1] = [1, 2)
IDRange[2] = [2, 3)
IDRange[3] = [3, 6)
IDRange[4] = [6, 9)
IDRange[5] = [9, 12)
IDRange[6] = [12, 15)
*/

inline __device__ uint32_t lowerBound(uint32_t const *data, uint32_t size, uint32_t value) {
    uint32_t left = 0, right = size;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (data[mid] < value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

inline __device__ uint32_t upperBound(uint32_t const *data, uint32_t size, uint32_t value) {
    uint32_t left = 0, right = size;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (data[mid] <= value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

__global__ void searchIDRange(uint32_t *treeIDs, uint32_t size, uint2 *IDRange, uint32_t level) {
    // start id (inclusive): 2^level - 1
    // end id (exclusive): 2^(level + 1) - 1
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size || idx < (1 << level) - 1 || idx >= (2 << level) - 1)
        return;
    uint32_t id = idx;

    uint32_t start = lowerBound(treeIDs, size, id);
    uint32_t end = upperBound(treeIDs, size, id);
    IDRange[id - (1 << level) + 1].x = start;
    IDRange[id - (1 << level) + 1].y = end;
}

__global__ void updateTreeID(uint32_t *treeIDs, uint32_t size, uint2 *IDRange, uint32_t level) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size || idx < (1 << level) - 1)
        return;

    uint32_t id = treeIDs[idx];
    uint32_t start = IDRange[id - (1 << level) + 1].x;
    uint32_t end = IDRange[id - (1 << level) + 1].y;

    uint32_t mid = (start + end) / 2;
    if (idx < mid)
        treeIDs[idx] = 2 * id + 1; // left subtree
    else if (idx > mid)
        treeIDs[idx] = 2 * id + 2; // right subtree
    // idx == mid => it's the root of the subtree
}

// Not using sqrt to make it faster
inline __device__ float estimateDistance(float3 const &a, float3 const &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

inline __device__ float getDifference(float3 const &point, float3 const &target, uint32_t axis) {
    if (axis == 0)
        return point.x - target.x;
    else if (axis == 1)
        return point.y - target.y;
    else
        return point.z - target.z;
}

__device__ int32_t findNearestPoint(float3 const &point, float3 const *d_target, uint32_t n_target,
                                    float inlierThreshold) {
    // TODO: Assume maximum level is 32, need to check if it is enough
    uint32_t stack[32]; // store tree id of the secondary node (left or right)
    uint32_t idx = 0, currID = 0;
    int32_t bestID = -1;
    float bestDistance = inlierThreshold * inlierThreshold;
    if (inlierThreshold <= 0) {
        bestID = 0;
        bestDistance = estimateDistance(point, d_target[0]);
    }

    uint32_t n_level = ceil(log2f(n_target + 1));
    uint32_t maxPoints = (1 << n_level) - 1;
    uint32_t missingPointsStart = 0, missingPointsEnd = maxPoints - n_target;
    uint2 missingRange[32];
    uint32_t level = 0;
    float diff, currDistance;

    // Check if currID is a valid point
    while (true) {
        // try to move down to the tree (children) given the current id
        while (currID < maxPoints && !(level == n_level - 1 && missingPointsEnd != missingPointsStart)) {
            if (level == n_level - 1)
                currDistance = estimateDistance(point, d_target[currID - missingPointsStart]);
            else
                currDistance = estimateDistance(point, d_target[currID]);

            if (currDistance < bestDistance) {
                bestDistance = currDistance;
                if (level == n_level - 1)
                    bestID = currID - missingPointsStart;
                else
                    bestID = currID;
            }

            level = log2f(currID + 1);
            uint32_t axis = level % 3;
            if (level == n_level - 1)
                diff = getDifference(point, d_target[currID - missingPointsStart], axis);
            else
                diff = getDifference(point, d_target[currID], axis);

            uint32_t firstID, secondID;
            uint32_t firstMissingPointsStart, firstMissingPointsEnd;
            uint32_t secondMissingPointsStart, secondMissingPointsEnd;
            // [left...root...right]
            if (diff < 0) {
                firstID = 2 * currID + 1;
                secondID = 2 * currID + 2;
                firstMissingPointsStart = missingPointsStart;
                firstMissingPointsEnd = (missingPointsStart + missingPointsEnd) / 2;
                secondMissingPointsStart = firstMissingPointsEnd;
                secondMissingPointsEnd = missingPointsEnd;
            } else {
                firstID = 2 * currID + 2;
                secondID = 2 * currID + 1;
                firstMissingPointsStart = (missingPointsStart + missingPointsEnd) / 2;
                firstMissingPointsEnd = missingPointsEnd;
                secondMissingPointsStart = missingPointsStart;
                secondMissingPointsEnd = firstMissingPointsStart;
            }

            if (secondID < maxPoints && diff * diff < bestDistance) {
                stack[idx] = secondID;
                missingRange[idx].x = secondMissingPointsStart;
                missingRange[idx].y = secondMissingPointsEnd;
                idx++;
            }
            currID = firstID;
            missingPointsStart = firstMissingPointsStart;
            missingPointsEnd = firstMissingPointsEnd;
            level++;
        }
        // move up to the tree (parent or sibling)
        bool canMove = false;
        while (idx > 0) {
            currID = stack[--idx];
            uint32_t parentID = (currID - 1) / 2;
            level = log2f(parentID + 1);
            uint32_t axis = level % 3;
            // NOTE: Parent ID is always valid
            diff = getDifference(point, d_target[parentID], axis);
            if (diff * diff < bestDistance) {
                canMove = true;
                level++;
                missingPointsStart = missingRange[idx].x;
                missingPointsEnd = missingRange[idx].y;
                break;
            }
        }
        if (!canMove) {
            // No more points to check
            break;
        }
    }
    return bestID;
}

__global__ void findAllNearestDistanceKernel(float3 *d_source, uint32_t n_source, float3 *d_target,
                                             uint32_t n_target, float *d_distance, float inlierThreshold) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_source)
        return;
    float3 point = d_source[idx];
    int32_t id = findNearestPoint(point, d_target, n_target, inlierThreshold);
    // There is no point in the tree that is in the inlier threshold
    if (id == -1)
        d_distance[idx] = -1;
    else
        d_distance[idx] = sqrt(estimateDistance(point, d_target[id]));
}

inline __host__ __device__ float getAxisValue(float3 const &point, uint32_t axis) {
    if (axis == 0)
        return point.x;
    else if (axis == 1)
        return point.y;
    else
        return point.z;
}

struct Comparator {
    __host__ __device__ Comparator(uint32_t axis) : axis(axis) {}
    inline __host__ __device__ bool operator()(thrust::tuple<uint32_t, float3> const &a,
                                               thrust::tuple<uint32_t, float3> const &b) {
        // Group by id first, then sort by value
        auto idA = thrust::get<0>(a);
        auto idB = thrust::get<0>(b);
        if (idA != idB)
            return idA < idB;
        auto pointA = thrust::get<1>(a);
        auto pointB = thrust::get<1>(b);
        return getAxisValue(pointA, axis) < getAxisValue(pointB, axis);
    }

  private:
    uint32_t axis;
};

KDTree::KDTree(std::vector<float3> target, cudaStream_t stream) {
    n_target = target.size();
    uint32_t n_level = ceil(log2(n_target + 1));

    // Assuming all points are root of the tree
    thrust::device_vector<uint32_t> treeIDs(n_target, 0);
    d_target = thrust::device_vector<float3>(n_target);
    cudaMemcpy(thrust::raw_pointer_cast(d_target.data()), target.data(), target.size() * sizeof(float3),
               cudaMemcpyHostToDevice);

    // Create the zip iterators for sorting
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(treeIDs.begin(), d_target.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(treeIDs.end(), d_target.end()));

    uint32_t blockSize = 1 << 8;
    uint32_t numBlocks = (n_target + blockSize - 1) / blockSize;
    // [start, end) of each level, to optimize:
    // - only reserve space for second last level
    // - subtract all ids of the current level to the smallest one [2^x - 1, 2^(x + 1) - 1) => [0, 2^x)
    thrust::device_vector<uint2> IDRange(1 << (n_level - 2));

    auto *treeIDsPtr = thrust::raw_pointer_cast(treeIDs.data());
    auto *IDRangePtr = thrust::raw_pointer_cast(IDRange.data());
    for (uint32_t level = 0; level < n_level - 1; ++level) {
        thrust::stable_sort(thrust::device.on(stream), begin, end, Comparator(level % 3));
        cudaStreamSynchronize(stream);
        searchIDRange<<<numBlocks, blockSize, 0, stream>>>(treeIDsPtr, n_target, IDRangePtr, level);
        cudaStreamSynchronize(stream);
        updateTreeID<<<numBlocks, blockSize, 0, stream>>>(treeIDsPtr, n_target, IDRangePtr, level);
        cudaStreamSynchronize(stream);
    }
    thrust::stable_sort(thrust::device.on(stream), begin, end, Comparator(n_level % 3));
    cudaStreamSynchronize(stream);
}

std::vector<float> KDTree::findAllNearestDistance(std::vector<float3> source, float inlierThreshold,
                                                  cudaStream_t stream) {
    uint32_t n_source = source.size();
    thrust::device_vector<float3> d_sources(source.data(), source.data() + n_source);
    thrust::device_vector<float> d_distance(n_source, -1);
    uint32_t blockSize = 1 << 8;
    uint32_t numBlocks = (n_source + blockSize - 1) / blockSize;
    findAllNearestDistanceKernel<<<numBlocks, blockSize, 0, stream>>>(
        thrust::raw_pointer_cast(d_sources.data()), n_source, thrust::raw_pointer_cast(d_target.data()),
        n_target, thrust::raw_pointer_cast(d_distance.data()), inlierThreshold);
    cudaStreamSynchronize(stream);
    std::vector<float> estimateDistance(n_source);
    cudaMemcpy(estimateDistance.data(), thrust::raw_pointer_cast(d_distance.data()), n_source * sizeof(float),
               cudaMemcpyDeviceToHost);
    return estimateDistance;
}
