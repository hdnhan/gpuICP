#include "icp.hpp"
#include "svd.hpp"
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>

void ICP::setTarget(std::vector<float3> const &target, cudaStream_t stream) {
    kdtree.buildTree(target, stream);
}

// Add operator for float3
__host__ __device__ inline float3 operator+(float3 const &a, float3 const &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline float3 operator-(float3 const &a, float3 const &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
#ifdef CUDA_FOUND
__host__ __device__ inline float3 operator*(float3 const &a, float3 const &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
#endif

struct SumFunctor {
    __host__ __device__ inline thrust::tuple<float3, float3>
    operator()(thrust::tuple<float3, float3> const &x, thrust::tuple<float3, float3> const &y) const {
        return thrust::make_tuple(thrust::get<0>(x) + thrust::get<0>(y),
                                  thrust::get<1>(x) + thrust::get<1>(y));
    }
};

struct PartitionLess {
    __host__ __device__ inline bool operator()(bool const x) { return x; }
};

/*
BinaryOp1 and BinaryOp2 are used in thrust::inner_product
for example:
    - x is first array
    - y is second array
    - init is the initial value
=> init op1 (x1 op2 y1) op1 (x2 op2 y2) op1 ... op1 (xn op2 yn)
*/
struct BinaryOp1 {
    __host__ __device__ inline thrust::tuple<float3, float3, float3>
    operator()(thrust::tuple<float3, float3, float3> const &a,
               thrust::tuple<float3, float3, float3> const &b) {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b),
                                  thrust::get<2>(a) + thrust::get<2>(b));
    };
};

struct BinaryOp2 {
    __host__ __device__ BinaryOp2(float3 const &csrc, float3 const &ctar) : csrc(csrc), ctar(ctar) {}
    __host__ __device__ inline thrust::tuple<float3, float3, float3> operator()(float3 const &src,
                                                                                float3 const &tar) {
        float3 a = src - csrc, b = tar - ctar;
        return thrust::make_tuple(make_float3(a.x * b.x, a.x * b.y, a.x * b.z),
                                  make_float3(a.y * b.x, a.y * b.y, a.y * b.z),
                                  make_float3(a.z * b.x, a.z * b.y, a.z * b.z));
    };

  private:
    float3 csrc, ctar; // centroids
};

// Transformation: Points * R + t
// in-place operation
struct TransformFunctor {
    __host__ __device__ TransformFunctor(float *R, float *t) : R(R), t(t) {}
    __host__ __device__ inline void operator()(float3 &p) const {
        float3 transformed = {R[0] * p.x + R[1] * p.y + R[2] * p.z + t[0],
                              R[3] * p.x + R[4] * p.y + R[5] * p.z + t[1],
                              R[6] * p.x + R[7] * p.y + R[8] * p.z + t[2]};
        p = transformed;
    }

  private:
    float *R; // 3x3 matrix
    float *t; // 3x1 vector
};

// Euclidean distance error
struct EuclideanDistanceFunctor {
    __host__ __device__ EuclideanDistanceFunctor(float *R, float *t) : R(R), t(t) {}
    __host__ __device__ inline float operator()(thrust::tuple<float3, float3> const &p) const {
        float3 src = thrust::get<0>(p);
        float3 tar = thrust::get<1>(p);
        float3 transformed = {R[0] * src.x + R[1] * src.y + R[2] * src.z + t[0],
                              R[3] * src.x + R[4] * src.y + R[5] * src.z + t[1],
                              R[6] * src.x + R[7] * src.y + R[8] * src.z + t[2]};
        // Compute the Euclidean distance
        float3 dist = (transformed - tar) * (transformed - tar);
        return dist.x + dist.y + dist.z;
    }

  private:
    float *R; // 3x3 matrix
    float *t; // 3x1 vector
};

// Get the next transformation
// gR, R: 3x3 matrix (global and current rotation)
// gt, t: 3x1 vector (global and current translation)
std::tuple<std::vector<float>, std::vector<float>> getTranformation(std::vector<float> const &gR,
                                                                    std::vector<float> const &gt,
                                                                    std::vector<float> const &R,
                                                                    std::vector<float> const &t) {
    // Next transformation
    std::vector<float> nR(9, 0); // R * gR
    std::vector<float> nt(3, 0); // R * gt + t

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            nR[3 * i + j] += R[3 * i] * gR[j] + R[3 * i + 1] * gR[j + 3] + R[3 * i + 2] * gR[j + 6];
        nt[i] += R[3 * i] * gt[0] + R[3 * i + 1] * gt[1] + R[3 * i + 2] * gt[2] + t[i];
    }
    return {nR, nt};
}

// Frobenius norm: ||A||_F = sqrt(Trace(A * A^T))
float getTranformationError(std::vector<float> const &gR, std::vector<float> const &gt,
                            std::vector<float> const &nR, std::vector<float> const &nt) {
    float error = 0.0f;
    for (int i = 0; i < 9; ++i)
        error += (nR[i] - gR[i]) * (nR[i] - gR[i]);
    for (int i = 0; i < 3; ++i)
        error += (nt[i] - gt[i]) * (nt[i] - gt[i]);
    return sqrt(error);
}

// Reference: https://learnopencv.com/iterative-closest-point-icp-explained/
std::tuple<bool, float> ICP::align(std::vector<float3> const &source, float maxCorrespondenceDistance,
                                   int maximumIterations, float transformationEpsilon,
                                   float euclideanFitnessEpsilon, std::vector<float> &Rt,
                                   cudaStream_t stream) {
    uint32_t n_source = source.size();
    thrust::device_vector<float3> d_source(source.begin(), source.end());
    thrust::device_vector<bool> inlier(n_source, false);

    // Allocate cuda memory for the source and target points
    thrust::device_vector<float3> dsrc(n_source), dtar(n_source);

    // R and t
    std::vector<float> gR(9, 0.0f);
    std::vector<float> gt(3, 0.0f);
    // Rt 4x4 matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            gR[3 * i + j] = Rt[i * 4 + j];
        gt[i] = Rt[i * 4 + 3];
    }
    thrust::device_vector<float> dR(9);
    thrust::device_vector<float> dt(3);

    GPU_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dR.data()), gR.data(), 9 * sizeof(float),
                         cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dt.data()), gt.data(), 3 * sizeof(float),
                         cudaMemcpyHostToDevice));
    // Apply the initial transformation
    auto policy = thrust::device.on(stream);
    thrust::for_each(
        policy, d_source.begin(), d_source.end(),
        TransformFunctor(thrust::raw_pointer_cast(dR.data()), thrust::raw_pointer_cast(dt.data())));

    float prevError = std::numeric_limits<float>::max();
    bool converged = false;
    float percentageInliers = 0.0f;

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(dsrc.begin(), dtar.begin()));
    for (int i = 0; i < maximumIterations; ++i) {
        // 1. Find correspondences
        kdtree.findCorrespondences(thrust::raw_pointer_cast(d_source.data()), n_source,
                                   maxCorrespondenceDistance, thrust::raw_pointer_cast(inlier.data()),
                                   thrust::raw_pointer_cast(dsrc.data()),
                                   thrust::raw_pointer_cast(dtar.data()), stream);
        GPU_CHECK(cudaStreamSynchronize(stream));

        // 2. Compute centroids
        // move all inliers to the front
        auto it = thrust::partition(policy, begin, begin + n_source, inlier.begin(), PartitionLess());
        uint32_t count = thrust::distance(begin, it); // number of inliers
        if (count < 2)
            break; // no inliers

        // compute centroids
        auto centroids = thrust::reduce(
            policy, begin, begin + count,
            thrust::make_tuple(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f)), SumFunctor());
        auto csrc = thrust::get<0>(centroids);
        auto ctar = thrust::get<1>(centroids);
        csrc = {csrc.x / count, csrc.y / count, csrc.z / count};
        ctar = {ctar.x / count, ctar.y / count, ctar.z / count};

        // 3. Compute the cross-covariance matrix
        auto cov = thrust::inner_product(policy, dsrc.begin(), dsrc.begin() + count, dtar.begin(),
                                         thrust::make_tuple(make_float3(0.0f, 0.0f, 0.0f),
                                                            make_float3(0.0f, 0.0f, 0.0f),
                                                            make_float3(0.0f, 0.0f, 0.0f)),
                                         BinaryOp1(), BinaryOp2(csrc, ctar));
        std::vector<float> H{thrust::get<0>(cov).x, thrust::get<0>(cov).y, thrust::get<0>(cov).z,
                             thrust::get<1>(cov).x, thrust::get<1>(cov).y, thrust::get<1>(cov).z,
                             thrust::get<2>(cov).x, thrust::get<2>(cov).y, thrust::get<2>(cov).z};

        // 4. Compute Rotation and translation using SVD
        auto [R, t] = computeRt(H, csrc.x, csrc.y, csrc.z, ctar.x, ctar.y, ctar.z);
        GPU_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dR.data()), R.data(), 9 * sizeof(float),
                             cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dt.data()), t.data(), 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        auto [nR, nt] = getTranformation(gR, gt, R, t);
        float error = getTranformationError(gR, gt, nR, nt);
        gR = nR, gt = nt;
        if (error < transformationEpsilon) {
            converged = true;
            percentageInliers = (float)count / n_source;
            break; // converged
        }

        // 5. Apply the transformation to the source points
        thrust::for_each(
            policy, d_source.begin(), d_source.end(),
            TransformFunctor(thrust::raw_pointer_cast(dR.data()), thrust::raw_pointer_cast(dt.data())));

        // 6. Compute the Euclidean distance error
        error = thrust::transform_reduce(policy, begin, begin + count,
                                         EuclideanDistanceFunctor(thrust::raw_pointer_cast(dR.data()),
                                                                  thrust::raw_pointer_cast(dt.data())),
                                         0.0f, thrust::plus<float>());
        error = sqrt(error);
        if (abs(error - prevError) < euclideanFitnessEpsilon) {
            converged = true;
            percentageInliers = (float)count / n_source;
            break; // converged
        }
        prevError = error;
    }
    // Copy the final transformation matrix to the output
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            Rt[i * 4 + j] = gR[3 * i + j];
        Rt[i * 4 + 3] = gt[i];
    }
    return {converged, percentageInliers};
}