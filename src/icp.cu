#include "icp.hpp"
#include "svd.hpp"
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

void ICP::setTarget(std::vector<float3> const &target, cudaStream_t stream) {
    kdtree.buildTree(target, stream);
}

// Add operator for float3
inline __host__ __device__ float3 operator+(float3 const &a, float3 const &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline __host__ __device__ float3 operator-(float3 const &a, float3 const &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline __host__ __device__ float3 operator*(float3 const &a, float3 const &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

struct SubtractFunctor {
    __host__ __device__ SubtractFunctor(float3 const &value) : value(value) {}
    inline __host__ __device__ float3 operator()(float3 x) const { return x - value; }

  private:
    float3 value;
};

struct BinaryOp1 {
    inline __host__ __device__ thrust::tuple<float3, float3, float3>
    operator()(thrust::tuple<float3, float3, float3> const &a,
               thrust::tuple<float3, float3, float3> const &b) {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b),
                                  thrust::get<2>(a) + thrust::get<2>(b));
    };
};

struct BinaryOp2 {
    inline __host__ __device__ thrust::tuple<float3, float3, float3> operator()(float3 const &a,
                                                                                float3 const &b) {
        return thrust::make_tuple(make_float3(a.x * b.x, a.x * b.y, a.x * b.z),
                                  make_float3(a.y * b.x, a.y * b.y, a.y * b.z),
                                  make_float3(a.z * b.x, a.z * b.y, a.z * b.z));
    };
};

// in-place operation
__global__ void applyTransformation(float3 *source, uint32_t n_source, float *R, float *t) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_source)
        return;
    // R: 3x3 matrix
    // t: 3x1 vector
    float3 s = source[idx];
    float3 r;
    r.x = R[0] * s.x + R[1] * s.y + R[2] * s.z + t[0];
    r.y = R[3] * s.x + R[4] * s.y + R[5] * s.z + t[1];
    r.z = R[6] * s.x + R[7] * s.y + R[8] * s.z + t[2];
    source[idx] = r;
}

// in-place operation, this is for estimating error
__global__ void applyTransformation2(float3 *source, float3 *target, uint32_t start, uint32_t end, float *R,
                                     float *t) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start || idx >= end)
        return;
    // Transform the source points (dsx, dsy, dsz)
    // R: 3x3 matrix
    // t: 3x1 vector
    float3 s = {R[0] * source[idx].x + R[1] * source[idx].y + R[2] * source[idx].z + t[0],
                R[3] * source[idx].x + R[4] * source[idx].y + R[5] * source[idx].z + t[1],
                R[6] * source[idx].x + R[7] * source[idx].y + R[8] * source[idx].z + t[2]};
    source[idx] = (s - target[idx]) * (s - target[idx]);
}

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
    thrust::device_vector<uint32_t> inlier(n_source, 0);

    // Allocate cuda memory for the source and target points
    thrust::device_vector<float3> dsrc(n_source), dtar(n_source);

    // For gathering the inliers
    thrust::device_vector<float3> gsrc(n_source), gtar(n_source);

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

    cudaMemcpy(thrust::raw_pointer_cast(dR.data()), gR.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(dt.data()), gt.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
    // Apply the initial transformation
    uint32_t blockSize = 1 << 8;
    uint32_t numBlocks = (n_source + blockSize - 1) / blockSize;
    applyTransformation<<<numBlocks, blockSize, 0, stream>>>(thrust::raw_pointer_cast(d_source.data()),
                                                             n_source, thrust::raw_pointer_cast(dR.data()),
                                                             thrust::raw_pointer_cast(dt.data()));
    cudaStreamSynchronize(stream);

    float prevError = std::numeric_limits<float>::max();
    bool converged = false;
    float percentageInliers = 0.0f;

    auto policy = thrust::device.on(stream);
    for (int i = 0; i < maximumIterations; ++i) {
        // 1. Find correspondences
        kdtree.findCorrespondences(thrust::raw_pointer_cast(d_source.data()), n_source,
                                   maxCorrespondenceDistance, thrust::raw_pointer_cast(inlier.data()),
                                   thrust::raw_pointer_cast(dsrc.data()),
                                   thrust::raw_pointer_cast(dtar.data()), stream);
        cudaStreamSynchronize(stream);

        // 2. Compute centroids
        // in-place scan: inlier[i] += inlier[i-1]
        thrust::inclusive_scan(policy, inlier.begin(), inlier.end(), inlier.begin());
        int32_t count; // number of inliers
        thrust::copy(inlier.end() - 1, inlier.end(), &count);
        if (count < 2)
            break;     // no inliers
        int32_t start; // start of inliers
        thrust::copy(inlier.begin(), inlier.begin() + 1, &start);
        start = 1 - start;

        // move all inliers to the front
        // TODO: can we do this in-place? if yes, remove gsrc and gtar
        thrust::gather(policy, inlier.begin(), inlier.end(), dsrc.begin(), gsrc.begin());
        thrust::gather(policy, inlier.begin(), inlier.end(), dtar.begin(), gtar.begin());
        cudaStreamSynchronize(stream);
        // compute centroids
        float3 csrc = thrust::reduce(policy, gsrc.begin() + start, gsrc.begin() + count + start,
                                     float3{0.0f, 0.0f, 0.0f});
        float3 ctar = thrust::reduce(policy, gtar.begin() + start, gtar.begin() + count + start,
                                     float3{0.0f, 0.0f, 0.0f});
        cudaStreamSynchronize(stream);
        csrc = {csrc.x / count, csrc.y / count, csrc.z / count};
        ctar = {ctar.x / count, ctar.y / count, ctar.z / count};

        // 3. Center the points, in-place operation
        thrust::transform(policy, gsrc.begin() + start, gsrc.begin() + count + start, gsrc.begin() + start,
                          SubtractFunctor(csrc));
        thrust::transform(policy, gtar.begin() + start, gtar.begin() + count + start, gtar.begin() + start,
                          SubtractFunctor(ctar));
        cudaStreamSynchronize(stream);

        // 4. Compute the covariance matrix
        auto cov = thrust::inner_product(
            policy, gsrc.begin() + start, gsrc.begin() + count + start, gtar.begin() + start,
            thrust::make_tuple(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f),
                               make_float3(0.0f, 0.0f, 0.0f)),
            BinaryOp1(), BinaryOp2());
        cudaStreamSynchronize(stream);
        std::vector<float> H{thrust::get<0>(cov).x, thrust::get<0>(cov).y, thrust::get<0>(cov).z,
                             thrust::get<1>(cov).x, thrust::get<1>(cov).y, thrust::get<1>(cov).z,
                             thrust::get<2>(cov).x, thrust::get<2>(cov).y, thrust::get<2>(cov).z};

        // 5. Compute Rotation and translation using SVD
        auto [R, t] = computeRt(H, csrc.x, csrc.y, csrc.z, ctar.x, ctar.y, ctar.z);
        cudaMemcpy(thrust::raw_pointer_cast(dR.data()), R.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(dt.data()), t.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        auto [nR, nt] = getTranformation(gR, gt, R, t);
        float error = getTranformationError(gR, gt, nR, nt);
        if (error < transformationEpsilon) {
            converged = true;
            percentageInliers = (float)count / n_source;
            break; // converged
        }
        gR = nR;
        gt = nt;

        // 6. Apply the transformation to the source points
        applyTransformation<<<numBlocks, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(d_source.data()), n_source, thrust::raw_pointer_cast(dR.data()),
            thrust::raw_pointer_cast(dt.data()));
        cudaStreamSynchronize(stream);

        // 7. Compute the Euclidean distance error
        applyTransformation2<<<numBlocks, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(gsrc.data()), thrust::raw_pointer_cast(gtar.data()), start,
            count + start, thrust::raw_pointer_cast(dR.data()), thrust::raw_pointer_cast(dt.data()));
        cudaStreamSynchronize(stream);
        float3 error2 = thrust::reduce(policy, gsrc.begin() + start, gsrc.begin() + count + start,
                                       make_float3(0.0f, 0.0f, 0.0f));
        cudaStreamSynchronize(stream);
        float error3 = sqrt(error2.x + error2.y + error2.z);
        if (abs(error3 - prevError) < euclideanFitnessEpsilon) {
            converged = true;
            percentageInliers = (float)count / n_source;
            break; // converged
        }
        prevError = error3;
    }
    // Copy the final transformation matrix to the output
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            Rt[i * 4 + j] = gR[3 * i + j];
        Rt[i * 4 + 3] = gt[i];
    }
    return {converged, percentageInliers};
}