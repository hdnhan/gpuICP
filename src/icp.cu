#include "icp.hpp"
#include "svd.hpp"
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

void ICP::setTarget(std::vector<float3> const &target, cudaStream_t stream) {
    kdtree.buildTree(target, stream);
}

struct SubtractFunctor {
    __host__ __device__ SubtractFunctor(float value) : value(value) {}
    inline __host__ __device__ float operator()(float x) const { return x - value; }

  private:
    float value;
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
void ICP::align(std::vector<float3> const &source, float maxCorrespondenceDistance, int maximumIterations,
                float transformationEpsilon, float euclideanFitnessEpsilon, cudaStream_t stream) {
    uint32_t n_source = source.size();
    thrust::device_vector<float3> d_source(source.begin(), source.end());
    thrust::device_vector<uint32_t> inlier(n_source, 0);

    // Allocate cuda memory for the source and target points
    thrust::device_vector<float> dsx(n_source), dsy(n_source), dsz(n_source);
    thrust::device_vector<float> dtx(n_source), dty(n_source), dtz(n_source);

    // For gathering the inliers
    thrust::device_vector<float> gsx(n_source), gsy(n_source), gsz(n_source);
    thrust::device_vector<float> gtx(n_source), gty(n_source), gtz(n_source);

    // R and t
    std::vector<float> gR(9, 0.0f);
    gR[0] = 1.0f, gR[4] = 1.0f, gR[8] = 1.0f;
    std::vector<float> gt(3, 0.0f);
    thrust::device_vector<float> dR(9);
    thrust::device_vector<float> dt(3);

    // float prevError = std::numeric_limits<float>::max();

    auto policy = thrust::cuda::par.on(stream);
    for (int i = 0; i < maximumIterations; ++i) {
        // 1. Find correspondences
        kdtree.findCorrespondences(thrust::raw_pointer_cast(d_source.data()), n_source,
                                   maxCorrespondenceDistance, thrust::raw_pointer_cast(inlier.data()),
                                   thrust::raw_pointer_cast(dsx.data()), thrust::raw_pointer_cast(dsy.data()),
                                   thrust::raw_pointer_cast(dsz.data()), thrust::raw_pointer_cast(dtx.data()),
                                   thrust::raw_pointer_cast(dty.data()), thrust::raw_pointer_cast(dtz.data()),
                                   stream);
        cudaStreamSynchronize(stream);

        // 2. Compute centroids
        // in-place scan: inlier[i] += inlier[i-1]
        thrust::inclusive_scan(thrust::device.on(stream), inlier.begin(), inlier.end(), inlier.begin());
        int32_t count; // number of inliers
        thrust::copy(inlier.end() - 1, inlier.end(), &count);
        if (count < 2) {
            break; // no inliers
        }
        int32_t start; // start of inliers
        thrust::copy(inlier.begin(), inlier.begin() + 1, &start);
        start = 1 - start;

        // move all inliers to the front
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dsx.begin(), gsx.begin());
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dsy.begin(), gsy.begin());
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dsz.begin(), gsz.begin());
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dtx.begin(), gtx.begin());
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dty.begin(), gty.begin());
        thrust::gather(thrust::device.on(stream), inlier.begin(), inlier.end(), dtz.begin(), gtz.begin());
        cudaStreamSynchronize(stream);
        // compute centroids
        float csx = thrust::reduce(policy, gsx.begin() + start, gsx.begin() + count + start, 0.0f) / count;
        float csy = thrust::reduce(policy, gsy.begin() + start, gsy.begin() + count + start, 0.0f) / count;
        float csz = thrust::reduce(policy, gsz.begin() + start, gsz.begin() + count + start, 0.0f) / count;
        float ctx = thrust::reduce(policy, gtx.begin() + start, gtx.begin() + count + start, 0.0f) / count;
        float cty = thrust::reduce(policy, gty.begin() + start, gty.begin() + count + start, 0.0f) / count;
        float ctz = thrust::reduce(policy, gtz.begin() + start, gtz.begin() + count + start, 0.0f) / count;
        cudaStreamSynchronize(stream);

        // 3. Center the points
        thrust::transform(thrust::device.on(stream), gsx.begin() + start, gsx.begin() + count + start,
                          gsx.begin() + start, SubtractFunctor(csx));
        thrust::transform(thrust::device.on(stream), gsy.begin() + start, gsy.begin() + count + start,
                          gsy.begin() + start, SubtractFunctor(csy));
        thrust::transform(thrust::device.on(stream), gsz.begin() + start, gsz.begin() + count + start,
                          gsz.begin() + start, SubtractFunctor(csz));
        thrust::transform(thrust::device.on(stream), gtx.begin() + start, gtx.begin() + count + start,
                          gtx.begin() + start, SubtractFunctor(ctx));
        thrust::transform(thrust::device.on(stream), gty.begin() + start, gty.begin() + count + start,
                          gty.begin() + start, SubtractFunctor(cty));
        thrust::transform(thrust::device.on(stream), gtz.begin() + start, gtz.begin() + count + start,
                          gtz.begin() + start, SubtractFunctor(ctz));
        cudaStreamSynchronize(stream);

        // 4. Compute the covariance matrix
        std::vector<float> H(9);
        H[0] = thrust::inner_product(gsx.begin() + start, gsx.begin() + count + start, gtx.begin() + start,
                                     0.0f);
        H[1] = thrust::inner_product(gsx.begin() + start, gsx.begin() + count + start, gty.begin() + start,
                                     0.0f);
        H[2] = thrust::inner_product(gsx.begin() + start, gsx.begin() + count + start, gtz.begin() + start,
                                     0.0f);

        H[3] = thrust::inner_product(gsy.begin() + start, gsy.begin() + count + start, gtx.begin() + start,
                                     0.0f);
        H[4] = thrust::inner_product(gsy.begin() + start, gsy.begin() + count + start, gty.begin() + start,
                                     0.0f);
        H[5] = thrust::inner_product(gsy.begin() + start, gsy.begin() + count + start, gtz.begin() + start,
                                     0.0f);

        H[6] = thrust::inner_product(gsz.begin() + start, gsz.begin() + count + start, gtx.begin() + start,
                                     0.0f);
        H[7] = thrust::inner_product(gsz.begin() + start, gsz.begin() + count + start, gty.begin() + start,
                                     0.0f);
        H[8] = thrust::inner_product(gsz.begin() + start, gsz.begin() + count + start, gtz.begin() + start,
                                     0.0f);
        cudaStreamSynchronize(stream);

        // 5. Compute Rotation and translation using SVD
        auto [R, t] = computeRt(H, csx, csy, csz, ctx, cty, ctz);
        cudaMemcpy(thrust::raw_pointer_cast(dR.data()), R.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(dt.data()), t.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
        auto [nR, nt] = getTranformation(gR, gt, R, t);
        float error = getTranformationError(gR, gt, nR, nt);
        if (error < transformationEpsilon) {
            break; // converged
        }
        gR = nR;
        gt = nt;

        // 6. Apply the transformation
        uint32_t blockSize = 1 << 8;
        uint32_t numBlocks = (n_source + blockSize - 1) / blockSize;
        applyTransformation<<<numBlocks, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(d_source.data()), n_source, thrust::raw_pointer_cast(dR.data()),
            thrust::raw_pointer_cast(dt.data()));
        cudaStreamSynchronize(stream);

        // TODO: Implement the fitness error (euclideanFitnessEpsilon)
    }
    // Print gR and gt
    std::cout << "gR: ";
    for (int i = 0; i < 9; ++i)
        std::cout << gR[i] << " ";
    std::cout << "\ngt: ";
    for (int i = 0; i < 3; ++i)
        std::cout << gt[i] << " ";
    std::cout << "\n";
}