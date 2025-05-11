#include "svd.hpp"
#include <Eigen/Dense>

std::tuple<std::vector<float>, std::vector<float>> computeRt(std::vector<float> const &H, float sx, float sy,
                                                             float sz, float tx, float ty, float tz) {
    // Check if the input matrix H is 3x3
    if (H.size() != 9) {
        throw std::invalid_argument("Input matrix H must be of size 3x3.");
    }

    // Convert the input vector to an Eigen matrix
    Eigen::Matrix3f matrix_H;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix_H(i, j) = H[i * 3 + j];
        }
    }

    // Perform SVD decomposition using Eigen
    // H = U * S * V^T
    // where U and V are orthogonal matrices and S is a diagonal matrix of singular values
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix_H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    // Eigen::Vector3f S = svd.singularValues();
    Eigen::Matrix3f V = svd.matrixV();

    // Compute Rotation, R = V * U^T
    Eigen::Matrix3f R = V * U.transpose();
    // Check if the determinant of R is negative
    if (R.determinant() < 0) {
        // If the determinant is negative, we need to correct it
        V.col(2) *= -1; // Flip the last column of V
        // Recompute the rotation matrix
        R = V * U.transpose();
    }

    // Compute the translation vector
    Eigen::Vector3f sourceCentroid(sx, sy, sz);
    Eigen::Vector3f targetCentroid(tx, ty, tz);
    Eigen::Vector3f t = targetCentroid - R * sourceCentroid;

    // Convert the results back to std::vector<float>
    std::vector<float> rotation(9);
    std::vector<float> translation(3);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j)
            rotation[i * 3 + j] = R(i, j);
        translation[i] = t(i);
    }
    return {rotation, translation};
}