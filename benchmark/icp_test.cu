#include "icp.hpp"
#include "kdtree.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <random>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <tuple>

class Timer {
  public:
    void start() { startTimes.push(std::chrono::high_resolution_clock::now()); }
    void end(std::string const &msg) {
        if (startTimes.empty()) {
            spdlog::error("No timer started.");
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTimes.top()).count();
        startTimes.pop();
        spdlog::info("{}: {} ms", msg, elapsed);
    }

  private:
    std::stack<std::chrono::high_resolution_clock::time_point> startTimes;
};

float pclKDTree(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                float inlierThreshold) {
    pcl::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    kdtree->setInputCloud(source);

    std::vector<float> minDistances(target->size(), -1);
#pragma omp parallel for
    for (int i = 0; i < target->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;
        if (kdtree->radiusSearch(target->at(i), inlierThreshold, indices, distances, 1) > 0)
            minDistances[i] = std::sqrt(distances[0]);
    }
    float sumDistances = 0, count = 0;
    for (int i = 0; i < minDistances.size(); ++i) {
        if (minDistances[i] > 0)
            count++;
    }
    float percent = 0;
    if (count > 0)
        percent = (float)count / source->size();
    return percent;
}

std::tuple<bool, float> pclICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr target, Eigen::Matrix4f &Rt,
                               float maxCorrespondenceDistance, int maximumIterations,
                               float transformationEpsilon, float euclideanFitnessEpsilon) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(maxCorrespondenceDistance);
    icp.setMaximumIterations(maximumIterations);
    icp.setTransformationEpsilon(transformationEpsilon);
    icp.setEuclideanFitnessEpsilon(euclideanFitnessEpsilon);

    icp.align(*transformed, Rt);
    Rt = icp.getFinalTransformation();
    bool converged = icp.hasConverged();
    auto percent = pclKDTree(transformed, target, maxCorrespondenceDistance);
    return {converged, percent};
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/benchmark/icp_test", "ICP Benchmark");
    // clang-format off
    options.add_options()("h,help", "Show help")(
        "s,source", "Source point cloud path", cxxopts::value<std::string>()->default_value("assets/source.ply"))(
        "t,target", "Target point cloud path", cxxopts::value<std::string>()->default_value("assets/target.ply"))(
        "repeat", "Repeat times", cxxopts::value<int>()->default_value("1"))(
        "maxiter", "Max iterations", cxxopts::value<int>()->default_value("1000"))(
        "epsilon", "Inlier threshold", cxxopts::value<float>()->default_value("0.05"));
    // clang-format on
    auto config = options.parse(argc, argv);
    if (config.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    Timer timer;

    spdlog::cfg::load_env_levels();
    // spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%x %X.%e] [%^%l%$] %v");

    std::string sourcePath = config["source"].as<std::string>();
    std::string targetPath = config["target"].as<std::string>();
    int repeat = config["repeat"].as<int>();
    spdlog::info("Source: {}, Target: {}, repeat: {}", sourcePath, targetPath, repeat);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile(sourcePath, *source) == -1) {
        spdlog::error("Failed to load source point cloud.");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile(targetPath, *target) == -1) {
        spdlog::error("Failed to load target point cloud.");
        return -1;
    }
    spdlog::info("Source size: {}, Target size: {}", source->size(), target->size());
    std::vector<float3> sourceVec(source->size());
    std::vector<float3> targetVec(target->size());
    for (int i = 0; i < source->size(); ++i)
        sourceVec[i] = {source->at(i).x, source->at(i).y, source->at(i).z};
    for (int i = 0; i < target->size(); ++i)
        targetVec[i] = {target->at(i).x, target->at(i).y, target->at(i).z};

    float maxCorrespondenceDistance = config["epsilon"].as<float>();
    int maximumIterations = config["maxiter"].as<int>();
    float transformationEpsilon = 1e-8f;
    float euclideanFitnessEpsilon = 1e-8f;

    for (int it = 0; it < repeat; ++it) {
        spdlog::info("Repeat: {}", it + 1);

        // Start with PCL ICP
        Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
        timer.start();
        auto [converged, percent] = pclICP(source, target, Rt, maxCorrespondenceDistance, maximumIterations,
                                           transformationEpsilon, euclideanFitnessEpsilon);
        timer.end("PCL ICP");
        if (converged)
            spdlog::info("PCL ICP converged, percent: {}", percent * 100);
        else
            spdlog::error("PCL ICP not converged.");
        spdlog::info("PCL ICP result:");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j)
                printf("%f ", Rt(i, j));
            printf("\n");
        }
        printf("\n");

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::vector<float> cuRt(16, 0); // Acts as initial guess
        for (int i = 0; i < 4; ++i)
            cuRt[i * 4 + i] = 1.0f;

        timer.start();
        ICP icp;
        icp.setTarget(targetVec, stream);
        auto [converged2, percent2] = icp.align(sourceVec, maxCorrespondenceDistance, maximumIterations,
                                                transformationEpsilon, euclideanFitnessEpsilon, cuRt, stream);
        timer.end("CUDA ICP");
        if (converged2)
            spdlog::info("CUDA ICP converged, percent: {}", percent2 * 100);
        else
            spdlog::error("CUDA ICP not converged.");
        spdlog::info("CUDA ICP result:");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j)
                printf("%f ", cuRt[i * 4 + j]);
            printf("\n");
        }
        cudaStreamDestroy(stream);
    }
    spdlog::info("Finished all repeats.");
    return 0;
}
