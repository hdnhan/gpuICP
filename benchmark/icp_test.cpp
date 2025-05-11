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

void pclKDTree(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target,
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
        if (minDistances[i] > 0) {
            sumDistances += minDistances[i];
            count++;
        }
    }
    if (count > 0) {
        float meanDistance = sumDistances / count;
        spdlog::info("Mean distance: {}, Count: {}", meanDistance, count);
    } else {
        spdlog::info("No inliers found.");
    }
}

std::vector<float> pclICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr target, float maxCorrespondenceDistance,
                          int maximumIterations, float transformationEpsilon, float euclideanFitnessEpsilon) {
    std::vector<float> Rt(16);
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(maxCorrespondenceDistance);
    icp.setMaximumIterations(maximumIterations);
    icp.setTransformationEpsilon(transformationEpsilon);
    icp.setEuclideanFitnessEpsilon(euclideanFitnessEpsilon);

    icp.align(*transformed);

    Eigen::Matrix4f trans = icp.getFinalTransformation();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Rt[i * 4 + j] = trans(i, j);

    pclKDTree(transformed, target, maxCorrespondenceDistance);
    return Rt;
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/benchmark/icp_test", "ICP Benchmark");
    options.add_options()("h,help", "Show help")(
        "s", "Source point cloud path", cxxopts::value<std::string>()->default_value("/tmp/source.ply"))(
        "t", "Target point cloud path", cxxopts::value<std::string>()->default_value("/tmp/target.ply"))(
        "iter", "Number of iterations", cxxopts::value<int>()->default_value("1"));

    auto config = options.parse(argc, argv);
    if (config.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    Timer timer;

    spdlog::cfg::load_env_levels();
    // spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%x %X.%e] [%^%l%$] %v");

    std::string sourcePath = config["s"].as<std::string>();
    std::string targetPath = config["t"].as<std::string>();
    int iter = config["iter"].as<int>();
    spdlog::info("Source: {}, Target: {}, iter: {}", sourcePath, targetPath, iter);

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

    float maxCorrespondenceDistance = 0.05f;
    int maximumIterations = 1000;
    float transformationEpsilon = 1e-8f;
    float euclideanFitnessEpsilon = 1e-8f;
    // print first 10 points of source and target
    spdlog::info("Source points:");
    for (int i = 0; i < std::min(10, (int)source->size()); ++i)
        spdlog::info("({:.3f}, {:.3f}, {:.3f})", source->points[i].x, source->points[i].y,
                     source->points[i].z);
    spdlog::info("Target points:");
    for (int i = 0; i < std::min(10, (int)target->size()); ++i)
        spdlog::info("({:.3f}, {:.3f}, {:.3f})", target->points[i].x, target->points[i].y,
                     target->points[i].z);

    timer.start();
    std::vector<float> Rt = pclICP(source, target, maxCorrespondenceDistance, maximumIterations,
                                   transformationEpsilon, euclideanFitnessEpsilon);
    timer.end("PCL ICP");
    spdlog::info("PCL ICP result:");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j)
            printf("%f ", Rt[i * 4 + j]);
        printf("\n");
    }

    return 0;
}
