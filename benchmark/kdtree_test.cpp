#include "kdtree.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
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

std::vector<float> pclKDTree(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr target, float inlierThreshold) {
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
    return minDistances;
}

std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr, std::vector<float3>>
generateRandomPointCloud(int n, float min, float max) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(n);
    std::vector<float3> cloud2(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int i = 0; i < n; ++i) {
        cloud->at(i).x = dis(gen);
        cloud->at(i).y = dis(gen);
        cloud->at(i).z = dis(gen);
        cloud2[i] = {cloud->at(i).x, cloud->at(i).y, cloud->at(i).z};
    }
    return {cloud, cloud2};
}

void run(int N, int Q) {
    float mn = -2, mx = 2;
    auto [source, source2] = generateRandomPointCloud(N, mn, mx);
    auto [target, target2] = generateRandomPointCloud(Q, mn, mx);
    float inlierThreshold = 0.5;
    Timer timer;

    timer.start();
    std::vector<float> distances = pclKDTree(source, target, inlierThreshold);
    auto end = std::chrono::high_resolution_clock::now();
    timer.end("PCL KdTree");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    spdlog::info("CUDA KdTree");
    timer.start();
    timer.start();
    auto kdtree = KDTree();
    kdtree.buildTree(source2, stream);
    timer.end("Build");
    timer.start();
    auto distances2 = kdtree.findAllNearestDistance(target2, inlierThreshold, stream);
    timer.end("Queries");
    timer.end("Total Latency");
    cudaStreamDestroy(stream);

    if (distances.size() != distances2.size()) {
        spdlog::error("Distances size mismatch.");
        return;
    }

    int queryIndex = 0;
    float queryMaxDiff = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        auto diff = std::abs(distances[i] - distances2[i]);
        if (diff > queryMaxDiff) {
            queryMaxDiff = diff;
            queryIndex = i;
        }
    }
    if (queryMaxDiff < 1e-6)
        spdlog::info("Max diff: {} at index {}", queryMaxDiff, queryIndex);
    else
        spdlog::error("Max diff: {} at index {}", queryMaxDiff, queryIndex);
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/benchmark/kdtree_test", "KdTree Benchmark");
    // clang-format off
    options.add_options()("h,help", "Show help")(
        "N", "Number of points in KdTree", cxxopts::value<int>()->default_value("1000000"))(
        "Q", "Number of queries/points", cxxopts::value<int>()->default_value("1000000"))(
        "r,repeat", "Repeat times", cxxopts::value<int>()->default_value("1"));
    // clang-format on
    auto config = options.parse(argc, argv);
    if (config.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    spdlog::cfg::load_env_levels();
    // spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%x %X.%e] [%^%l%$] %v");

    int N = config["N"].as<int>();
    int Q = config["Q"].as<int>();
    int repeat = config["repeat"].as<int>();
    spdlog::info("N: {}, Q: {}, iter: {}", N, Q, iter);

    for (int i = 0; i < iter; ++i) {
        spdlog::info("Repeat: {}", i + 1);
        run(N, Q);
    }
    spdlog::info("Finished all repeats.");
    return 0;
}
