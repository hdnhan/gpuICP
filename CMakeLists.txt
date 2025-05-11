cmake_minimum_required(VERSION 3.20)

project(cuICP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print time taken to compile and link, should use -j1 to print correct time corresponding to the command
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(
  cuicp SHARED
  src/icp.cu
  src/kdtree.cu
  src/svd.cpp
)
target_include_directories(
  cuicp PUBLIC 
  ${EIGEN3_INCLUDE_DIR}
)
add_subdirectory(benchmark)
