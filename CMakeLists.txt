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

add_library(
  cukdtree SHARED
  src/kdtree.cu
)
add_subdirectory(benchmark)
