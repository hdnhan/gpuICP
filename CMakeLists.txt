cmake_minimum_required(VERSION 3.20)

project(cuICP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

add_library(
  cukdtree SHARED
  src/kdtree.cu
)
add_subdirectory(benchmark)
