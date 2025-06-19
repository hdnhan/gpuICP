cmake_minimum_required(VERSION 3.20)

project(gpuICP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print time taken to compile and link, should use -j1 to print correct time corresponding to the command
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

find_package(CUDA)
find_package(HIP)
find_package(Eigen3 REQUIRED)

if (NOT CUDA_FOUND AND NOT HIP_FOUND)
  message(FATAL_ERROR "CUDA or HIP not found. Please install CUDA or HIP.")
elseif (CUDA_FOUND AND HIP_FOUND)
  message(STATUS "Both CUDA and HIP found. Using CUDA by default.")
endif()

set(target gpuicp)
if(CUDA_FOUND)
  enable_language(CUDA)
  file(GLOB_RECURSE SOURCES src/*.cu src/*.cpp)
  add_library(${target} SHARED ${SOURCES})
  target_compile_definitions(${target} PRIVATE CUDA_FOUND)
elseif(HIP_FOUND)
  enable_language(HIP)
  file(GLOB_RECURSE SOURCES src/*.hip src/*.cpp)
  add_library(${target} SHARED ${SOURCES})
  target_compile_definitions(${target} PRIVATE HIP_FOUND)
else()
  message(FATAL_ERROR "Neither CUDA nor HIP found. Please install one of them.")
endif()

target_include_directories(
  ${target} PUBLIC 
  ${EIGEN3_INCLUDE_DIR}
)
add_subdirectory(benchmark)
