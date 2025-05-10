# CMAKE MODULES
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CPM)

# EXTERNAL LIBRARIES
cpmaddpackage("gh:gabime/spdlog#v1.15.2")
cpmaddpackage("gh:jarro2783/cxxopts#v3.2.0")

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

cuda_add_executable(kdtree_test kdtree.cpp)
target_include_directories(
    kdtree_test PUBLIC 
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(
    kdtree_test  
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
    cukdtree
    spdlog::spdlog
    cxxopts::cxxopts
)
