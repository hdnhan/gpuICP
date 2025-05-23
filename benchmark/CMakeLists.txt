# CMAKE MODULES
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CPM)

# EXTERNAL LIBRARIES
cpmaddpackage("gh:gabime/spdlog#v1.15.2")
cpmaddpackage("gh:jarro2783/cxxopts#v3.2.0")

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

set(target_list 
    kdtree_test
    icp_test
)
foreach(target ${target_list})
    if(CUDA_FOUND)
        cuda_add_executable(${target} ${target}.cu)
        target_compile_definitions(${target} PRIVATE CUDA_FOUND)
    endif()

    if(HIP_FOUND)
        add_executable(${target} ${target}.hip)
        target_compile_definitions(${target} PRIVATE HIP_FOUND)
    endif()

    target_include_directories(
        ${target} PUBLIC 
        ${PCL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(
        ${target}  
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
        gpuicp
        spdlog::spdlog
        cxxopts::cxxopts
    )
endforeach()
