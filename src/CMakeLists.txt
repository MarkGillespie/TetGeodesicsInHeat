cmake_minimum_required(VERSION 2.8.9)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(
    tetMesh
    tet.h
    tet.cpp
    vector3.h
    vector3.ipp
    cluster.h
    cluster.cpp
)
target_link_libraries(tetMesh Eigen3::Eigen)

add_library(
    cudaSolver
    cuda/cg.cuh
    cuda/cg.cu
)
target_link_libraries(cudaSolver Eigen3::Eigen tetMesh -lcublas)
target_compile_options(cudaSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)

add_library(cudaCG STATIC
    cuda/cg.cuh
    cuda/cg.cu
)
target_include_directories(cudaCG PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(cudaCG PROPERTIES
  CUDA_STANDARD 11)
    # CUDA_SEPARABLE_COMPILATION ON
target_link_libraries(cudaCG Eigen3::Eigen tetMesh)

add_executable(
  run
  main.cpp
)

target_include_directories(run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include" "/usr/local/cuda/include")
target_link_libraries(run cudaCG tetMesh args ${CUDA_LIBRARIES} -lcublas)

option(PACKAGE_TESTS "Build the tests" OFF)

if(PACKAGE_TESTS)
  enable_testing()
  add_subdirectory(
    tests
  )
endif()
