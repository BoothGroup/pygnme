cmake_minimum_required(VERSION 3.16)
project(pygnme_build)

# C++ arguments
set(CMAKE_CXX_FLAGS "-Wall -O3 ${CMAKE_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/carma/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/armadillo/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/wick)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/slater)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/wick)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/slater)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/utils)

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(WITH_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
else()
    set(WITH_OPENMP 0)
endif()

# BLAS
find_package(BLAS)
find_package(CBLAS)
find_package(LAPACK)

# Armadillo
set(ARMADILLO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/armadillo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/armadillo)

# pybind11
find_package(pybind11 REQUIRED)

# CARMA
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/carma)

# libGNME
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme)

# Compile the library
pybind11_add_module(libpygnme
    ${CMAKE_CURRENT_SOURCE_DIR}/pygnme/libpygnme.cc
)
target_link_libraries(libpygnme PUBLIC gnme_wick)
target_link_libraries(libpygnme PUBLIC gnme_slater)
target_link_libraries(libpygnme PUBLIC gnme_utils)
target_link_libraries(libpygnme PUBLIC blas)
target_link_libraries(libpygnme PUBLIC lapack)
target_link_libraries(libpygnme PUBLIC carma::carma)
