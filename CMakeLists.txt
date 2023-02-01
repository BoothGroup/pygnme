cmake_minimum_required(VERSION 3.16)
project(pygnme_build)

set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/carma/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/carma/extern/armadillo-code/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/wick)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/slater)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme/libgnme/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/wick)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/slater)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pygnme/utils)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/carma/extern/armadillo-code)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/carma)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/libgnme)

pybind11_add_module(wick ${CMAKE_CURRENT_SOURCE_DIR}/pygnme/wick_api.cc)
target_link_libraries(wick PUBLIC gnme_wick)
target_link_libraries(wick PUBLIC gnme_utils)
target_link_libraries(wick PUBLIC carma::carma)

file(TOUCH "${CMAKE_CURRENT_SOURCE_DIR}/build/__init__.py")
