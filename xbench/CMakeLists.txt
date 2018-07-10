cmake_minimum_required(VERSION 2.8.12)
project(tutorial)

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/home/wolfv/Programs/xtensor-python/cmake")

set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -march=native -g")

find_package(xtensor)
find_package(xtl)
find_package(xtensor-python)
find_package(xsimd)
add_definitions(-DXTENSOR_USE_XSIMD)

find_package(NumPy)

set(PYBIND11_PYTHON_VERSION 2)

add_subdirectory(pybind11)
include_directories(../include ${xtl_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} ${xtensor-python_INCLUDE_DIRS} ${xsimd_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

pybind11_add_module(xbench xtensor_bench.cpp)
