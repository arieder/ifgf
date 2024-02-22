cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(pybind11 REQUIRED)
pybind11_add_module(pyifgf SHARED py_ifgf.cpp )
target_link_libraries (pyifgf PRIVATE tbb)


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example
#                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
