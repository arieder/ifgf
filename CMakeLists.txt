cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Ofast -fno-math-errno -ffast-math -funroll-loops  -UUSE_NGSOLVE -flto=auto")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -Ofast  -flto=auto")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2  -flto=auto")
add_definitions(-DEIGEN_FAST_MATH=1 -DEIGEN_DONT_PARALLELIZE   )

find_package(unordered_dense CONFIG )



find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (TBB REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${TBB_INCLUDE_DIR})

find_package(pybind11 REQUIRED)
pybind11_add_module(pyifgf SHARED py_ifgf.cpp )
target_link_libraries (pyifgf PRIVATE tbb)
set_property(TARGET pyifgf PROPERTY CXX_STANDARD 20)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example
#                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
