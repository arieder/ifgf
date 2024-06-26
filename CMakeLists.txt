
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()


cmake_minimum_required(VERSION 3.4...3.18)
project(ifgf CXX)
#


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Ofast -fno-math-errno -ffast-math -funroll-loops")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -Ofast ")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2 ")
add_definitions(-DEIGEN_FAST_MATH=1 -DEIGEN_DONT_PARALLELIZE  )
#add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)



find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (TBB REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${TBB_INCLUDE_DIR})

# find_package(pybind11 REQUIRED)
# pybind11_add_module(pyifgf SHARED py_ifgf.cpp )
# target_link_libraries (pyifgf PRIVATE tbb)
# set_property(TARGET pyifgf PROPERTY CXX_STANDARD 20)



set(TARGET_NAME "test_cheb")
set(SOURCE_FILE test_chebinterp.cpp)
set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} -O2")
set(LINK_FLAGS "-fsycl")

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})
 set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example
#                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
