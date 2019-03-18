cmake_minimum_required(VERSION 3.1)
project(TravellingSalesman)

# Add the c++11 flag
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(awful awful.cc)

include_directories(${CMAKE_SOURCE_DIR}/88cpplibraries/cpp/Eigen/eigen-3.3.7)

add_executable(less_bad less_bad.cc)
add_executable(micro_benchmark micro_benchmark.cc)

