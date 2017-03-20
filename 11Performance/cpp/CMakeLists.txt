cmake_minimum_required(VERSION 3.1)
project(TravellingSalesman)

# Add the c++11 flag
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(awful awful.cc)

find_package(Eigen3)
if(Eigen3_FOUND)
  add_executable(less_bad less_bad.cc)
  target_include_directories(less_bad SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
endif()
