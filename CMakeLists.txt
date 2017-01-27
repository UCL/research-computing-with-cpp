cmake_minimum_required(VERSION 2.8.8)
project(ResearchComputingWithCPP)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Add the c++11 flag
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Apparently now this is also needed for ITK-related stuff to compile...
add_compile_options(-fpermissive)

add_subdirectory(01research/cpp)
add_subdirectory(02cpp/cpp)
add_subdirectory(02stl/cpp)
add_subdirectory(03libraries/cpp)
add_subdirectory(04HPC/cpp)
add_subdirectory(05OpenMP/cpp)
add_subdirectory(06MPI/cpp)
add_subdirectory(07MPIExample/cpp)
add_subdirectory(09Accelerators/cpp)
add_subdirectory(97TemplateMetaProg/cpp)
