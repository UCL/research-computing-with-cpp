cmake_minimum_required(VERSION 3.1)
project(ResearchComputingWithCPP)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Add the c++11 flag
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Apparently now this is also needed for ITK-related stuff to compile...
add_compile_options(-fpermissive)

add_subdirectory(01research/cpp)
add_subdirectory(03cpp/cpp)
add_subdirectory(03cpppatterns/cpp)
add_subdirectory(04HPC/cpp)
add_subdirectory(05OpenMP/cpp)
add_subdirectory(06MPI/cpp)
add_subdirectory(09Accelerators/cpp)
add_subdirectory(11Performance/cpp)
add_subdirectory(88cpplibraries/cpp)
add_subdirectory(90stl/cpp)
add_subdirectory(92TemplateMetaProg/cpp)
add_subdirectory(93MPIExample/cpp)
