find_package(MPI)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")

include_directories(src)
include_directories(lib)

add_subdirectory(src)
add_subdirectory(test)

