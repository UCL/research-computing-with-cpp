### "hello"
cmake_minimum_required(VERSION 2.8)

find_package(MPI REQUIRED)

include_directories(${MPI_C_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")

add_executable(hello hello.cc)
target_link_libraries(hello ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

### "dummy"
add_executable(point2point point2point.cc)
target_link_libraries(point2point ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(collective collective.cc)
target_link_libraries(collective ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(all2all all2all.cc)
target_link_libraries(all2all ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(split split.cc)
target_link_libraries(split ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
