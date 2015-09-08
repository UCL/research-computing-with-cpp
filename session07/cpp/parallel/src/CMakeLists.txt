add_executable(smooth_mpi.x parallel_main.cpp )
add_library(smooth_library Smooth.cpp SmoothWriter.cpp TextWriter.cpp BinaryWriter.cpp XdrWriter.cpp SingleWriter.cpp ParallelWriter.cpp)
target_link_libraries(smooth_mpi smooth_library  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${XDR_LIBRARIES})
