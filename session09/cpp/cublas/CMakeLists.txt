find_package(CUDA)
find_package(BLAS)

if(BLAS_FOUND AND CUDA_FOUND)
include_directories(CUDA_INCLUDE_DIRS)
add_executable(cublas_sgemm cublas_sgemm.c)
target_link_libraries(cublas_sgemm CUDA_CUBLAS_LIBRARIES CUDA_LIBRARIES BLAS_LIBRARIES)
set_target_properties(cublas_sgemm PROPERTIES LINK_FLAGS BLAS_LINKER_FLAGS)
endif()