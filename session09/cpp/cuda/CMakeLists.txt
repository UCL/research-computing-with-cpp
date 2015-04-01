find_package(CUDA)
cuda_add_executable(cusaxpy.x saxpy.cu)
cuda_add_executable(cusgemm.x sgemm.cu)
