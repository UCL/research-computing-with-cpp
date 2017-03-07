find_package(CUDA)

add_subdirectory(saxpy)
add_subdirectory(arrayfire)

if(CUDA_FOUND)
   add_subdirectory(cuda)
   add_subdirectory(sgemm)
endif()
