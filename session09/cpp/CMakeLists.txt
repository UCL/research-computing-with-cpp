find_package(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

if(CUDA_FOUND)
   add_subdirectory(saxpy)
   add_subdirectory(cuda)
   add_subdirectory(sgemm)
endif()
