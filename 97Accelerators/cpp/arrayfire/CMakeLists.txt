find_package(ArrayFire)
if(ArrayFire_FOUND)
  find_package(OpenCL)

  add_executable(max_norm max_norm.cc)
  target_include_directories(max_norm SYSTEM PUBLIC
    "${ArrayFire_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/01research/cpp")
  target_link_libraries(max_norm
    "${ArrayFire_Unified_LIBRARIES}"
    "${ArrayFire_CPU_LIBRARIES}"
    "${ArrayFire_OpenCL_LIBRARIES}")
endif()
