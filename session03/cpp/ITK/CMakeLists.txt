find_package(ITK COMPONENTS ITKCommon ITKIOImageBase REQUIRED)
include(${ITK_USE_FILE})

find_package(GDCM)
if (GDCM_FOUND)
  include(${GDCM_USE_FILE})
endif()

find_package(OpenCV)

add_executable(ITKAdd.x ITKAdd.cc)
target_link_libraries(ITKAdd.x ${ITK_LIBRARIES} ${OpenCV_LIBS})

add_executable(ITKThreshold.x ITKThreshold.cc)
target_link_libraries(ITKThreshold.x ${ITK_LIBRARIES} ${OpenCV_LIBS})

add_executable(ITKTest.x ITKTest.cc)
