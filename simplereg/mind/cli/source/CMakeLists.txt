cmake_minimum_required(VERSION 3.9.5)
 
project(VectorImageRegistration)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()

add_executable(runVectorImageRegistration runVectorImageRegistration.cpp)

target_link_libraries(runVectorImageRegistration ${ITK_LIBRARIES})
