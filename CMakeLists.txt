cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(relocalization)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# pcl 
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})	

add_executable (point_cloud_relocalization point_cloud_relocalization.cpp)
target_link_libraries (point_cloud_relocalization ${OpenCV_LIBS} ${PCL_LIBRARIES})

 

