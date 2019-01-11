cmake_minimum_required(VERSION 2.8)
project(joint_bilateral_filter)
add_executable(joint_bilateral_filter src/main.cpp)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-O0 -g -Dlinux -std=gnu++11 -lpthread")
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(joint_bilateral_filter ${OpenCV_LDFLAGS})
target_link_libraries(joint_bilateral_filter -lopencv_ximgproc)
