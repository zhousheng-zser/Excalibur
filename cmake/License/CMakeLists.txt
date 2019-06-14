#cmake_minimum_required(VERSION 3.0.0)
#project(Julius)

set(CMAKE_CXX_STANDARD 14)
include_directories(../../include/License)
file(GLOB License_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/License/*.cpp)

add_library(License ${License_source_files})
target_link_libraries(License ssl stdc++fs)
