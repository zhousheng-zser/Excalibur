#cmake_minimum_required(VERSION 3.0.0)
#project(Irisvian)

#find_package(OpenMP)

include_directories(../../include/Irisvian)
file(GLOB Irisvian_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Irisvian/*.cpp)

#add_definitions(-Dx86)
add_library(Irisvian ${Irisvian_source_files})

