cmake_minimum_required(VERSION 3.14.3)
set(module_name julius)
include_directories(../../include/Julius ../../include)
file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Julius/*.cpp)

list(REMOVE_ITEM src "${CMAKE_CURRENT_SOURCE_DIR}/../../src/Julius/julius_test.cpp")

add_library(${module_name} STATIC ${src})

if(USE_OPENMP)
	target_link_libraries(${module_name} OpenMP::OpenMP_CXX)
endif()