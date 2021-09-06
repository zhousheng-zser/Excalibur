cmake_minimum_required(VERSION 3.14.3)
set(module_name speed_benchmark)
include_directories(../../include)

add_executable(${module_name} ../../src/SpeedBenchmark/speedbenchmark.cpp)

# message(STATUS "***CUDNN_LIBRARY_DIRS = ${CUDNN_LIBRARY_DIRS}")

target_link_directories(${module_name} PRIVATE ${CUDNN_LIBRARY_DIRS})
target_link_libraries(${module_name} primitives excalibur)
add_dependencies(${module_name} primitives excalibur)
