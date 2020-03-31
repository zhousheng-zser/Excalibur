# CMakeList.txt: Tensor 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

include_directories(../JavaInterop)
add_compile_options(-I${EXCALIBUR_INCLUDE_CXX}/Infrastructure)

file(GLOB TENSOR_SOURCE_FILES *.cpp)
add_library(Tensor SHARED ${TENSOR_SOURCE_FILES})
add_dependencies(Tensor JavaInterop)
