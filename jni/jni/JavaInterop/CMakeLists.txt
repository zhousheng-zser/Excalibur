
# CMakeList.txt: Primitives 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

file(GLOB JAVAINTEROP_SOURCE_FILES *.cpp)
add_library(JavaInterop STATIC ${JAVAINTEROP_SOURCE_FILES})
