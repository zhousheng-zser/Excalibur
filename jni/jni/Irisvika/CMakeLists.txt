
# CMakeList.txt: Irisvika 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

file(GLOB IRISVIKA_SOURCE_FILES *.cpp)
add_library(Irisvika SHARED ${IRISVIKA_SOURCE_FILES})
add_dependencies(Irisvika JavaInterop)
