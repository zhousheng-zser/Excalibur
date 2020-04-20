
# CMakeList.txt: NativeUtils 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

set(NATIVEUTILS_TARGET_NAME NativeUtils)

file(GLOB NATIVEUTILS_SOURCE_FILES *.cpp)
add_library(${NATIVEUTILS_TARGET_NAME} SHARED ${NATIVEUTILS_SOURCE_FILES})

if(USE_ANDROID_NDK)
	set_target_stripping_command(${NATIVEUTILS_TARGET_NAME})
endif()
