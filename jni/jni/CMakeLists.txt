# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Excalibur")

add_compile_options(
	-I${ANDROID_INCLUDE_SYSROOT}
	-I${ANDROID_INCLUDE_CXX}
	-I${ANDROID_INCLUDE_CLANG}
	-I${ANDROID_INCLUDE_PLATFORM}
	-IC:/Tools/vcpkg/installed/x64-windows/include)

option(USE_ANDROID_NDK "Sets whether to use the Android NDK." on)

# 包含子项目。
add_subdirectory("JavaInterop")
add_subdirectory("Tensor")
add_subdirectory("Irisvika")
