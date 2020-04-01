# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("Excalibur")

option(USE_ANDROID_NDK "Sets whether to use the Android NDK." off)

set(CMAKE_CXX_STANDARD 17)
get_filename_component(GLASSSIX_COMMAND_HEADER_INCLUDE C:/Tools/vcpkg/installed/x64-windows/include ABSOLUTE)
get_filename_component(GLASSSIX_EXCALIBUR_INCLUDE ../../include ABSOLUTE)
get_filename_component(GLASSSIX_EXCALIBUR_SOURCE ../../src ABSOLUTE)

add_compile_options(-I${GLASSSIX_COMMAND_HEADER_INCLUDE})

# 包含子项目。
if (USE_ANDROID_NDK)
	add_compile_options(
		-I${ANDROID_INCLUDE_SYSROOT}
		-I${ANDROID_INCLUDE_CXX}
		-I${ANDROID_INCLUDE_CLANG}
		-I${ANDROID_INCLUDE_PLATFORM})

	add_subdirectory("JavaInterop")
	add_subdirectory("Tensor")
	add_subdirectory("Irisvika")
else()
	add_compile_options(
		-IE:/Tools/jdk1.8.0_201/include
		-IE:/Tools/jdk1.8.0_201/include/win32)

	add_subdirectory("JavaInterop")
	add_subdirectory("Irisvika")
endif()
