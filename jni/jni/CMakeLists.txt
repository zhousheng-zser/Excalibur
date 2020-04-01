# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project("Excalibur")

option(USE_ANDROID_NDK "Sets whether to use the Android NDK." off)
set(CMAKE_CXX_STANDARD 17)

get_filename_component(GLASSSIX_COMMAND_HEADER_INCLUDE C:/Tools/vcpkg/installed/x64-windows/include ABSOLUTE)
get_filename_component(GLASSSIX_EXCALIBUR_INCLUDE ../../include ABSOLUTE)
get_filename_component(GLASSSIX_EXCALIBUR_SOURCE ../../src ABSOLUTE)

add_compile_options(-I${GLASSSIX_COMMAND_HEADER_INCLUDE})

# Provides support for NDK or x86-64 JDK.
if(USE_ANDROID_NDK)
	get_filename_component(ANDROID_NDK_CLANG_INCLUDE ${ANDROID_TOOLCHAIN_ROOT}/lib64/clang/8.0.7/include ABSOLUTE)
	get_filename_component(ANDROID_NDK_SYSROOT_USER_INCLUDE ${CMAKE_SYSROOT}/usr/include ABSOLUTE)
	get_filename_component(ANDROID_NDK_SYSROOT_CXX_INCLUDE ${ANDROID_NDK_SYSROOT_USER_INCLUDE}/c++/v1 ABSOLUTE)
	get_filename_component(ANDROID_NDK_SYSROOT_PLATFORM_INCLUDE ${ANDROID_NDK_SYSROOT_USER_INCLUDE}/${ANDROID_TOOLCHAIN_NAME} ABSOLUTE)

	add_compile_options(
		-I${ANDROID_NDK_CLANG_INCLUDE}
		-I${ANDROID_NDK_SYSROOT_USER_INCLUDE}
		-I${ANDROID_NDK_SYSROOT_CXX_INCLUDE}
		-I${ANDROID_NDK_SYSROOT_PLATFORM_INCLUDE})

	# Creates a stripping function.
	function(set_target_stripping_command TARGET_NAME)
		add_custom_command(
			TARGET ${TARGET_NAME} POST_BUILD
			COMMAND mkdir stripped
			COMMAND ${ANDROID_TOOLCHAIN_PREFIX}strip --strip-unneeded $<TARGET_FILE:${TARGET_NAME}> -o stripped/$<TARGET_FILE_NAME:${TARGET_NAME}>)
	endfunction()

	add_subdirectory("JavaInterop")
	#add_subdirectory("Tensor")
	add_subdirectory("Irisvika")
else()
	set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
	set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
	set(CMAKE_IMPORT_LIBRARY_PREFIX "lib")

	add_compile_options(
		-IE:/Tools/jdk1.8.0_201/include
		-IE:/Tools/jdk1.8.0_201/include/win32)

	add_subdirectory("JavaInterop")
	add_subdirectory("Irisvika")
endif()
