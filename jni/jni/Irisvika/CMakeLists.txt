
# CMakeList.txt: Irisvika 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

if(NOT USE_ANDROID_NDK)
	set(ADDITIONAL_LIBRARIES bcrypt.lib)
endif()

set(IRISVIKA_TARGET_NAME Irisvika)
file(GLOB IRISVIKA_SOURCE_FILES *.cpp ${GLASSSIX_EXCALIBUR_SOURCE}/Irisviel/*.cpp)

add_library(${IRISVIKA_TARGET_NAME} SHARED ${IRISVIKA_SOURCE_FILES})
add_dependencies(${IRISVIKA_TARGET_NAME} JavaInterop)

target_link_libraries(${IRISVIKA_TARGET_NAME} PRIVATE JavaInterop ${ADDITIONAL_LIBRARIES})
target_compile_definitions(${IRISVIKA_TARGET_NAME} PRIVATE -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS -DCOSINE_DISTANCE -DFMT_HEADER_ONLY)

target_include_directories(
	${IRISVIKA_TARGET_NAME} PRIVATE
	${GLASSSIX_EXCALIBUR_INCLUDE}
	${GLASSSIX_EXCALIBUR_INCLUDE}/Irisviel
	../JavaInterop)

if(USE_ANDROID_NDK)
	set_target_stripping_command(${IRISVIKA_TARGET_NAME})
endif()
