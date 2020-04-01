
# CMakeList.txt: Irisvika 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

if (USE_ANDROID_NDK)
	set(IRISVIKA_OUTPUT_NAME Irisvika)
else()
	set(IRISVIKA_OUTPUT_NAME libIrisvika)
endif()

file(GLOB IRISVIKA_SOURCE_FILES *.cpp ${GLASSSIX_EXCALIBUR_SOURCE}/Irisviel/*.cpp)

add_library(${IRISVIKA_OUTPUT_NAME} SHARED ${IRISVIKA_SOURCE_FILES})
add_dependencies(${IRISVIKA_OUTPUT_NAME} JavaInterop)

target_link_libraries(${IRISVIKA_OUTPUT_NAME} PRIVATE JavaInterop bcrypt.lib)
target_compile_definitions(${IRISVIKA_OUTPUT_NAME} PRIVATE -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DCOSINE_DISTANCE -D_CRT_SECURE_NO_WARNINGS -DFMT_HEADER_ONLY)

target_include_directories(${IRISVIKA_OUTPUT_NAME} PRIVATE
	${GLASSSIX_EXCALIBUR_INCLUDE}
	${GLASSSIX_EXCALIBUR_INCLUDE}/Irisviel
	../JavaInterop)
