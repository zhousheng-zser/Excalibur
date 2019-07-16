#cmake_minimum_required(VERSION 3.0.0)
#project(Excalibur)

#find_package(CUDA)


include_directories(../../include/Excalibur)
file(GLOB Excalibur_source_cpp_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp)

foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
	set(${var}_backup_in_cuda_compile_ "${${var}}")
	unset(${var})
endforeach()
foreach(var HAVE_SSE HAVE_AVX HAVE_AVX2)
	if(${var})
		remove_definitions(-D${var})
	endif()
endforeach()

if(USE_CUDA)
	file(GLOB Excalibur_source_cu_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cu)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -O3)
	cuda_compile(cuda_objs ${Excalibur_source_cu_files})
endif()
foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
	set(${var} "${${var}_backup_in_cuda_compile_}")
	unset(${var}_backup_in_cuda_compile_)
endforeach()

foreach(var HAVE_SSE HAVE_AVX HAVE_AVX2)
	if(${var})
		add_definitions(-D${var})
	endif()
endforeach()
if(USE_CUDA)
	add_library(Excalibur STATIC ${cuda_objs} ${Excalibur_source_cpp_files})
else()
	add_library(Excalibur STATIC ${Excalibur_source_cpp_files})
endif()

target_link_libraries(Excalibur Julius ${Excalibur_LINKER_LIBS})
add_dependencies(Excalibur Julius )

