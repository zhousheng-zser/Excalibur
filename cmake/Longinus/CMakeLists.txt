#cmake_minimum_required(VERSION 3.0.0)
#project(Longinus)


include_directories(../../include/Longinus 
			../../include/Damocles)

file(GLOB Longinus_source_cpp_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Longinus/*.cpp)
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
	file(GLOB Longinus_source_cu_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Longinus/*.cu)
	list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -O3)
	cuda_compile(cuda_objs ${Longinus_source_cu_files})
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
	add_library(Longinus ${cuda_objs} ${Longinus_source_cpp_files})
else()
	add_library(Longinus ${Longinus_source_cpp_files})
endif()

target_link_libraries(Longinus Romancia Damocles ${Excalibur_LINKER_LIBS})
add_dependencies(Longinus Romancia Damocles)
