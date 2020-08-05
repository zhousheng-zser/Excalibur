
include_directories(../../include/Excalibur 
			../../include)
if(USE_CUDA)
	file(GLOB Excalibur_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/cuda/*.cu)
else()
	file(GLOB Excalibur_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp)
endif()


if(USE_CUDA)
	CUDA_ADD_LIBRARY(Excalibur SHARED ${Excalibur_source_files})
#	set(CUDA_NVCC_FLAGS -arch=compute_30)
else()
	add_library(Excalibur SHARED ${Excalibur_source_files})
endif()

target_link_libraries(Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Excalibur Julius Primitives)

