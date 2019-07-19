
include_directories(../../include/Excalibur)
if(USE_CUDA)
	file(GLOB Excalibur_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cu)
else()
	file(GLOB Excalibur_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp)
endif()


if(USE_CUDA)
	CUDA_ADD_LIBRARY(Excalibur STATIC ${Excalibur_source_files})
#	set(CUDA_NVCC_FLAGS -arch=compute_30)
else()
	add_library(Excalibur STATIC ${Excalibur_source_files})
endif()

target_link_libraries(Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Excalibur Julius )

