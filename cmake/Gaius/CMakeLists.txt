
include_directories(../../include/Gaius 
			../../include/Excalibur)
if(USE_CUDA)
	file(GLOB Gaius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cu)
else()
	file(GLOB Gaius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cpp)
endif()

if(USE_CUDA)
	CUDA_ADD_LIBRARY(Gaius ${Gaius_source_files})
else()
	add_library(Gaius ${Gaius_source_files})
endif()

target_link_libraries(Gaius Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Gaius Excalibur)
