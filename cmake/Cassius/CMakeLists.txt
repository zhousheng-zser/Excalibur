#cmake_minimum_required(VERSION 3.0.0)
#project(Cassius)

#find_package(CUDA)
#add_definitions(-Dx86)

include_directories(../../include/Cassius 
			../../include/Excalibur)
if(USE_CUDA)
	file(GLOB Cassius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Cassius/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Cassius/*.cu)
else()
	file(GLOB Cassius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Cassius/*.cpp)
endif()

#set(CUDA_NVCC_FLAGS -arch=compute_30)
#set(libs ../Julius/build/libJulius.a 
#	../Excalibur/build/libExcalibur.a)

if(USE_CUDA)
	CUDA_ADD_LIBRARY(Cassius ${Cassius_source_files})
else()
	add_library(Cassius ${Cassius_source_files})
endif()

target_link_libraries(Cassius Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Cassius Excalibur)
