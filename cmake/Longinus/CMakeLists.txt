#cmake_minimum_required(VERSION 3.0.0)
#project(Longinus)


include_directories(../../include/Longinus 
			../../include/Damocles)
if(USE_CUDA)
	file(GLOB Longinus_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Longinus/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Longinus/*.cu)
else()
	file(GLOB Longinus_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Longinus/*.cpp)
endif()

#set(CUDA_NVCC_FLAGS -arch=compute_30)
#set(libs ../Julius/build/libJulius.a 
#	../Excalibur/build/libExcalibur.a 
#	../Damocles/build/libDamocles.a)

if(USE_CUDA)
	CUDA_ADD_LIBRARY(Longinus ${Longinus_source_files})
else()
	add_library(Longinus ${Longinus_source_files})
endif()

target_link_libraries(Longinus Romancia Damocles ${Excalibur_LINKER_LIBS})
add_dependencies(Longinus Romancia Damocles)
