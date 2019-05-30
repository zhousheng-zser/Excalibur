#cmake_minimum_required(VERSION 3.0.0)
#project(Damocles)

#find_package(CUDA)
#add_definitions(-Dx86)

include_directories(../../include/Damocles 
			../../include/Excalibur)
if(USE_CUDA)
	file(GLOB Damocles_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Damocles/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Damocles/*.cu)
else()
	file(GLOB Damocles_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Damocles/*.cpp)
endif()

#set(CUDA_NVCC_FLAGS -arch=compute_30)
#set(libs ../Julius/build/libJulius.a 
#	../Excalibur/build/libExcalibur.a)

if(USE_CUDA)
	CUDA_ADD_LIBRARY(Damocles STATIC ${Damocles_source_files})
else()
	add_library(Damocles STATIC ${Damocles_source_files})
endif()

target_link_libraries(Damocles Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Damocles Excalibur ${Excalibur_LINKER_LIBS})
