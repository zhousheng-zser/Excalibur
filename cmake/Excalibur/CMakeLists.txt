cmake_minimum_required(VERSION 3.14.3)
set(module_name excalibur)
include_directories(../../include/Excalibur ../../include)
if(USE_CUDA)
	file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/cuda/*.cu)
else()
	file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp)
endif()

list(REMOVE_ITEM src "${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/main.cpp")

if(USE_CUDA)
	CUDA_ADD_LIBRARY(${module_name} SHARED ${src})
#	set(CUDA_NVCC_FLAGS -arch=compute_30)
else()
	add_library(${module_name} SHARED ${src})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_definitions(${module_name} PRIVATE -DEXPORT_EXCALIBUR_PRIMITIVES -D_WINDLL)
endif()

if(USE_CUDA)
	target_include_directories(${module_name} SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
	target_compile_definitions(${module_name} PRIVATE -DUSE_CUDA)
	target_link_libraries(${module_name} ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
endif()

if(USE_CUDNN)
	target_include_directories(${module_name} PRIVATE ${CUDNN_INCLUDE_DIRS})
	target_link_directories(${module_name} PRIVATE ${CUDNN_LIBRARY_DIRS})
	target_compile_definitions(${module_name} PRIVATE -DUSE_CUDNN)
	target_link_libraries(${module_name} ${CUDNN_LIBRARIES})
endif()

if(USE_OPENMP)
	target_link_libraries(${module_name} OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${module_name} julius primitives)
add_dependencies(${module_name} julius primitives)

