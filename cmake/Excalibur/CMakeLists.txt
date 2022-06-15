cmake_minimum_required(VERSION 3.14.3)
set(module_name excalibur)
option(SUPPORT_QUANTIZATION "Support quantization" off)
include_directories(../../include/Excalibur ../../include)
if(USE_CUDA)
	file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/cuda/*.cu)
else()
	file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/arm/*.cpp)
endif()

file(GLOB header ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Excalibur/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Excalibur/arm/*.hpp)

list(REMOVE_ITEM src "${CMAKE_CURRENT_SOURCE_DIR}/../../src/Excalibur/main.cpp")

if(USE_CUDA)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60;
											-gencode arch=compute_61,code=sm_61;
											-gencode arch=compute_70,code=sm_70;
											-gencode arch=compute_72,code=sm_72;
											-gencode arch=compute_75,code=sm_75;
											-gencode arch=compute_80,code=sm_80;
											-gencode arch=compute_86,code=sm_86")
	CUDA_ADD_LIBRARY(${module_name} SHARED ${src} ${header})
else()
	add_library(${module_name} SHARED ${src} ${header})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_definitions(${module_name} PRIVATE -DEXPORT_EXCALIBUR_PRIMITIVES -D_WINDLL)
endif()

if(USE_CUDA)
	target_include_directories(${module_name} SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
	target_compile_definitions(${module_name} PRIVATE -DUSE_CUDA)
	target_link_libraries(${module_name} ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
	if(USE_CUDNN)
		target_include_directories(${module_name} PRIVATE ${CUDNN_INCLUDE_DIRS})
		target_link_directories(${module_name} PRIVATE ${CUDNN_LIBRARY_DIRS})
		target_compile_definitions(${module_name} PRIVATE -DUSE_CUDNN)
		target_link_libraries(${module_name} ${CUDNN_LIBRARIES})
	endif()
endif()

if(USE_OPENMP)
	target_link_libraries(${module_name} OpenMP::OpenMP_CXX)
endif()

if(SUPPORT_QUANTIZATION)
	target_compile_definitions(${module_name} PRIVATE -DSUPPORT_QUANTIZATION)
endif()

target_link_libraries(${module_name} julius primitives Threads::Threads)
add_dependencies(${module_name} julius primitives)

