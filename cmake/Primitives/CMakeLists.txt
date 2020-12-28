cmake_minimum_required(VERSION 3.14.3)
set(module_name primitives)
include_directories(../../include/Primitives)

file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/*.cpp 
					${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/abi/*.cpp 
					${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/fmt/*.cc
					${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/nlohmann/*.cpp)

file(GLOB header ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Primitives/*.hpp 
					${CMAKE_CURRENT_SOURCE_DIR}/../../include/Primitives/abi/*.hpp 
					${CMAKE_CURRENT_SOURCE_DIR}/../../include/Primitives/fmt/*.h
					${CMAKE_CURRENT_SOURCE_DIR}/../../include/Primitives/nlohmann/*.hpp)

list(REMOVE_ITEM src "${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/log.cpp")
list(REMOVE_ITEM header "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Primitives/log.hpp")

add_library(${module_name} SHARED ${src} ${header})

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_definitions(${module_name} PRIVATE -DFMT_EXPORT -DEXPORT_EXCALIBUR_PRIMITIVES -D_WINDLL)
endif()

if(USE_CUDA)
	target_include_directories(${module_name} SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
	target_compile_definitions(${module_name} PRIVATE -DUSE_CUDA)
	target_link_libraries(${module_name} ${CUDA_LIBRARIES})
endif()

if(USE_CUDNN)
	target_include_directories(${module_name} PRIVATE ${CUDNN_INCLUDE_DIRS})
	target_link_directories(${module_name} PRIVATE ${CUDNN_LIBRARY_DIRS})
	target_compile_definitions(${module_name} PRIVATE -DUSE_CUDNN)
	target_link_libraries(${module_name} ${CUDNN_LIBRARIES})
endif()

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
	target_link_libraries(${module_name} ${CMAKE_DL_LIBS})
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	target_link_libraries(${module_name} stdc++fs)
endif()

if(USE_OPENMP)
	target_link_libraries(${module_name} OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${module_name} Threads::Threads)
