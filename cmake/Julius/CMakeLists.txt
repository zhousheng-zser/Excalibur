project(Julius)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

if(UNIX OR APPLE)
	set(CMAKE_MACOSX_RPATH 0)
	
	include_directories(../)
	
	AUX_SOURCE_DIRECTORY(. DIR_SRCS)
	
	SET(CMAKE_C_COMPILER "emcc")
	
	SET(CMAKE_CXX_COMPILER "em++")
	
	set(CMAKE_CXX_STANDARD 11)
	
	set(CMAKE_CXX_FLAGS " -Wall ${CMAKE_CXX_FLAGS}")
	
	add_definitions(-D x86)
	
	if (SSE3_FOUND)
	set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse3")
	endif()
	if (SSSE3_FOUND)
	set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -mssse3")
	endif()
	if (SSE4_1_FOUND)
	set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse4.1")
	endif()
	if (SSE4_2_FOUND)
	set (CMAKE_C_FLAGS_ALL "${CMAKE_C_FLAGS_ALL} -msse4.2")
	endif()
	
	add_definitions("-std=c++11")
	
	ADD_LIBRARY(Julius SHARED ${DIR_SRCS})
endif()