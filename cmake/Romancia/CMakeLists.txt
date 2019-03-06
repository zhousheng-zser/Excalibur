project(Romancia)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

if(UNIX OR APPLE)
	set(CMAKE_MACOSX_RPATH 0)
	
	AUX_SOURCE_DIRECTORY(. DIR_SRCS)
	
	SET(CMAKE_C_COMPILER "emcc")
	
	SET(CMAKE_CXX_COMPILER "em++")
	
	set(CMAKE_CXX_STANDARD 11)
	
	set(CMAKE_CXX_FLAGS " -Wall ${CMAKE_CXX_FLAGS}")
	
	add_definitions("-std=c++11")
	
	add_definitions("-Dplatform=x86")
	
	add_definitions(-D x86)
	
	link_directories(/Users/glasssix-mk/Documents/code/Excalibur/Excalibur/build /Users/glasssix-mk/Documents/Julius/build)
	
	include_directories(../../ /Users/glasssix-mk/Documents/Julius)
	
	ADD_LIBRARY(Romancia SHARED ${DIR_SRCS})
	
	target_link_libraries(Romancia Julius Excalibur)
endif()