project(Excalibur)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

if(UNIX OR APPLE)
	set(CMAKE_MACOSX_RPATH 0)
	
	AUX_SOURCE_DIRECTORY(. DIR_SRCS)
	
	SET(CMAKE_C_COMPILER "emcc")
	
	SET(CMAKE_CXX_COMPILER "em++")
	
	set(CMAKE_CXX_STANDARD 11)
	
	set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")
	
	add_definitions("-std=c++11")
	
	add_definitions("-Dplatform=x86")
	
	add_definitions(-D x86)
	
	message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
	message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
	
	include_directories(/Users/glasssix-mk/Documents/code /Users/glasssix-mk/Documents/Julius /usr/include/architecture/i386)
	
	link_directories(/Users/glasssix-mk/Documents/Julius)
	
	ADD_LIBRARY(Excalibur SHARED ${DIR_SRCS})
	
	target_link_libraries(Excalibur libJulius.so)
endif()