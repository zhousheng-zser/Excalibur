cmake_minimum_required(VERSION 3.10.2)
project(Excalibur C CXX)

option(USE_CUDA "Build Excalibur with CUDA support" OFF)
option(USE_CUDNN "Build Excalibur with cuDNN library support" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(HAVE_SSE "CPU SSE support" OFF)
option(HAVE_AVX "CPU AVX support" OFF)
option(HAVE_AVX2 "CPU AVX2 support" OFF)

if(HAVE_AVX2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()
if(HAVE_AVX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
endif()
if(HAVE_SSE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mf16c")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

include(Dependencies.cmake)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
if(UNIX OR APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -fpermissive")
endif()

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Primitives)
add_subdirectory(Julius)
add_subdirectory(Excalibur)
add_subdirectory(SpeedBenchmark)
