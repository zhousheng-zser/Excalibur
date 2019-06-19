#cmake_minimum_required(VERSION 3.0.0)
#project(Cassius)

#find_package(CUDA)
#add_definitions(-Dx86)

include_directories(../../include/Cassius 
			../../include/Excalibur)

file(GLOB Cassius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Cassius/*.cpp)

add_library(Cassius ${Cassius_source_files})

target_link_libraries(Cassius Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Cassius Excalibur)
