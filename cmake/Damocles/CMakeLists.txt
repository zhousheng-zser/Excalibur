#cmake_minimum_required(VERSION 3.0.0)
#project(Damocles)

#find_package(CUDA)
#add_definitions(-Dx86)

include_directories(../../include/Damocles 
			../../include/Excalibur)
file(GLOB Damocles_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Damocles/*.cpp)

add_library(Damocles STATIC ${Damocles_source_files})

target_link_libraries(Damocles Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Damocles Excalibur)
