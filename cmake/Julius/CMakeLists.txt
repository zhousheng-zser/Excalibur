#cmake_minimum_required(VERSION 3.0.0)
#project(Julius)


include_directories(../../include/Julius)
file(GLOB Julius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Julius/*.cpp)

add_library(Julius STATIC ${Julius_source_files})
#target_link_libraries(Julius ${Excalibur_LINKER_LIBS})
