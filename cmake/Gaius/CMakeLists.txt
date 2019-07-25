
include_directories(../../include/Gaius 
			../../include/Excalibur)
file(GLOB Gaius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cpp)
add_library(Gaius ${Gaius_source_files})

target_link_libraries(Gaius Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(Gaius Excalibur)
