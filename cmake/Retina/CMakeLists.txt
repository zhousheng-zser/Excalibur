
include_directories(../../include/Retina)

file(GLOB Retina_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Retina/*.cpp)

add_library(Retina STATIC ${Retina_source_files})
