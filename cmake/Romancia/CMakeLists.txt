
include_directories(../../include/Romancia ../../include/Primitives ../../include)

file(GLOB Romancia_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Romancia/*.cpp)

add_library(Romancia STATIC ${Romancia_source_files})
