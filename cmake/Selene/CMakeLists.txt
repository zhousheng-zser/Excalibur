
include_directories(../../include/Selene)

file(GLOB Selene_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Selene/*.cpp)

add_library(Selene STATIC ${Selene_source_files})
