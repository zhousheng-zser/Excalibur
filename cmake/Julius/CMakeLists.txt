
include_directories(../../include/Julius 
			../../include)
file(GLOB Julius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Julius/*.cpp)

add_library(Julius STATIC ${Julius_source_files})
