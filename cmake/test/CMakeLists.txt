option(USE_OPENCV "Use opencv library" ON)
if(USE_OPENCV)
	add_definitions(-DUSE_OPENCV)
endif()

include_directories(../../include/Cassius 
			../../include/Gaius
			../../include/Longinus
			../../include/Irisvian)

set(LonginusCassiusMainSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/LonginusCassiusMain.cpp)
set(IrisvianExampleSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/IrisvianExample.cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(LonginusCassiusMain ${LonginusCassiusMainSRC})
add_executable(IrisvianExample ${IrisvianExampleSRC})

target_link_libraries(LonginusCassiusMain stdc++fs opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui Longinus Cassius Gaius Excalibur ${Excalibur_LINKER_LIBS})
add_dependencies(LonginusCassiusMain Longinus Cassius Gaius Excalibur)
target_link_libraries(IrisvianExample Irisvian ${Excalibur_LINKER_LIBS})
add_dependencies(IrisvianExample Irisvian)
