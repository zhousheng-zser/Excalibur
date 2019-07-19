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
set(LonginusSpeedTestSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/LonginusSpeedTest.cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(LonginusCassiusMain ${LonginusCassiusMainSRC})
add_executable(IrisvianExample ${IrisvianExampleSRC})
add_executable(LonginusSpeedTest ${LonginusSpeedTestSRC})

target_link_libraries(LonginusCassiusMain stdc++fs opencv_core opencv_imgproc opencv_imgcodecs Longinus Cassius Gaius Excalibur Irisvian ${Excalibur_LINKER_LIBS})
target_link_libraries(LonginusSpeedTest stdc++fs opencv_core opencv_imgproc opencv_imgcodecs Longinus Cassius Gaius Excalibur Irisvian ${Excalibur_LINKER_LIBS})
add_dependencies(LonginusCassiusMain Longinus Cassius Gaius Excalibur Irisvian)
add_dependencies(LonginusSpeedTest Longinus Cassius Gaius Excalibur Irisvian)
target_link_libraries(IrisvianExample Irisvian ${Excalibur_LINKER_LIBS})
add_dependencies(IrisvianExample Irisvian)
