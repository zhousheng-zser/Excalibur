include_directories(../../include/Cassius 
			../../include/Gaius
			../../include/Longinus
			../../include/Irisviel)

set(LonginusCassiusMainSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/LonginusCassiusMain.cpp)
set(IrisvielExampleSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/IrisvielExample.cpp)
set(LonginusSpeedTestSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../test/LonginusSpeedTest.cpp)

set(CMAKE_CXX_STANDARD 14)

#add_executable(LonginusCassiusMain ${LonginusCassiusMainSRC})
add_executable(IrisvielExample ${IrisvielExampleSRC})
add_executable(LonginusSpeedTest ${LonginusSpeedTestSRC})

#target_link_libraries(LonginusCassiusMain stdc++fs opencv_core opencv_imgproc opencv_imgcodecs Longinus Cassius Gaius Excalibur Irisvian ${Excalibur_LINKER_LIBS})
target_link_libraries(LonginusSpeedTest stdc++fs opencv_core opencv_imgproc opencv_imgcodecs Longinus Cassius Gaius Excalibur Irisviel ${Excalibur_LINKER_LIBS})
#add_dependencies(LonginusCassiusMain Longinus Cassius Gaius Excalibur Irisvian)
add_dependencies(LonginusSpeedTest Longinus Cassius Gaius Excalibur Irisviel)
target_link_libraries(IrisvielExample Irisviel ${Excalibur_LINKER_LIBS})
add_dependencies(IrisvielExample Irisviel)
