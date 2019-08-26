if(WITH_JNI)
include_directories(../../include/Cassius 
				../../include/Excalibur 
				${JNI_INCLUDE})
else()
include_directories(../../include/Cassius 
				../../include/Excalibur)
endif()

file(GLOB Cassius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Cassius/*.cpp)

if(WITH_JNI)
	file(GLOB Cassius_jni_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../jni/jni/Cassiutia/*.cpp)
endif()

if(WITH_JNI)
	add_library(Cassiutia ${Cassius_source_files} ${Cassius_jni_source_files})
	target_link_libraries(Cassiutia Excalibur ${Excalibur_LINKER_LIBS})
	add_dependencies(Cassiutia Excalibur)
else()
	add_library(Cassius ${Cassius_source_files})
	target_link_libraries(Cassius Excalibur ${Excalibur_LINKER_LIBS})
	add_dependencies(Cassius Excalibur)
endif()
