if(WITH_JNI)
include_directories(../../include/Gaius 
			../../include/Excalibur 
			${JNI_INCLUDE})
else()
include_directories(../../include/Gaius 
			../../include/Excalibur)
endif()


file(GLOB Gaius_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Gaius/*.cxx)

if(WITH_JNI)
	file(GLOB Gaius_jni_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../jni/jni/Gaiulinya/*.cpp)
endif()

if(WITH_JNI)
	add_library(Gaiulinya ${Gaius_source_files} ${Gaius_jni_source_files})
	target_link_libraries(Gaiulinya Excalibur ${PRIMITIVES_TARGET_NAME} ${Excalibur_LINKER_LIBS})
	add_dependencies(Gaiulinya Excalibur ${PRIMITIVES_TARGET_NAME})
else()
	add_library(Gaius ${Gaius_source_files})
	target_link_libraries(Gaius Excalibur ${PRIMITIVES_TARGET_NAME} ${Excalibur_LINKER_LIBS})
	add_dependencies(Gaius Excalibur ${PRIMITIVES_TARGET_NAME})
endif()
