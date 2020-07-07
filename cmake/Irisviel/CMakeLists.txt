if(USE_CUDA)
	remove_definitions(-DUSE_CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)

if(WITH_JNI)
include_directories(../../include/Irisviel 
			${JNI_INCLUDE})
else()
include_directories(../../include/Irisviel)
endif()

file(GLOB Irisviel_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Irisviel/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Irisviel/*.cxx)

if(WITH_JNI)
	file(GLOB Irisviel_jni_source_files ${CMAKE_CURRENT_SOURCE_DIR}/../../jni/jni/Irisvika/*.cpp)
endif()

if(WITH_JNI)
	add_library(Irisvika ${Irisviel_source_files} ${Irisviel_jni_source_files})
	target_link_libraries(Irisvika Primitives stdc++fs)
	add_dependencies(Irisvika Primitives)
else()
	add_library(Irisviel ${Irisviel_source_files})
	target_link_libraries(Irisviel Primitives stdc++fs)
	add_dependencies(Irisviel Primitives)
endif()

