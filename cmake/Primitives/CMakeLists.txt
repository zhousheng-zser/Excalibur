include_directories(../../include/Primitives)

set(CMAKE_CXX_STANDARD 17)

file(GLOB PRIMITIVES_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/*.cpp 
					${CMAKE_CURRENT_SOURCE_DIR}/../../src/Primitives/abi/*.cpp)
add_library(Primitives SHARED ${PRIMITIVES_SOURCE_FILES})
if(USE_CUDA)
	target_link_libraries(Primitives ${CUDA_LIBRARIES} dl)
else()
	target_link_libraries(Primitives ${CUDA_LIBRARIES} dl)
endif()

