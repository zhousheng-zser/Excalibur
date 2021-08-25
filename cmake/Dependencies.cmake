find_package(Threads REQUIRED)

if(USE_CUDA)
	find_package(CUDA REQUIRED)
	if(${CUDA_VERSION} VERSION_LESS 11.1)
		message(FATAL_ERROR "CUDA version is too lower(${CUDA_VERSION} vs 11.1)")
	endif()
endif()

if(USE_OPENMP)
	find_package(OpenMP REQUIRED)
	add_compile_options(${OpenMP_CXX_FLAGS})
endif()

#include(cmake/Cuda.cmake)
