if(USE_CUDA)
	find_package(CUDA REQUIRED)
endif()

if(USE_OPENMP)
	find_package(OpenMP REQUIRED)
	add_compile_options(${OpenMP_CXX_FLAGS})
endif()

#include(cmake/Cuda.cmake)
