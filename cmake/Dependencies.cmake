if(USE_CUDA)
	find_package(CUDA REQUIRED)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

if(USE_CUDNN)
	include_directories(/usr/local/cuda/include)
	link_directories(/usr/local/cuda/lib64)
endif()

if(USE_OPENMP)
	find_package(OpenMP REQUIRED)
	add_compile_options(${OpenMP_CXX_FLAGS})
endif()

#include(cmake/Cuda.cmake)
