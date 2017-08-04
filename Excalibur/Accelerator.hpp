#pragma once
#ifndef _ACCELERATOR_HPP_
#define _ACCELERATOR_HPP_
//x86 or ARM platform: if on ARM, comment this macro
//#define x86

//If on x86, choose your accelerator before compiling
#ifdef x86
#define USE_MKL//For Intel platform
#ifdef USE_MKL // If use MKL, simply include the MKL header
#include <mkl.h>
#define USE_MKLDNN
#ifdef USE_MKLDNN // Use MKLDNN
#include "mkldnn.hpp"
#endif // Use MKLDNN
#else   //Else, use OpenBLAS
extern "C" {
#include <cblas.h>
}
#endif
#define USE_CUDA
#ifdef USE_CUDA //GPU, use cuBLAS or CUDNN
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
//#define USE_CUDNN
#ifdef USE_CUDNN
#include <cudnn.hpp> //A wrapper of cudnn
#endif //USE_CUDNN
#endif //USE_CUDA
#else //ARM, default use OpenBLAS
extern "C" {
#include <cblas.h>
}
//#define USE_NEON
#ifdef USE_NEON //Use NEON Instruction Set on ARM
#include"arm_neon.h"
#endif
#endif

//common libraries
#ifdef WIN32
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include <glog/logging.h>
#define CAFFEMODEL_SUPPORT
namespace Excalibur
{
	//Processor platform, default CPU
	enum Avalon { CPU //default OpenBLAS(works on both X86 and ARM)
#ifdef USE_CUDA
		, GPU
#endif
#ifndef x86
		, ARM //NEON accelerated(only works on ARM)
#endif
	};

#ifdef USE_CUDA
	//
	// CUDA macros
	//

	// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << curandGetErrorString(status); \
  } while (0)

	// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

	// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


	// CUDA: library error reporting.
	inline const char* cublasGetErrorString(cublasStatus_t error)
	{
		switch (error) 
		{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
		}
		return "Unknown cublas status";
	}

	inline const char* curandGetErrorString(curandStatus_t error)
	{
		switch (error) {
		case CURAND_STATUS_SUCCESS:
			return "CURAND_STATUS_SUCCESS";
		case CURAND_STATUS_VERSION_MISMATCH:
			return "CURAND_STATUS_VERSION_MISMATCH";
		case CURAND_STATUS_NOT_INITIALIZED:
			return "CURAND_STATUS_NOT_INITIALIZED";
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "CURAND_STATUS_ALLOCATION_FAILED";
		case CURAND_STATUS_TYPE_ERROR:
			return "CURAND_STATUS_TYPE_ERROR";
		case CURAND_STATUS_OUT_OF_RANGE:
			return "CURAND_STATUS_OUT_OF_RANGE";
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "CURAND_STATUS_LAUNCH_FAILURE";
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "CURAND_STATUS_PREEXISTING_FAILURE";
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "CURAND_STATUS_INITIALIZATION_FAILED";
		case CURAND_STATUS_ARCH_MISMATCH:
			return "CURAND_STATUS_ARCH_MISMATCH";
		case CURAND_STATUS_INTERNAL_ERROR:
			return "CURAND_STATUS_INTERNAL_ERROR";
		}
		return "Unknown curand status";
	}

	// CUDA: use 512 threads per block
	const int EXCALIBUR_CUDA_NUM_THREADS = 512;

	// CUDA: number of blocks for threads.
	inline int EXCALIBUR_GET_BLOCKS(const int N) {
		return (N + EXCALIBUR_CUDA_NUM_THREADS - 1) / EXCALIBUR_CUDA_NUM_THREADS;
	}
#endif
}
#endif  //_ACCELERATOR_HPP_