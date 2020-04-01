#pragma once
#ifndef _BLAS_HPP_
#define _BLAS_HPP_
#include "cpu.hpp"
#include "gpu.hpp"

#if defined(Parallel) && defined(USE_MKL) && defined(x86)
#include <mkl.h> // USE INTEL MKL
#elif defined(USE_OPENBLAS)
#include <cblas.h> // USE OpenBLAS
#else
#include "../../include/Julius/julius.hpp" //USE Glasssix Julius BLAS
#endif

#if defined(USE_CUDA) && defined(x86)
#include <cublas_v2.h>
namespace glasssix
{
	static const char* cublasGetErrorString(cublasStatus_t error)
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
}

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << glasssix::excalibur::cublasGetErrorString(status); \
  } while (0)
#endif // !USE_CUBLAS


#endif // !_BLAS_HPP_
