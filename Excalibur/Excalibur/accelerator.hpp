#pragma once
#ifndef _ACCELERATOR_HPP_
#define  _ACCELERATOR_HPP_

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Excalibur: check mode."
#define X86
#define USE_OPENCV
//#define USE_CUDA
//#define USE_CUDNN
//#define USE_MKL
//#define USE_MKLDNN

#ifdef USE_CUDA
#include <cublas_v2.h>
//#include <cusparse_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace excalibur
{
	const char* cublasGetErrorString(cublasStatus_t error);

	// CUDA: use 512 threads per block
	const int CUDA_NUM_THREADS = 512;

	// CUDA: number of blocks for threads.
	inline int CUDA_GET_BLOCKS(const int N) {
		return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
	}
}

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << excalibur::cublasGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())



#endif

#include <glog/logging.h>
#include<memory>
#endif // _ACCELERATOR_HPP_