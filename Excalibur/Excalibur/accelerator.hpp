#pragma once
#ifndef _ACCELERATOR_HPP_
#define  _ACCELERATOR_HPP_

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."


#define USE_CUDA
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)


#endif

#include <glog/logging.h>
#include<memory>
#endif // _ACCELERATOR_HPP_