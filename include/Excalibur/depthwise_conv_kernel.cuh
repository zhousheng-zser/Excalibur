#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <glasssix\tensor.hpp>

namespace glasssix
{
	namespace excalibur
	{
		__global__ void depthwise_conv_kernel(const int nthreads,
			const float* const bottom_data, const int num, const int channels,
			const int height, const int width, const int conved_height,
			const int conved_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, const float* const weight, const float* const bias, const bool bias_term_, orderType order);
	}
}

#endif//!USE_CUDA

#endif // !_KERNEL_CUH_