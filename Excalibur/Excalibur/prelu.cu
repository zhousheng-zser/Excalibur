#include "prelu.hpp"
#ifdef USE_CUDA
#include <filesystem>
#include <iostream>
namespace excalibur
{
	// CUDA kernele for forward
	__global__ void PReLUForward(const int n, const int channels, const int dim,
		const float* in, float* out, const float* slope_data,
		const int div_factor) {
		CUDA_KERNEL_LOOP(index, n) {
			int c = (index / dim) % channels / div_factor;
			out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
		}
	}

	void prelu::Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom)
	{
		const float* bottom_data = bottom->gpu_data();
		float* top_data = bottom->mutable_gpu_data();
		const int count = bottom->count();
		int dim;
		if (bottom->data_shape().size()<=2)
		{
			dim = 1;
		}
		else
		{
			dim = bottom->count(2, 4);
		}
		const int channels = bottom->channels();
		const float* slope_data = slope_data_->gpu_data();
		const int div_factor = false ? channels : 1;
		
		// NOLINT_NEXT_LINE(whitespace/operators)
		PReLUForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
			count, channels, dim, bottom_data, top_data, slope_data, div_factor);
		CUDA_POST_KERNEL_CHECK;
	}
}
#endif