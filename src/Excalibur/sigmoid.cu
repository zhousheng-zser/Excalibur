#ifdef USE_CUDA
#include "Excalibur/sigmoid.hpp"
#include <iostream>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		// CUDA kernele for forward
		__global__ void sigmoidForward(const int n, const float* in, float* out) {
			CUDA_KERNEL_LOOP(index, n) {
				out[index] = 1. / (1 + exp(-in[index]));
			}
		}

		void sigmoid::Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom)
		{
			const float* bottom_data = bottom->gpu_data();
			float* top_data = bottom->mutable_gpu_data();
			const int count = bottom->count();

			// NOLINT_NEXT_LINE(whitespace/operators)
			sigmoidForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
				count, bottom_data, top_data);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

#endif