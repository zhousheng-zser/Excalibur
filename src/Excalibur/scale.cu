#ifdef USE_CUDA
#include "Excalibur/scale.hpp"
#include <iostream>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		// CUDA kernele for forward
		__global__ void scaleForward(const int n, int channels, int height, int width, float* bottom_data, const float* scales_data, const float* bias_data, orderType order) {
			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = (index / height / width) % channels;
					float scale_data0 = scales_data[c];
					float bias_data0 = bias_data[c];
					bottom_data[index] = bottom_data[index] * scale_data0 + bias_data0;
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = index % channels;
					float scale_data0 = scales_data[c];
					float bias_data0 = bias_data[c];
					bottom_data[index] = bottom_data[index] * scale_data0 + bias_data0;
				}
			}
			else
			{
				return;
			}
		}

		void scale::Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom)
		{
			int bottom_channel = bottom->channels();
			CHECK_EQ(channel_, bottom_channel);
			order_ = bottom->order();
			height_ = bottom->height();
			width_ = bottom->width();
			const int count = bottom->count();
			float* bottom_data = bottom->mutable_gpu_data();
			const float *scales_data = scales_data_->gpu_data();
			const float *bias_data = bias_data_->gpu_data();

			// NOLINT_NEXT_LINE(whitespace/operators)
			scaleForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
				count, channel_, height_, width_, bottom_data, scales_data, bias_data, order_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

#endif