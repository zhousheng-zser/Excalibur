#ifdef USE_CUDA
#include "upsample.hpp"
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		// CUDA kernele for forward
		__global__ void UpsampleForward(int n, const int channel, const int top_height, const int top_width,
			const float* bottom_data, float* top_data,
			const int scale, orderType order) {

			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int w = n % top_width;
					n = n / top_width;
					int h = n % top_height;
					n = n / top_height;
					int c = n % channel;
					n = n / channel;

					int nh = h / scale;
					int nw = w / scale;

					int top_idx = (((n * channel + c) * top_height) + h) * top_width + w;
					int bottom_idx = (((n * channel + c) * (top_height / scale)) + nh) * (top_width / scale) + nw;

					top_data[top_idx] = bottom_data[bottom_idx];
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = n % channel;
					n = n / channel;
					int w = n % top_width;
					n = n / top_width;
					int h = n % top_height;
					n = n / top_height;

					int nh = h / scale;
					int nw = w / scale;

					int top_idx = (((n * top_height + h) * top_width) + w) * channel + c;
					int bottom_idx = (((n * (top_height / scale) + nh) * (top_width / scale)) + nw) * channel + c;

					top_data[top_idx] = bottom_data[bottom_idx];
				}
			}
			else
			{
				return;
			}
		}

		void upsample::Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = (bottom)->data_shape()[0];
			const float* bottom_data = (bottom)->gpu_data();
			height_ = (bottom)->height();
			width_ = (bottom)->width();
			channel_ = (bottom)->channels();
			order_ = (bottom)->order();

			int top_height = scale_ * height_;
			int top_width = scale_ * width_;

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{num, channel_, top_height, top_width}, device_, order_));
			}
			else if (order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{num, top_height, top_width, channel_}, device_, order_));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			float *top_data = top->mutable_gpu_data();

			int count = top->count();
			// NOLINT_NEXT_LINE(whitespace/operators)
			UpsampleForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
				count, channel_, top_height, top_width, bottom_data, top_data, scale_, order_);
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

#endif