#ifdef USE_CUDA
#include "../../include/excalibur/normalize.hpp"

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, float epsilon,
			const float* data, float* norm_data, orderType order) {
			CUDA_KERNEL_LOOP(index, num * spatial_dim) {
				int n = index / spatial_dim;
				int s = index % spatial_dim;
				float sum = 0;

				if (order == NCHW)
				{
					for (int c = 0; c < channels; ++c) {
						sum += data[(n * channels + c) * spatial_dim + s];
					}
				}
				else if (order == NHWC)
				{
					for (int c = 0; c < channels; ++c) {
						sum += data[n * spatial_dim * channels + s * channels + c];
					}
				}
				else
				{
					return;
				}

				norm_data[index] = sum + epsilon;
			}
		}

		__global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dim,
			const float* data, const float* norm_data,
			float* output_data, orderType order) {

			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
					int n = index / channels / spatial_dim;
					int s = index % spatial_dim;
					output_data[index] = data[index] * norm_data[n * spatial_dim + s];
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
					int n = index / channels / spatial_dim;
					int s = (index / channels) % spatial_dim;
					output_data[index] = data[index] * norm_data[n * spatial_dim + s];
				}
			}
			else
			{
				return;
			}
		}

		void normalize::Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom)
		{
			int num = bottom->num();
			int channels = bottom->channels();
			int height = bottom->height();
			int width = bottom->width();
			order_ = bottom->order();

			if (order_ == NCHW)
			{
				squared_.reset(new tensor<float>(std::vector<int>{num, channels, height, width}, device_, order_));
				norm_.reset(new tensor<float>(std::vector<int>{num, 1, height, width}, device_, order_));
				//
				const float* bottom_data = bottom->gpu_data();
				float* top_data = bottom->mutable_gpu_data();
				float* square_data = squared_->mutable_gpu_data();
				float* norm_data = norm_->mutable_gpu_data();
				int spatial_dim = height * width;
				//
				if (type_ == L2)
				{
					math_functions::gpu_powx(num*channels*spatial_dim, bottom_data, 2.0f, square_data);
					kernel_channel_sum << <CUDA_GET_BLOCKS(num*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data, order_);
					math_functions::gpu_powx(num * spatial_dim, norm_data, -0.5f, norm_data);
					kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data, order_);
				}
				else if (type_ == L1)
				{
					math_functions::gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
					kernel_channel_sum << <CUDA_GET_BLOCKS(num*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-6, square_data, norm_data, order_);
					math_functions::gpu_powx(num * spatial_dim, norm_data, -1.0f, norm_data);
					kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data, order_);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else if (order_ == NHWC)
			{
				squared_.reset(new tensor<float>(std::vector<int>{num, height, width, channels}, device_, order_));
				norm_.reset(new tensor<float>(std::vector<int>{num, height, width, 1}, device_, order_));
				//
				const float* bottom_data = bottom->gpu_data();
				float* top_data = bottom->mutable_gpu_data();
				float* square_data = squared_->mutable_gpu_data();
				float* norm_data = norm_->mutable_gpu_data();
				int spatial_dim = height * width;
				//
				if (type_ == L2)
				{
					math_functions::gpu_powx(num*channels*spatial_dim, bottom_data, 2.0f, square_data);
					kernel_channel_sum << <CUDA_GET_BLOCKS(num*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data, order_);
					math_functions::gpu_powx(num * spatial_dim, norm_data, -0.5f, norm_data);
					kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data, order_);
				}
				else if (type_ == L1)
				{
					math_functions::gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
					kernel_channel_sum << <CUDA_GET_BLOCKS(num*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-6, square_data, norm_data, order_);
					math_functions::gpu_powx(num * spatial_dim, norm_data, -1.0f, norm_data);
					kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
						CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data, order_);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}


#endif