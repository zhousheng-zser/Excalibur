#include "normalize.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, float epsilon,
		const float* data, float* norm_data) {
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
			int n = index / spatial_dim;
			int s = index % spatial_dim;
			float sum = 0;
			for (int c = 0; c < channels; ++c) {
				sum += data[(n * channels + c) * spatial_dim + s];
			}
			norm_data[index] = sum + epsilon;
		}
	}

	__global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dim,
		const float* data, const float* norm_data,
		float* output_data) {
		CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
			int n = index / channels / spatial_dim;
			int s = index % spatial_dim;
			output_data[index] = data[index] * norm_data[n * spatial_dim + s];
		}
	}

	void normalize::Forward_native_gpu(const std::shared_ptr<tensor>& bottom)
	{
		int num = bottom->num();
		int channels = bottom->channels();
		int height = bottom->height();
		int width = bottom->width();
		squared_.reset(new tensor(std::vector<int>{num, channels, height, width}, device_));
		norm_.reset(new tensor(std::vector<int>{num, 1, height, width}, device_));
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
				CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data);
			math_functions::gpu_powx(num * spatial_dim, norm_data, -0.5f, norm_data);
			kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
				CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data);
		}
		else if (type_ == L1)
		{
			math_functions::gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
			kernel_channel_sum << <CUDA_GET_BLOCKS(num*spatial_dim),
				CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-6, square_data, norm_data);
			math_functions::gpu_powx(num * spatial_dim, norm_data, -1.0f, norm_data);
			kernel_channel_scale << <CUDA_GET_BLOCKS(num*channels*spatial_dim),
				CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data);
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
}

#endif