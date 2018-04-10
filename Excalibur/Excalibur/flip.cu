#include "flip.hpp"
#ifdef USE_CUDA

namespace excalibur
{

	__global__ void FlipKernel(const int num, const int channels, const int height, const int width,
		const float* src_data, float* target_data, bool flip_height, bool flip_width) {
		CUDA_KERNEL_LOOP(index, num * channels * height * width) {
			int n = index / (channels * height * width);
			int cs = index % (channels * height * width);
			int c = cs / (height * width);
			int s = cs % (height * width);
			int h = s / width;
			int w = s % width;
			target_data[(((n * channels + c) * height + h) * width) + w] =
				src_data[(((n * channels + c) * height + (flip_height ? (height - 1 - h) : h)) * width) + (flip_width ? (width - 1 - w) : w)];
		}
	}

	void flip::Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		top.reset(new tensor<float>(bottom->data_shape(), device_));
		const float* bottom_data = bottom->cpu_data();
		float* top_data = top->mutable_cpu_data();
		int num = bottom->num();
		int channels = bottom->channels();
		int width = bottom->width();
		int height = bottom->height();

		FlipKernel << <CUDA_GET_BLOCKS(num * channels * height * width),
			CUDA_NUM_THREADS >> >(num, channels, height, width,
				bottom_data, top_data, flip_height_, flip_width_);
	}
}


#endif