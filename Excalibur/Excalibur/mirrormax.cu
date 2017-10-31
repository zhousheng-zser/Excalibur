#include "mirrormax.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void MirrorMax(const int nthreads, const float* bottom_data, float* top_data)
	{
		CUDA_KERNEL_LOOP(index, nthreads) {
			top_data[index] = max(bottom_data[index], bottom_data[index + nthreads]);
		}
	}

	void mirrormax::Forward_native_gpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		if (mirror_axis_ == 0)
		{
			int num = bottom->num();
			int channels = bottom->channels();
			int height = bottom->height();
			int width = bottom->width();
			CHECK_EQ(num % 2, 0);
			top.reset(new tensor(std::vector<int>{num / 2, channels, height, width}, device_));
			const float* bottom_data = bottom->gpu_data();
			float* top_data = top->mutable_gpu_data();
			const int mirror_offset = num / 2 * channels * height * width;
			MirrorMax << <CUDA_GET_BLOCKS(mirror_offset), CUDA_NUM_THREADS >> > (
				mirror_offset, bottom_data, top_data);
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
}

#endif