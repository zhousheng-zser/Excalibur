#include "ImagePyramid.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void ResizeKernel(const int nthreads, const unsigned char* src, int src_w, int src_h, unsigned char* dst, int dst_w, int dst_h)
	{
		float x_ratio = ((float)(src_w)) / dst_w;
		float y_ratio = ((float)(src_h)) / dst_h;
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			const int i = index / dst_w;
			const int j = index - (i*dst_w);
			const int x = (int)(x_ratio * j);
			const int y = (int)(y_ratio * i);
			const float x_diff = (x_ratio * j) - x;
			const float y_diff = (y_ratio * i) - y;
			const int idx = (y*src_w + x);
			const unsigned char a = src[idx];
			const unsigned char b = src[idx + 1];
			const unsigned char c = src[idx + src_w];
			const unsigned char d = src[idx + src_w + 1];
			dst[index] = (a)*(1 - x_diff)*(1 - y_diff) + (b)*(x_diff)*(1 - y_diff) + (c)*(y_diff)*(1 - x_diff) + (d)*(x_diff*y_diff);
		}
	}


	void ImagePyramid::ResizeImageGPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest)
	{
		int src_width = src->width();
		int src_height = src->height();
		int dest_width = dest->width();
		int dest_height = dest->height();
		if (src_width == dest_width && src_height == dest_height) {
			math_functions::excalibur_copy(src_width * src_height, src->gpu_data(), dest->mutable_gpu_data(), -1);
			return;
		}
		const unsigned char* src_data = src->gpu_data();
		unsigned char* dst_data = dest->mutable_gpu_data();
		const int count = dest_width * dest_height;
		ResizeKernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >
			(count, src_data, src_width, src_height, dst_data, dest_width, dest_height);
	}
}

#endif