#include "ImagePyramid.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void ResizeKernel(const unsigned char* src, int src_w, int src_h, unsigned char* dst, int dst_w, int dst_h)
	{

	}


	void ResizeImageGPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest)
	{

	}
}

#endif