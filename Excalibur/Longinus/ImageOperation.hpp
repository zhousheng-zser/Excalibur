#ifndef IMAGEOPERATION_HPP
#define IMAGEOPERATION_HPP

#include "common.hpp"

namespace glasssix
{
	namespace longinus
	{
		LONGINUS_DLL void myResize(const unsigned char * psrc, int swidth, int sheight, int sstep,
			unsigned char * pdst, int dwidth, int dheight, int dstep);
		LONGINUS_DLL void myFlip(const unsigned char * psrc, int width, int height, int step,
			unsigned char * pdst);
		LONGINUS_DLL void myIntegral(const unsigned char *pSrc, int width, int height, int step, int *pSum, int sum_width);

		LONGINUS_DLL void resize_cpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
			unsigned char* dst_data, int new_height, int new_width);
#ifdef USE_CUDA
		LONGINUS_DLL void resize_gpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
			unsigned char* dst_data, int new_height, int new_width, int device);

		LONGINUS_DLL void integral_gpu(const unsigned char* src_data, int width, int height, int* dst_data, int sum_width, int sum_height);
#endif
		LONGINUS_DLL void matrix_transpose(const unsigned char *src_data, int height, int width, unsigned char *dst_data);
	}
}

#endif