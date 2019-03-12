#ifndef IMAGEOPERATION_HPP
#define IMAGEOPERATION_HPP

#include "common.hpp"

namespace glasssix
{
	namespace longinus
	{
		EXPORT_LONGINUS void myResize(const unsigned char * psrc, int swidth, int sheight, int sstep,
			unsigned char * pdst, int dwidth, int dheight, int dstep);
		EXPORT_LONGINUS void myFlip(const unsigned char * psrc, int width, int height, int step,
			unsigned char * pdst);
		EXPORT_LONGINUS void myIntegral(const unsigned char *pSrc, int width, int height, int step, int *pSum, int sum_width);

		EXPORT_LONGINUS void resize_cpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
			unsigned char* dst_data, int new_height, int new_width);
#ifdef USE_CUDA
		EXPORT_LONGINUS void resize_gpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
			unsigned char* dst_data, int new_height, int new_width, int device);

		EXPORT_LONGINUS void integral_gpu(const unsigned char* src_data, int width, int height, int* dst_data, int sum_width, int sum_height);
#endif
		EXPORT_LONGINUS void matrix_transpose(const unsigned char *src_data, int height, int width, unsigned char *dst_data);
	}
}

#endif