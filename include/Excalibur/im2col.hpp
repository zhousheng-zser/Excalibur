#pragma once
#ifndef _IM2COL_HPP_
#define _IM2COL_HPP_
#include "../../include/Primitives/tensor.hpp"

namespace glasssix
{
	namespace excalibur
	{

		void im2col_nd_cpu(const float* data_im, const int num_spatial_axes,
			const int* im_shape, const int* col_shape,
			const int* kernel_shape, const int* pad, const int* stride,
			const int* dilation, float* data_col);


		void im2col_cpu(const float* data_im, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w, const int stride_h,
			const int stride_w, const int dilation_h, const int dilation_w,
			float* data_col, memory::orderType order = memory::NCHW, int num = 1);

		void im2col_cpu(const signed char* data_im, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w, const int stride_h,
			const int stride_w, const int dilation_h, const int dilation_w,
			signed char* data_col, memory::orderType order = memory::NCHW, int num = 1);

		void col2im_nd_cpu(const float* data_col, const int num_spatial_axes,
			const int* im_shape, const int* col_shape,
			const int* kernel_shape, const int* pad, const int* stride,
			const int* dilation, float* data_im);

		void col2im_cpu(const float* data_col, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w, const int stride_h,
			const int stride_w, const int dilation_h, const int dilation_w,
			float* data_im);
#ifdef USE_CUDA
		template <typename Dtype>
		void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
			const int col_size, const int* im_shape, const int* col_shape,
			const int* kernel_shape, const int* pad, const int* stride,
			const int* dilation, Dtype* data_col);

		template <typename Dtype>
		void im2col_gpu(const Dtype* data_im, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w, const int stride_h,
			const int stride_w, const int dilation_h, const int dilation_w,
			Dtype* data_col, memory::orderType order = memory::NCHW, int num = 1);


		template <typename Dtype>
		void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
			const int im_size, const int* im_shape, const int* col_shape,
			const int* kernel_shape, const int* pad, const int* stride,
			const int* dilation, Dtype* data_im);

		template <typename Dtype>
		void col2im_gpu(const Dtype* data_col, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w, const int stride_h,
			const int stride_w, const int dilation_h, const int dilation_w,
			Dtype* data_im);
#endif
	}
}


#endif // _IM2COL_HPP_