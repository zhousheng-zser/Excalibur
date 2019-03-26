#pragma once
#ifndef _TENSOR_OPERATION_GPU_HPP_
#define _TENSOR_OPERATION_GPU_HPP_
#ifdef USE_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glasssix\accelerator.hpp>
#include <glasssix\tensor.hpp>
#include <iostream>
#ifdef USE_OPENCV
#include <opencv2\opencv.hpp>
#endif
#include <glasssix\timer.hpp>


namespace glasssix 
{
	namespace excalibur
	{
		class tensor_operation_gpu
		{
		public:
			tensor_operation_gpu() {};
			~tensor_operation_gpu() {};


#ifdef USE_OPENCV

			template <typename Dtype>
			static void tensor2mat_gpu(const std::shared_ptr<tensor<Dtype>> &src, cv::Mat& dst);

			template <typename Dtype>

			static void tensor2mat_gpu(const tensor<Dtype>& src, cv::Mat& dst);



			template <typename Dtype>
			static void mat2tensor_gpu(const cv::Mat &src, std::shared_ptr<tensor<Dtype>>& dst, orderType order = NHWC);

			template <typename Dtype>
			static void mat2tensor_gpu(const cv::Mat &src, tensor<Dtype>& dst, orderType order = NHWC);

#endif // USE_OPENCV


			template <typename Dtype>
			static void nchw2nhwc_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst);

			template <typename Dtype>
			static void nchw2nhwc_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst);



			template <typename Dtype>
			static void nhwc2nchw_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst);

			template <typename Dtype>
			static void nhwc2nchw_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst);



			template <typename Dtype>
			static void resize_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
				int dst_height, int dst_width, interpolationType type = Bilinear);

			template <typename Dtype>
			static void resize_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
				int dst_height, int dst_width, interpolationType type = Bilinear);



			template <typename Dtype>
			static void rotate_with_center_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
				float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);

			template <typename Dtype>
			static void rotate_with_center_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
				float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);



			template <typename Dtype, typename Ptype>
			static void rotate_with_points_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);

			template <typename Dtype, typename Ptype>
			static void rotate_with_points_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);



			template <typename Dtype>
			static void flip_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, flipType axis = Width_Wise);

			template <typename Dtype>
			static void flip_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, flipType axis = Width_Wise);



			template <typename Dtype>
			static void rgb2gray_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst);

			template <typename Dtype>
			static void rgb2gray_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst);



			template <typename Dtype>
			static void matrix_transpose_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst);

			template <typename Dtype>
			static void matrix_transpose_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst);



			template <typename Dtype, typename Rtype>
			static void roi_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, excalibur::rectangle<Rtype> rect);

			template <typename Dtype, typename Rtype>
			static void roi_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, excalibur::rectangle<Rtype> rect);



			template <typename Dtype>
			static void threshold_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);

			template <typename Dtype>
			static void threshold_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);



			template <typename Dtype, typename Ptype>
			static void warp_affine_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

			template <typename Dtype, typename Ptype>
			static void warp_affine_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);



			template <typename Dtype>
			static void gaussian_blur_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int ksize = 3);

			template <typename Dtype>
			static void gaussian_blur_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int ksize = 3);



			template <typename Dtype>
			static void sobel_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int dx = 1, int dy = 1);

			template <typename Dtype>
			static void sobel_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int dx = 1, int dy = 1);



			template <typename Dtype>
			static void morph_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, excalibur::morphType type = Dilate, int ksize = 3);

			template <typename Dtype>
			static void morph_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, excalibur::morphType type = Dilate, int ksize = 3);



			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_gpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst);



			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_gpu(const tensor<DtypeSRC>* src, tensor<DtypeDST>* dst);



			template <typename Dtype>
			static void preprocess_tensors_gpu(std::shared_ptr<tensor<Dtype>> dst);



			template <typename Dtype>
			static void preprocess_tensors_gpu(tensor<Dtype>* dst);



			template <typename Dtype>
			static void make_border_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
				int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);


			template <typename Dtype>
			static void cut_border_gpu(const std::shared_ptr<tensor<Dtype>> &src,
				std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right);


			template <typename Dtype, typename Rtype>
			static void safty_cut_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, rectangle<Rtype>* rect);


			template <typename Dtype>
			static void equalize_hist_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst);


			template <typename Dtype>
			static void merge_channel_gpu(const std::vector<std::shared_ptr<tensor<Dtype>>> &src_vector, std::shared_ptr<tensor<Dtype>> &dst);

		};
	}
}
#endif //!USE_CUDA
#endif // !_TENSOR_OPERATION_GPU_HPP_