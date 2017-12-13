#pragma once
#ifndef _IMAGEPYRAMID_HPP_
#define _IMAGEPYRAMID_HPP_

#include "ImageTensor.hpp"
#include <iostream>

namespace excalibur
{
	

	static void ResizeImageCPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest) {
		int32_t src_width = src->width();
		int32_t src_height = src->height();
		int32_t dest_width = dest->width();
		int32_t dest_height = dest->height();
		/*if (src_width == dest_width && src_height == dest_height) {
			math_functions::excalibur_copy(src_width * src_height * sizeof(uint8_t), src->cpu_data(), dest->mutable_cpu_data(), -1);
			return;
		}*/
		//int dstep = (((dest_width * 8 + 7) / 8) + 4 - 1) & (~(4 - 1));
		const unsigned char* src_data = src->cpu_data();
		unsigned char* dst_data = dest->mutable_cpu_data();
		float x_ratio = ((float)(src_width )) / dest_width;
		float y_ratio = ((float)(src_height)) / dest_height;
		/*for (int i = 0; i < 100; i++)
		{
			std::cout << (int)src_data[i] << " ";
		}*/
#ifdef _OPENMP
		omp_set_num_threads(OMP_NUM_THREADS);
#pragma omp parallel //num_threads(8)
		{
#pragma omp for //schedule(static, w /24)
#endif
			for (int i = 0; i < dest_height; i++)
			{
				unsigned char a, b, c, d;
				int x, y, index;
				float x_diff, y_diff;
				unsigned char gray;
				int offset = i*dest_width;
				for (int j = 0; j < dest_width; j++)
				{
					x = (int)(x_ratio * j);
					y = (int)(y_ratio * i);
					x_diff = (x_ratio * j) - x;
					y_diff = (y_ratio * i) - y;
					index = (y*src_width + x);
					a = src_data[index];
					b = src_data[index + 1];
					c = src_data[index + src_width];
					d = src_data[index + src_width + 1];
					gray = (a )*(1 - x_diff)*(1 - y_diff) + (b )*(x_diff)*(1 - y_diff) + (c )*(y_diff)*(1 - x_diff) + (d )*(x_diff*y_diff);
					dst_data[offset + j] = gray;/*src_data[(int)(i  * src_height / x_ratio + j / y_ratio)];*/
				}
			}
#ifdef _OPENMP
	}
#endif
	}

	class ImagePyramid
	{
		void UpdateBufScaled();

		float max_scale_;
		float min_scale_;

		float scale_factor_;
		float scale_step_;

		int32_t width1x_;
		int32_t height1x_;

		int32_t width_scaled_;
		int32_t height_scaled_;

		unsigned char* cpu_buf_img_;
		unsigned char* gpu_buf_img_;
		int32_t buf_img_width_;
		int32_t buf_img_height_;

		unsigned char* cpu_buf_img_scaled_;
		unsigned char* gpu_buf_img_scaled_;
		int32_t buf_scaled_width_;
		int32_t buf_scaled_height_;

		std::shared_ptr<ImageTensor<unsigned char>> img_scaled_;

		int device_;
	public:
		ImagePyramid(int device)
			: max_scale_(1.0f), min_scale_(1.0f),
			scale_factor_(1.0f), scale_step_(0.8f),
			width1x_(0), height1x_(0),
			width_scaled_(0), height_scaled_(0),
			buf_img_width_(2), buf_img_height_(2),
			buf_scaled_width_(2), buf_scaled_height_(2), device_(device)
		{
			/*buf_img_.reset(new ImageTensor<unsigned char>(buf_img_width_, buf_img_height_, 1, device_));
			buf_img_scaled_.reset(new ImageTensor<unsigned char>(buf_scaled_width_, buf_scaled_height_, 1, device_));*/
			cpu_buf_img_ = new unsigned char[buf_img_width_ * buf_img_height_];
			cpu_buf_img_scaled_ = new unsigned char[buf_scaled_width_ * buf_scaled_height_];
		}

		~ImagePyramid() 
		{
			delete[] cpu_buf_img_;
			cpu_buf_img_ = nullptr;

			buf_img_width_ = 0;
			buf_img_height_ = 0;

			delete[] cpu_buf_img_scaled_;
			cpu_buf_img_scaled_ = nullptr;

			buf_scaled_width_ = 0;
			buf_scaled_height_ = 0;

			img_scaled_ = nullptr;
		}

		void SetScaleStep(float step) {
			if (step > 0.0f && step <= 1.0f)
				scale_step_ = step;
		}

		void SetMinScale(float min_scale) {
			min_scale_ = min_scale;
		}

		void SetMaxScale(float max_scale) {
			max_scale_ = max_scale;
			scale_factor_ = max_scale;
			UpdateBufScaled();
		}

		void SetImage1x(const uint8_t* img_data, int32_t width, int32_t height);

		float min_scale() const { return min_scale_; }
		float max_scale() const { return max_scale_; }

		ImageTensor<unsigned char> image1x() {
			ImageTensor<unsigned char> img(width1x_, height1x_, 1, device_);
			if (device_>=0)
			{
				math_functions::excalibur_copy(width1x_ * height1x_, cpu_buf_img_, img.mutable_gpu_data(), device_);
			}
			else
			{
				math_functions::excalibur_copy(width1x_ * height1x_, cpu_buf_img_, img.mutable_cpu_data(), device_);
			}
			return img;
		}

		const std::shared_ptr<ImageTensor<unsigned char>> GetNextScaleImage(float* scale_factor = nullptr);
	};
}

#endif // _IMAGEPYRAMID_HPP_