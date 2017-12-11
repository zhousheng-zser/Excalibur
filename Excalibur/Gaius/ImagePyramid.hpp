#pragma once
#ifndef _IMAGEPYRAMID_HPP_
#define _IMAGEPYRAMID_HPP_

#include "ImageTensor.hpp"

namespace excalibur
{
	static void ResizeImageCPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest) {
		int32_t src_width = src->width();
		int32_t src_height = src->height();
		int32_t dest_width = dest->width();
		int32_t dest_height = dest->height();
		if (src_width == dest_width && src_height == dest_height) {
			math_functions::excalibur_copy(src_width * src_height * sizeof(uint8_t), src->cpu_data(), dest->mutable_cpu_data(), -1);
			return;
		}
		double lf_x_scl = static_cast<double>(src_width) / dest_width;
		double lf_y_Scl = static_cast<double>(src_height) / dest_height;
		const uint8_t* src_data = src->cpu_data();
		uint8_t* dest_data = dest->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel num_threads(SEETA_NUM_THREADS)
		{
#pragma omp for nowait
#endif
			for (int32_t y = 0; y < dest_height; y++) {
				for (int32_t x = 0; x < dest_width; x++) {
					double lf_x_s = lf_x_scl * x;
					double lf_y_s = lf_y_Scl * y;

					int32_t n_x_s = static_cast<int>(lf_x_s);
					n_x_s = (n_x_s <= (src_width - 2) ? n_x_s : (src_width - 2));
					int32_t n_y_s = static_cast<int>(lf_y_s);
					n_y_s = (n_y_s <= (src_height - 2) ? n_y_s : (src_height - 2));

					double lf_weight_x = lf_x_s - n_x_s;
					double lf_weight_y = lf_y_s - n_y_s;

					double dest_val = (1 - lf_weight_y) * ((1 - lf_weight_x) *
						src_data[n_y_s * src_width + n_x_s] +
						lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1]) +
						lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s] +
							lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1]);

					dest_data[y * dest_width + x] = static_cast<uint8_t>(dest_val);
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

		inline void SetScaleStep(float step) {
			if (step > 0.0f && step <= 1.0f)
				scale_step_ = step;
		}

		inline void SetMinScale(float min_scale) {
			min_scale_ = min_scale;
		}

		inline void SetMaxScale(float max_scale) {
			max_scale_ = max_scale;
			scale_factor_ = max_scale;
			UpdateBufScaled();
		}

		void SetImage1x(const uint8_t* img_data, int32_t width, int32_t height);

		inline float min_scale() const { return min_scale_; }
		inline float max_scale() const { return max_scale_; }

		inline ImageTensor<unsigned char> image1x() {
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