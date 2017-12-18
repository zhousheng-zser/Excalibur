#pragma once
#ifndef _IMAGEPYRAMID_HPP_
#define _IMAGEPYRAMID_HPP_

#include "ImageTensor.hpp"
#include <iostream>

namespace excalibur
{
	class ImagePyramid
	{
		float max_scale_;
		float min_scale_;

		float scale_factor_;
		float scale_step_;

		int width1x_;
		int height1x_;

		int width_scaled_;
		int height_scaled_;

		std::shared_ptr<ImageTensor<unsigned char>> img_scaled_;
		std::shared_ptr<ImageTensor<unsigned char>> img_ori_;

		int device_;
	public:
		ImagePyramid(int device)
			: max_scale_(1.0f), min_scale_(1.0f),
			scale_factor_(1.0f), scale_step_(0.8f),
			 device_(device)
		{
			// legal device check only
			if (device_>=0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif
			}
		}

		~ImagePyramid() 
		{
			img_ori_ = nullptr;
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
		}

		void SetImage1x(const unsigned char* img_data, int width, int height);

		float min_scale() const { return min_scale_; }
		float max_scale() const { return max_scale_; }

		const std::shared_ptr<ImageTensor<unsigned char>> GetNextScaleImage(float* scale_factor = nullptr);

#ifdef USE_CUDA
		static void ResizeImageGPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest, int device = 0);
#endif

		static void ResizeImageCPU(const std::shared_ptr<ImageTensor<unsigned char>>src, std::shared_ptr<ImageTensor<unsigned char>> & dest)
		{
			int src_width = src->width();
			int src_height = src->height();
			int dest_width = dest->width();
			int dest_height = dest->height();
			if (src_width == dest_width && src_height == dest_height) {
				math_functions::excalibur_copy(src_width * src_height, src->cpu_data(), dest->mutable_cpu_data(), -1);
				return;
			}
			const unsigned char* src_data = src->cpu_data();
			unsigned char* dst_data = dest->mutable_cpu_data();
			float x_ratio = ((float)(src_width)) / dest_width;
			float y_ratio = ((float)(src_height)) / dest_height;
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
						gray = (a)*(1 - x_diff)*(1 - y_diff) + (b)*(x_diff)*(1 - y_diff) + (c)*(y_diff)*(1 - x_diff) + (d)*(x_diff*y_diff);
						dst_data[offset + j] = gray;
					}
				}
#ifdef _OPENMP
			}
#endif
		}

	};
}

#endif // _IMAGEPYRAMID_HPP_