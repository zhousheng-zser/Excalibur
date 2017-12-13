#include "ImagePyramid.hpp"
#include <iostream>

namespace excalibur
{
	const std::shared_ptr<ImageTensor<unsigned char>> ImagePyramid::GetNextScaleImage(float* scale_factor)
	{
		if (scale_factor_ >= min_scale_)
		{
			if (scale_factor != nullptr)
				*scale_factor = scale_factor_;
			width_scaled_ = static_cast<int>(width1x_ * scale_factor_);
			height_scaled_ = static_cast<int>(height1x_ * scale_factor_);
			img_scaled_.reset(new ImageTensor<unsigned char>(width_scaled_, height_scaled_, 1, device_));
			if (device_>=0) // gpu code
			{
				NOT_IMPLEMENTED;
				//ResizeImageGPU(img_ori_, img_scaled_);
			}
			else // cpu code
			{
				ResizeImageCPU(img_ori_, img_scaled_);
			}
			scale_factor_ *= scale_step_;
			return img_scaled_;
		}
		else
		{
			return nullptr;
		}
	}

	// init ori image data
	void ImagePyramid::SetImage1x(const unsigned char* img_data, int width, int height)
	{
		width1x_ = width;
		height1x_ = height;
		img_ori_.reset(new ImageTensor<unsigned char>(width, height, 1, device_));
		unsigned char* temp;
		if (device_>=0)
		{
#ifdef USE_CUDA
			temp = img_ori_->mutable_gpu_data();
#else
			NO_GPU;
#endif
		}
		else
		{
			temp = img_ori_->mutable_cpu_data();
		}
		math_functions::excalibur_copy(width * height, img_data, temp, device_);
		scale_factor_ = max_scale_;
	}

}