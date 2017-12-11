#include "ImagePyramid.hpp"

namespace excalibur
{
	const std::shared_ptr<ImageTensor<unsigned char>> ImagePyramid::GetNextScaleImage(float* scale_factor)
	{
		if (scale_factor_ >= min_scale_)
		{
			if (scale_factor != nullptr)
				*scale_factor = scale_factor_;
			width_scaled_ = static_cast<int32_t>(width1x_ * scale_factor_);
			height_scaled_ = static_cast<int32_t>(height1x_ * scale_factor_);
			std::shared_ptr<ImageTensor<unsigned char>> src_img = std::make_shared<ImageTensor<unsigned char>>(width1x_, height1x_, 1, device_);
			std::shared_ptr<ImageTensor<unsigned char>> dest_img = std::make_shared<ImageTensor<unsigned char>>(width_scaled_, height_scaled_, 1, device_);
			if (device_>=0) // gpu code
			{
				/*gpu_buf_img_ = src_img->mutable_gpu_data();
				gpu_buf_img_scaled_ = dest_img->mutable_gpu_data();
				ResizeImageGPU(src_img, dest_img);
				img_scaled_.reset(new ImageTensor<unsigned char>(width_scaled_, height_scaled_, 1, device_));
				img_scaled_->set_gpu_data(buf_img_scaled_);*/
			}
			else // cpu code
			{
				cpu_buf_img_ = src_img->mutable_cpu_data();
				cpu_buf_img_scaled_ = dest_img->mutable_cpu_data();
				ResizeImageCPU(src_img, dest_img);
				img_scaled_.reset(new ImageTensor<unsigned char>(width_scaled_, height_scaled_, 1, device_));
				img_scaled_->set_cpu_data(cpu_buf_img_scaled_);
			}
			return img_scaled_;
		}
		else
		{
			return nullptr;
		}
	}

	void ImagePyramid::SetImage1x(const uint8_t* img_data, int32_t width, int32_t height)
	{
		if (width > buf_img_width_ || height > buf_img_height_) {
			delete[] cpu_buf_img_;

			buf_img_width_ = width;
			buf_img_height_ = height;
			cpu_buf_img_ = new uint8_t[width * height];
		}

		width1x_ = width;
		height1x_ = height;

		std::memcpy(cpu_buf_img_, img_data, width * height * sizeof(uint8_t));
		scale_factor_ = max_scale_;
		UpdateBufScaled();
	}

	void ImagePyramid::UpdateBufScaled()
	{
		if (width1x_ == 0 || height1x_ == 0)
			return;

		int32_t max_width = static_cast<int32_t>(width1x_ * max_scale_ + 0.5);
		int32_t max_height = static_cast<int32_t>(height1x_ * max_scale_ + 0.5);

		if (max_width > buf_scaled_width_ || max_height > buf_scaled_height_) {
			delete[] cpu_buf_img_scaled_;

			buf_scaled_width_ = max_width;
			buf_scaled_height_ = max_height;
			cpu_buf_img_scaled_ = new uint8_t[max_width * max_height];

			img_scaled_ = nullptr;
			/*img_scaled_.data = nullptr;
			img_scaled_.width = 0;
			img_scaled_.height = 0;*/
		}
	}

}