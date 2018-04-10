#include "flip.hpp"

namespace excalibur
{
	flip::flip(bool flip_height, bool flip_width, int device)
	{
		flip_height_ = flip_height;
		flip_width_ = flip_width;
		device_ = device;
	}


	flip::~flip()
	{
	}

	void flip::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		top.reset(new tensor<float>(bottom->data_shape(), device_));
		const float* bottom_data = bottom->cpu_data();
		float* top_data = top->mutable_cpu_data();
		int num = bottom->num();
		int channels = bottom->channels();
		int width = bottom->width();
		int height = bottom->height();
		for (int n = 0; n < num; n++) {
			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						top_data[(((n * channels + c) * height + h) * width) + w] =
							bottom_data[(((n * channels + c) * height + (flip_height_ ? (height - 1 - h) : h)) * width) + (flip_width_ ? (width - 1 - w) : w)];
					}
				}
			}
		}
	}

}
