#include "upsample.hpp"
#include <memory>
#include <algorithm>
#include <iostream>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		upsample::upsample(int scale, int device):scale_(scale), device_(device) {}

		upsample::~upsample() {}

		void upsample::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = (bottom)->data_shape()[0];
			const float* bottom_data = (bottom)->cpu_data();
			height_ = (bottom)->height();
			width_ = (bottom)->width();
			channel_ = (bottom)->channels();
			order_ = (bottom)->order();

			int top_height = scale_ * height_;
			int top_width = scale_ * width_;

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{num, channel_, top_height, top_width}, device_, order_));
				float *top_data = top->mutable_cpu_data();

				for (int n = 0; n < num; n++) {
					for (int c = 0; c < channel_; c++) {
						for (int h = 0; h < top_height; h++) {
							for (int w = 0; w < top_width; w++) {
								int nw = w / scale_;
								int nh = h / scale_;
								int out_idx = (((n * channel_ + c) * top_height) + h) * top_width + w;
								int in_idx = (((n * channel_ + c) * (top_height / scale_)) + nh) * (top_width / scale_) + nw;
								top_data[out_idx] = bottom_data[in_idx];
							}
						}
					}
				}
			}
			else if (order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{num, top_height, top_width, channel_}, device_, order_));
				float *top_data = top->mutable_cpu_data();

				for (int n = 0; n < num; n++) {
					for (int c = 0; c < channel_; c++) {
						for (int h = 0; h < top_height; h++) {
							for (int w = 0; w < top_width; w++) {
								int nw = w / scale_;
								int nh = h / scale_;
								int out_idx = (((n * top_height + h) * top_width) + w) * channel_ + c;
								int in_idx = (((n * (top_height / scale_) + nh) * (top_width / scale_)) + nw) * channel_ + c;
								top_data[out_idx] = bottom_data[in_idx];
							}
						}
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}

