#include "pooling.hpp"
#include <algorithm>

namespace excalibur
{
	pooling::pooling(int kernel, int stride, int pad, int type, int device)
	{
		kernel_ = kernel;
		stride_ = stride;
		pad_ = pad;
		type_ = (pooling_type)type;
		device_ = device;
	}

	pooling::~pooling()
	{
	}

	void pooling::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		int num = bottom->data_shape()[0];
		channels_ = bottom->data_shape()[1];
		height_ = bottom->data_shape()[2];
		width_ = bottom->data_shape()[3];
		pooled_height_ = static_cast<int>(ceil(static_cast<float>(
			height_ + 2 * pad_ - kernel_) / stride_)) + 1;
		pooled_width_ = static_cast<int>(ceil(static_cast<float>(
			width_ + 2 * pad_ - kernel_) / stride_)) + 1;
		top.reset(new tensor<float>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, device_));
		//
		const float* bottom_data = bottom->cpu_data();
		float* top_data = (top)->mutable_cpu_data();
		const int top_count = (top)->count(0, 4);
		const int bottom_offset = bottom->offset(0, 1);
		const int top_offset = (top)->offset(0, 1);
		switch(type_)
		{
		case MAX:
			for (int n = 0; n < num; ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < pooled_height_; ++ph) {
						for (int pw = 0; pw < pooled_width_; ++pw) {
							int hstart = ph * stride_ - pad_;
							int wstart = pw * stride_ - pad_;
							int hend = std::min(hstart + kernel_, height_);
							int wend = std::min(wstart + kernel_, width_);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							float top_val = -FLT_MAX;
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = h * width_ + w;
									top_val = std::max(top_val, bottom_data[index]);
								}
							}
							const int pool_index = ph * pooled_width_ + pw;
							top_data[pool_index] = top_val;
						}
					}
					// compute offset
					bottom_data += bottom_offset;
					top_data += top_offset;
				}
			}
			break;
		case AVE:
			memset(top_data, 0, top_count * sizeof(float));
			// The main loop
			for (int n = 0; n < num; ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < pooled_height_; ++ph) {
						for (int pw = 0; pw < pooled_width_; ++pw) {
							int hstart = ph * stride_ - pad_;
							int wstart = pw * stride_ - pad_;
							int hend = std::min(hstart + kernel_, height_ + pad_);
							int wend = std::min(wstart + kernel_, width_ + pad_);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, height_);
							wend = std::min(wend, width_);
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									top_data[ph * pooled_width_ + pw] += bottom_data[h * width_ + w];
								}
							}
							top_data[ph * pooled_width_ + pw] /= pool_size;
						}
					}
					// compute offset
					bottom_data += bottom_offset;
					top_data += top_offset;
				}
			}
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

}
