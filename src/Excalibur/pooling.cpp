#include "pooling.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
namespace glasssix
{
	namespace excalibur
	{
		pooling::pooling(int kernel, int stride, int pad, int type, int device)
		{
			kernel_ = kernel;
			stride_ = stride;
			pad_ = pad;
			type_ = (pooling_type)type;
			device_ = device;

#ifdef USE_CUDA
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
			if (type_ == MAX)
			{
				mode_ = CUDNN_POOLING_MAX;
			}
			else if (type_ == AVE)
			{
				mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
			}
			else
			{
				LOG(FATAL) << "Unknown pooling type.";
			}
			CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
			CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc_, mode_,
				CUDNN_PROPAGATE_NAN, kernel_, kernel_, pad_, pad_, stride_, stride_));
#endif
#endif // USE_CUDA


		}

		pooling::~pooling()
		{

#ifdef USE_CUDA
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
			cudnnDestroyTensorDescriptor(bottom_desc_);
			cudnnDestroyTensorDescriptor(top_desc_);
			cudnnDestroyPoolingDescriptor(pooling_desc_);
#endif
#endif // USE_CUDA


		}

		void pooling::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();

			if (order_ == NCHW)
			{
				int num = bottom->data_shape()[0];
				channels_ = bottom->data_shape()[1];
				height_ = bottom->data_shape()[2];
				width_ = bottom->data_shape()[3];
				pooled_height_ = static_cast<int>(ceil(static_cast<float>(
					height_ + 2 * pad_ - kernel_) / stride_)) + 1;
				pooled_width_ = static_cast<int>(ceil(static_cast<float>(
					width_ + 2 * pad_ - kernel_) / stride_)) + 1;
				top.reset(new tensor<float>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, device_, order_));
				//
				const float* bottom_data = bottom->cpu_data();
				float* top_data = (top)->mutable_cpu_data();
				const int top_count = (top)->count(0, 4);

				const int bottom_offset = bottom->offset(0, 1);
				const int top_offset = (top)->offset(0, 1);
				switch (type_)
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
			else if (order_ == NHWC)
			{
				int num = bottom->data_shape()[0];
				height_ = bottom->data_shape()[1];
				width_ = bottom->data_shape()[2];
				channels_ = bottom->data_shape()[3];

				pooled_height_ = static_cast<int>(ceil(static_cast<float>(
					height_ + 2 * pad_ - kernel_) / stride_)) + 1;
				pooled_width_ = static_cast<int>(ceil(static_cast<float>(
					width_ + 2 * pad_ - kernel_) / stride_)) + 1;
				top.reset(new tensor<float>(std::vector<int>{num, pooled_height_, pooled_width_, channels_}, device_, order_));
				//
				const float* bottom_data = bottom->cpu_data();
				float* top_data = (top)->mutable_cpu_data();
				const int top_count = (top)->count(0, 4);

				switch (type_)
				{
				case MAX:
					for (int n = 0; n < num; ++n) {
						int top_index0 = n * pooled_width_ * pooled_height_ * channels_;
						int bottom_index0 = n * width_ * height_ * channels_;
						for (int ph = 0; ph < pooled_height_; ++ph) {
							int top_index1 = top_index0 + ph * pooled_width_ * channels_;
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int top_index2 = top_index1 + pw * channels_;
								int hstart = ph * stride_ - pad_;
								int wstart = pw * stride_ - pad_;
								int hend = std::min(hstart + kernel_, height_);
								int wend = std::min(wstart + kernel_, width_);
								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);

								for (int c = 0; c < channels_; ++c)
								{
									float top_val = -FLT_MAX;
									for (int h = hstart; h < hend; ++h) {
										int bottom_index1 = bottom_index0 + h * width_ * channels_;
										for (int w = wstart; w < wend; ++w)
										{
											int bottom_index2 = bottom_index1 + w * channels_;
											top_val = std::max(top_val, bottom_data[bottom_index2 + c]);
										}
									}
									top_data[top_index2 + c] = top_val;
								}
							}
						}
					}
					break;
				case AVE:
					memset(top_data, 0, top_count * sizeof(float));
					// The main loop
					for (int n = 0; n < num; ++n) {
						int top_index0 = n * pooled_width_ * pooled_height_ * channels_;
						int bottom_index0 = n * width_ * height_ * channels_;
						for (int ph = 0; ph < pooled_height_; ++ph) {
							int top_index1 = top_index0 + ph * pooled_width_ * channels_;
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int top_index2 = top_index1 + pw * channels_;
								int hstart = ph * stride_ - pad_;
								int wstart = pw * stride_ - pad_;
								int hend = std::min(hstart + kernel_, height_ + pad_);
								int wend = std::min(wstart + kernel_, width_ + pad_);
								int pool_size = (hend - hstart) * (wend - wstart);
								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);
								hend = std::min(hend, height_);
								wend = std::min(wend, width_);

								for (int c = 0; c < channels_; ++c) {
									for (int h = hstart; h < hend; ++h) {
										int bottom_index1 = bottom_index0 + h * width_ * channels_;
										for (int w = wstart; w < wend; ++w) {
											int bottom_index2 = bottom_index1 + w * channels_;
											top_data[top_index2 + c] += bottom_data[bottom_index2 + c];
										}
									}
									top_data[top_index2 + c] /= pool_size;
								}
							}
						}
					}
					break;
				default:
					LOG(FATAL) << "Unknown pooling method.";
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}

