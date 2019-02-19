#include "softmax.hpp"
#include <algorithm>

namespace glasssix
{
	namespace excalibur
	{
		softmax::softmax(int input_channel, int device)
		{
			device_ = device;
			softmax_axis_ = 1;
			sum_multiplier_.reset(new tensor<float>(std::vector<int>{input_channel}, device_));
			float* multiplier_data = sum_multiplier_->mutable_cpu_data();
			for (int i = 0; i < input_channel; i++)
			{
				multiplier_data[i] = 1.0f;
			}
#ifdef USE_CUDNN
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
#endif
		}

		softmax::~softmax()
		{
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
			cudnnDestroyTensorDescriptor(bottom_desc_);
			cudnnDestroyTensorDescriptor(top_desc_);
#endif
		}

		void softmax::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			outer_num_ = bottom->num();
			int height = bottom->height();
			int width = bottom->width();
			order_ = bottom->order();
			if (bottom->data_shape().size() <= 2)
			{
				inner_num_ = 1;
				scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1}, device_));
			}
			else
			{
				inner_num_ = height * width;
				scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1, bottom->height(), bottom->width()}, device_));//单通道，NHWC和NCHW无差别
			}

			top.reset(new tensor<float>(bottom->data_shape()/*std::vector<int>{outer_num_, bottom->channels()}*/, -1));
			//
			const float* bottom_data = bottom->cpu_data();
			float* top_data = (top)->mutable_cpu_data();
			float* scale_data = scale_->mutable_cpu_data();
			int channels = bottom->channels();
			int dim = bottom->count(0, bottom->data_shape().size()) / outer_num_;
			memcpy(top_data, bottom_data, bottom->count(0, bottom->data_shape().size()) * sizeof(float));

			if (order_ == NCHW)
			{
				for (int i = 0; i < outer_num_; ++i)
				{
					memcpy(scale_data, bottom_data + i * dim, inner_num_ * sizeof(float));
					for (int j = 0; j < channels; j++) {
						for (int k = 0; k < inner_num_; k++) {
							scale_data[k] = std::max(scale_data[k],
								bottom_data[i * dim + j * inner_num_ + k]);
						}
					}
					cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, channels, inner_num_,
						1, -1.0f, sum_multiplier_->cpu_data(), 1, scale_data, inner_num_, 1.0f, top_data, inner_num_);
					for (int k = 0; k < dim; k++)
					{
						top_data[k] = exp(top_data[k]);
					}
					cblas_sgemv(CblasRowMajor, CblasTrans, channels, inner_num_, 1.0f,
						top_data, inner_num_, sum_multiplier_->cpu_data(), 1, 0.0f, scale_data, 1);
					// division
					for (int j = 0; j < channels; j++) {
						for (int k = 0; k < inner_num_; k++)
						{
							top_data[k] = top_data[k] / scale_data[k];
						}
						top_data += inner_num_;
					}
				}
			}
			else if (order_ == NHWC)
			{
				for (int i = 0; i < outer_num_; ++i)
				{
					memcpy(scale_data, bottom_data + i * dim, inner_num_ * sizeof(float));

					//caculate max channel value, saved in scale_data
					for (int row = 0; row < height; row++)
					{
						int bottom_index1 = i * dim + row * width * channels;
						int scale_index1 = row * width;
						for (int col = 0; col < width; col++)
						{
							int bottom_index2 = bottom_index1 + col * channels;
							int scale_index2 = scale_index1 + col;
							for (int ch = 0; ch < channels; ch++)
							{
								scale_data[scale_index2] = std::max(scale_data[scale_index2],
									bottom_data[bottom_index2 + ch]);
							}
						}
					}

					//top_data subtract max value for each channel
					for (int row = 0; row < height; row++)
					{
						int top_index1 = i * dim + row * width * channels;
						int scale_index1 = row * width;
						for (int col = 0; col < width; col++)
						{
							int top_index2 = top_index1 + col * channels;
							int scale_index2 = scale_index1 + col;
							for (int ch = 0; ch < channels; ch++)
							{
								top_data[top_index2 + ch] -= scale_data[scale_index2];
							}
						}
					}

					//calculate exp for each top_data
					for (int k = 0; k < dim; k++)
					{
						top_data[k] = exp(top_data[k]);
					}

					//caculate channel sum, saved in scale_data
					for (int row = 0; row < height; row++)
					{
						int top_index1 = i * dim + row * width * channels;
						int scale_index1 = row * width;
						for (int col = 0; col < width; col++)
						{
							int top_index2 = top_index1 + col * channels;
							int scale_index2 = scale_index1 + col;
							float sum = 0.0f;
							for (int ch = 0; ch < channels; ch++)
							{
								sum += top_data[top_index2 + ch];
							}
							scale_data[scale_index2] = sum;
						}
					}

					// divide channel sum for each channel of top_data
					for (int row = 0; row < height; row++)
					{
						int top_index1 = i * dim + row * width * channels;
						int scale_index1 = row * width;
						for (int col = 0; col < width; col++)
						{
							int top_index2 = top_index1 + col * channels;
							int scale_index2 = scale_index1 + col;
							for (int ch = 0; ch < channels; ch++)
							{
								top_data[top_index2 + ch] = top_data[top_index2 + ch] / scale_data[scale_index2];
							}
						}
					}

					//offset
					top_data += dim;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}

