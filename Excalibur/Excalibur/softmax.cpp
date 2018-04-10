#include "softmax.hpp"
#include <algorithm>

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
	}

	softmax::~softmax()
	{
		
	}

	void softmax::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		outer_num_ = bottom->num();
		if (bottom->data_shape().size()<=2)
		{
			inner_num_ = 1;
			scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1}, device_));
		}
		else
		{
			inner_num_ = bottom->height()*bottom->width();
			scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1, bottom->height(), bottom->width()}, device_));
		}
		
		top.reset(new tensor<float>(bottom->data_shape()/*std::vector<int>{outer_num_, bottom->channels()}*/, -1));
		//
		const float* bottom_data = bottom->cpu_data();
		float* top_data = (top)->mutable_cpu_data();
		float* scale_data = scale_->mutable_cpu_data();
		int channels = bottom->channels();
		int dim = bottom->count(0, bottom->data_shape().size()) / outer_num_;
		memcpy(top_data, bottom_data, bottom->count(0, bottom->data_shape().size()) * sizeof(float));
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

}
