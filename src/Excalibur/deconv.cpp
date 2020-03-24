#include "deconv.hpp"
#include <iostream>
#include "../../include/Julius/simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		void deconv::forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col) {}

		void deconv::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			float* col_buff = col_buffer_->mutable_cpu_data();

			if (order_ == NCHW)
			{
				for (int g = 0; g < group_; g++)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasNoTrans, output_Channel_ * kernel_length_ / group_,
						input_spatial_dim_, input_Channel_ / group_, 1.0f,
						weights + weight_offset_ * g, input + g * input_Channel_ * input_spatial_dim_ / group_, 0.0f, col_buff + col_offset_ * g);
				}

				conv_col2im_cpu(col_buff, output);
			}
			else if (order_ == NHWC)
			{
				NOT_IMPLEMENTED;
				//if (group_ == 1)
				//{
				//	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, input_spatial_dim_, output_Channel_ * kernel_length_,
				//		input_Channel_, 1.0f,
				//		input, weights, 0.0f, col_buff);
				//}
				//else if (group_ > 1)
				//{
				//	float* temp = (float*)malloc(input_spatial_dim_ * kernel_length_ * sizeof(float));
				//	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, input_spatial_dim_, output_Channel_ * kernel_length_ / group_,
				//		input_Channel_, 1.0f,
				//		input, weights, 0.0f, temp);

				//	for (int i = 0; i < input_spatial_dim_; i++)
				//	{
				//		for (int j = 0; j < output_Channel_; j++)
				//		{
				//			memcpy(col_buff + i * output_Channel_ * kernel_length_ + j * kernel_length_, 
				//				temp + i * kernel_length_, kernel_length_ * sizeof(float));
				//		}
				//	}

				//	free(temp);
				//}
				//else
				//{
				//	LOG(FATAL) << "illegal group!";
				//}

				//conv_col2im_cpu(col_buff, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void deconv::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
					1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void deconv::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();
			num_ = bottom->data_shape()[0];
			input_shape_.clear();
			input_shape_ = bottom->data_shape();
			bottom_dim_ = bottom->count(1, 4);
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();

			if (order_ == NCHW)
			{
				input_Channel_ = bottom->data_shape()[1];
				input_dim_h_ = bottom->data_shape()[2];
				input_dim_w_ = bottom->data_shape()[3];
				input_spatial_dim_ = input_dim_h_ * input_dim_w_;
				output_dim_h_ =	(input_dim_h_ - 1) * stride_ + kernelSize_ - 2 * pad_;
				output_dim_w_ = (input_dim_w_ - 1) * stride_ + kernelSize_ - 2 * pad_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				output_shape_ = top->data_shape();
				float* top_data = (top)->mutable_cpu_data();

				col_buffer_.reset(new tensor<float>(std::vector<int>{output_Channel_ * kernel_length_, input_dim_h_, input_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
				col_offset_ = kernel_length_ * input_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				top_dim_ = top->count(1, 4);
				for (int n = 0; n < num_; n++)
				{
					forward_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim_);
					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim_, bias);
					}
				}
			}
			else if (order_ == NHWC)
			{
				input_Channel_ = bottom->data_shape()[3];
				input_dim_h_ = bottom->data_shape()[1];
				input_dim_w_ = bottom->data_shape()[2];
				input_spatial_dim_ = input_dim_h_ * input_dim_w_;
				output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				output_shape_ = top->data_shape();
				float* top_data = (top)->mutable_cpu_data();

				col_buffer_.reset(new tensor<float>(std::vector<int>{output_Channel_ * kernel_length_, input_dim_h_, input_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int top_dim = top->count(1, 4);
				for (int n = 0; n < num_; n++)
				{
					forward_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim, bias);
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