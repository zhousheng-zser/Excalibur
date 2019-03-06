#ifdef USE_CUDA
#include "conv_native_gpu.hpp"
#include <filesystem>
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		void conv_native_gpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			if (group_>1)
			{
				conv_depthwise_native_gpu depthwise_conv(input_Channel_, output_Channel_, kernelSize_, group_, stride_, pad_, bias_term_, device_);
				depthwise_conv.Forward(bottom, top);
				return;
			}
			const int num = bottom->num();
			const float* bottom_data = bottom->gpu_data();
			const float* weights = weights_->gpu_data();
			const float* bias = bias_->gpu_data();
			order_ = bottom->order();
			//
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();

			if (order_ == NCHW)
			{
				int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				//

				float* top_data = (top)->mutable_gpu_data();
				if (isfirst)
				{
					last_height = bottom->height();
					last_width = bottom->width();
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
					gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
					math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					isfirst = false;
				}
				else
				{
					if (last_height != bottom->height() || last_width != bottom->width())
					{
						last_height = bottom->height();
						last_width = bottom->width();
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
						gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * output_spatial_dim_;
						output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
				{
					forward_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim, bias);
					}
				}
			}
			else if (order_ == NHWC)
			{
				int output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				int output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				//

				float* top_data = (top)->mutable_gpu_data();
				if (isfirst)
				{
					last_height = bottom->height();
					last_width = bottom->width();
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
					gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
					math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					isfirst = false;
				}
				else
				{
					if (last_height != bottom->height() || last_width != bottom->width())
					{
						last_height = bottom->height();
						last_width = bottom->width();
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
						gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * output_spatial_dim_;
						output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
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

		void conv_native_gpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;
			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
				col_buff = col_buffer_->gpu_data();
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f, weights, col_buff, 0.0f, output);
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_, kernel_dim_, 1.0f, col_buff, weights, 0.0f, output);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_gpu::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

	}
}

#endif