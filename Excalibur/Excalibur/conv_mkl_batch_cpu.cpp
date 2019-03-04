#ifdef USE_MKL
#include "conv_mkl_batch_cpu.hpp"

namespace glasssix
{
	namespace excalibur
	{
		void conv_mkl_batch_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
				col_buff = col_buffer_->cpu_data();
			}

			if (order_ == NCHW)
			{
				for (int g = 0; g < group_; ++g)
				{
					math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
						output_spatial_dim_, kernel_dim_ / group_, 1.0f,
						weights + g * kernel_dim_ / group_, 0,
						col_buff + g * output_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_, 0.0f,
						output + output_spatial_dim_ * g, top_dim_, num_);
				}
			}
			else if (order_ == NHWC)
			{
				for (int g = 0; g < group_; ++g)
				{
					math_functions::cpu_batch_sgemm(CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_ / group_, kernel_dim_ / group_, 1.0f,
						col_buff + g * output_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_,
						weights + g * kernel_dim_ / group_, 0, 0.0f,
						output + output_spatial_dim_ * g, top_dim_, num_);
				}

				if (group_ > 1)
				{
					float* temp_data = (float*)malloc(output_spatial_dim_ * output_Channel_ * sizeof(float));
					for (int n = 0; n < num_; n++)
					{
						int n_offset = n * output_spatial_dim_ * output_Channel_;
						for (int ch = 0; ch < output_Channel_; ++ch)
						{
							int channel_offset = ch * output_dim_h_ * output_dim_w_;
							for (int row = 0; row < output_dim_h_; ++row)
							{
								int row_offset = row * output_dim_w_;
								for (int col = 0; col < output_dim_w_; ++col)
								{
									temp_data[(row_offset + col) * output_Channel_ + ch] = output[n_offset + channel_offset + row_offset + col];
								}
							}
						}
						memcpy(output + n_offset, temp_data, output_spatial_dim_ * output_Channel_ * sizeof(float));
					}

					delete temp_data;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_mkl_batch_cpu::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, 0,
					bias_multiplier_->cpu_data(), output_spatial_dim_, 1.0f, output, output_spatial_dim_ * output_Channel_, num_);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), output_spatial_dim_, bias, 0,
					1.0f, output, output_spatial_dim_ * output_Channel_, num_);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_mkl_batch_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			num_ = bottom->data_shape()[0];
			bottom_dim_ = bottom->count(1, 4);
			order_ = bottom->order();
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();

			if (order_ == NCHW)
			{
				output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				float* top_data = (top)->mutable_cpu_data();
				top_dim_ = top->count(1, 4);

				col_buffer_.reset(new tensor<float>(std::vector<int>{num_*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{num_*output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(num_*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());

				forward_gemm(bottom_data, weights, top_data);
				if (bias_term_)
				{
					forward_bias(top_data, bias);
				}
			}
			else if (order_ == NHWC)
			{
				output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				float* top_data = (top)->mutable_cpu_data();
				top_dim_ = top->count(1, 4);

				col_buffer_.reset(new tensor<float>(std::vector<int>{num_*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{num_*output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(num_*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				
				forward_gemm(bottom_data, weights, top_data);
				if (bias_term_)
				{
					forward_bias(top_data, bias);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}
#endif //!USE_MKL