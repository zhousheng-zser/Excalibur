#include "conv_native_cpu.hpp"
#include <iostream>
#include "../../include/Julius/simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		void conv_native_cpu::forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col)
		{
			const signed char* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_int8_data);
				col_buff = col_buffer_int8_data;
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::cpu_fgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_fgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
							output_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
							weights + kernelSize_ * kernelSize_ * g, col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::cpu_fgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_fgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_ / group_,
							kernelSize_ * kernelSize_, 1.0f,
							col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, weights + kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}

					std::shared_ptr<tensor<int>> temp;
					temp.reset(new tensor<int>(std::vector<int>{output_spatial_dim_ * output_Channel_}));
					int* temp_data = temp->mutable_cpu_data();
					for (int ch = 0; ch < output_Channel_; ++ch)
					{
						int channel_offset = ch * output_dim_h_ * output_dim_w_;
						for (int row = 0; row < output_dim_h_; ++row)
						{
							int row_offset = row * output_dim_w_;
							for (int col = 0; col < output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * output_Channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, output_spatial_dim_ * output_Channel_ * sizeof(int));
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_data);
				col_buff = col_buffer_data;
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
							output_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
							weights + kernelSize_ * kernelSize_ * g, col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_ / group_,
							kernelSize_ * kernelSize_, 1.0f,
							col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, weights + kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}

					std::shared_ptr<tensor<float>> temp;
					temp.reset(new tensor<float>(std::vector<int>{output_spatial_dim_ * output_Channel_}));
					float* temp_data = temp->mutable_cpu_data();

					for (int ch = 0; ch < output_Channel_; ++ch)
					{
						int channel_offset = ch * output_dim_h_ * output_dim_w_;
						for (int row = 0; row < output_dim_h_; ++row)
						{
							int row_offset = row * output_dim_w_;
							for (int col = 0; col < output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * output_Channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, output_spatial_dim_ * output_Channel_ * sizeof(float));
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_cpu::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_data,
					1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_data, bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			num_ = bottom->data_shape()[0];
			order_ = bottom->order();
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();
			bottom_dim_ = bottom->count(1, 4);
			bottom_data = bottom->cpu_data();

			if (int8_quantization_)
			{
				bottom_int8_.reset(new tensor<signed char>(std::vector<int>{num_ * bottom_dim_}));
				bottom_int8_data = bottom_int8_->mutable_cpu_data();

#if SIMD_TYPE >= SIMDTYPE_SSE
				int circle_num = num_ * bottom_dim_ / mm_align_size;				
				mm_type scale = mm_set1_ps(scales_data[0]);
				int index = 0;

				for (; index < circle_num; index++)
				{
					int index_offset = index * mm_align_size;
					mm_type data = mm_load_ps(bottom_data + index_offset);
					mm_type res_mul = mm_mul_ps(data, scale);
					mm_type res_round = mm_round_ps(res_mul, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
					mm_store_ps(bottom_round_data_, res_round);
					for (int i = 0; i < mm_align_size; i++)
					{
						if (bottom_round_data_[i] > 127)
						{
							bottom_int8_data[index_offset + i] = (signed char)(127);
						}
						else if (bottom_round_data_[i] < -128)
						{
							bottom_int8_data[index_offset + i] = (signed char)(-128);
						}
						else
						{
							bottom_int8_data[index_offset + i] = (signed char)(bottom_round_data_[i]);
						}
					}
				}

				for (index = mm_align_size * index; index < num_ * bottom_dim_; index++)
				{
					bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
				}
#else
#pragma omp for
				for (int index = 0; index < num_ * bottom_dim_; index++)
				{
					bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
				}
#endif
			}

			if (order_ == NCHW)
			{
				output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				top_data = top->mutable_cpu_data();
				top_dim_ = top->count(1, 4);

				if (int8_quantization_)
				{
					top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
					top_int32_data = top_int32_->mutable_cpu_data();
					col_buffer_int8_.reset(new tensor<signed char>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
					col_buffer_int8_data = col_buffer_int8_->mutable_cpu_data();
				}
				else
				{
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
					col_buffer_data = col_buffer_->mutable_cpu_data();
				}

				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
				output_spatial_dim_ = output_dim_w_ * output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);

				for (int n = 0; n < num_; n++)
				{
					if (int8_quantization_)
					{
						forward_gemm(bottom_int8_data + n * bottom_dim_, weights_int8_data, top_int32_data + n * top_dim_);

						int offset = top_dim_ / group_;

#if SIMD_TYPE >= SIMDTYPE_SSE
						int circle_num = offset / mm_align_size;
						for (int j = 0; j < group_; j++)
						{
							float total_scale = scales_data[0] * scales_data[1 + j];
							mm_type scale = mm_set1_ps(1.0f / total_scale);
							int index = 0;
							for (; index < circle_num; index++)
							{
								int index_offset = index * mm_align_size;
								mm_typei temp1 = mm_load_si((mm_typei*)(top_int32_data + n * top_dim_ + j * offset + index_offset));
								mm_type temp2 = mm_cvtepi32_ps(temp1);
								mm_type res = mm_mul_ps(temp2, scale);
								mm_store_ps(top_data + n * top_dim_ + j * offset + index_offset, res);
							}

							for (index = index * mm_align_size; index < (j + 1) * offset; index++)
							{
								top_data[index + n * top_dim_ + j * offset] = top_int32_data[index + n * top_dim_ + j * offset] / total_scale;
							}
						}
#else
						for (int index = 0; index < offset; index++)
						{
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j];
								top_data[index + n * top_dim_ + j * offset] = top_int32_data[index + n * top_dim_ + j * offset] / total_scale;
							}
						}
#endif
					}
					else
					{
						forward_gemm(bottom_data + n * bottom_dim_, weights_data, top_data + n * top_dim_);
					}

					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim_, bias_data);
					}
				}
			}
			else if (order_ == NHWC)
			{
				output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				top_data = top->mutable_cpu_data();
				top_dim_ = top->count(1, 4);

				if (int8_quantization_)
				{
					top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
					top_int32_data = top_int32_->mutable_cpu_data();
					col_buffer_int8_.reset(new tensor<signed char>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
					col_buffer_int8_data = col_buffer_int8_->mutable_cpu_data();
				}
				else
				{
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
					col_buffer_data = col_buffer_->mutable_cpu_data();
				}

				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
				output_spatial_dim_ = output_dim_w_ * output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);

				for (int n = 0; n < num_; n++)
				{
					if (int8_quantization_)
					{
						forward_gemm(bottom_int8_data + n * bottom_dim_, weights_int8_data, top_int32_data + n * top_dim_);

						int offset = top_dim_ / group_;

#if SIMD_TYPE >= SIMDTYPE_SSE
						int circle_num = offset / mm_align_size;
						for (int j = 0; j < group_; j++)
						{
							float total_scale = scales_data[0] * scales_data[1 + j];
							mm_type scale = mm_set1_ps(1.0f / total_scale);
							int index = 0;
							for (; index < circle_num; index++)
							{
								int index_offset = index * mm_align_size;
								mm_typei temp1 = mm_load_si((mm_typei*)(top_int32_data + n * top_dim_ + j * offset + index_offset));
								mm_type temp2 = mm_cvtepi32_ps(temp1);
								mm_type res = mm_mul_ps(temp2, scale);
								mm_store_ps(top_data + n * top_dim_ + j * offset + index_offset, res);
							}

							for (index = index * mm_align_size; index < (j + 1) * offset; index++)
							{
								top_data[index + n * top_dim_ + j * offset] = top_int32_data[index + n * top_dim_ + j * offset] / total_scale;
							}
						}
#else
						for (int index = 0; index < top_dim_ / group_; index++)
						{
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j];
								top_data[index + n * top_dim_ + j * top_dim_ / group_] = top_int32_data[index + n * top_dim_ + j * top_dim_ / group_] / total_scale;
							}
						}
#endif
					}
					else
					{
						forward_gemm(bottom_data + n * bottom_dim_, weights_data, top_data + n * top_dim_);
					}

					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim_, bias_data);
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