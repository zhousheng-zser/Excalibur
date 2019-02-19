#include "convolution.hpp"
//#include "cblas.h"
#include <iostream>
#include <filesystem>

namespace glasssix
{
	namespace excalibur
	{
		convolution::convolution(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, bool bias_term, int device)
		{
			input_Channel_ = input_Channel;
			output_Channel_ = output_Channel;
			kernelSize_ = kernelSize;
			stride_ = stride;
			pad_ = pad;
			bias_term_ = bias_term;
			device_ = device;
			weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_}, device_));
			bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
			setup_internal_params();
		}

		convolution::convolution(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device)
		{
			CHECK_EQ(output_Channel % group, 0);
			CHECK_EQ(input_Channel % group, 0);
			input_Channel_ = input_Channel;
			output_Channel_ = output_Channel;
			kernelSize_ = kernelSize;
			stride_ = stride;
			pad_ = pad;
			bias_term_ = bias_term;
			device_ = device;
			weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_ / group}, device_));
			bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
			setup_internal_params(group);
		}

		convolution::~convolution()
		{
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
			CUDNN_CHECK(cudnnDestroyTensorDescriptor(xdesc));
			CUDNN_CHECK(cudnnDestroyTensorDescriptor(ydesc));
			CUDNN_CHECK(cudnnDestroyFilterDescriptor(wdesc));
			CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
			if (bias_term_)
			{
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(bdesc));
			}
			if (extra != nullptr)
			{
				cudaFree(extra);
			}
#endif
		}

		void convolution::set_bias(float* bias)
		{
			if (bias_term_)
			{
				bias_->set_cpu_data(bias);
			}
		}

		void convolution::set_weights(float* weights)
		{
			weights_->set_cpu_data(weights);
		}

		void convolution::conv_im2col_cpu(const float* data, float* col_buff)
		{
			if (order_ == NCHW)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
			}
			else if (order_ == NHWC)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::conv_col2im_cpu(const float* col_buff, float* data)
		{
			col2im_cpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
				kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
		}

#ifdef USE_CUDA
		void convolution::conv_im2col_gpu(const float* data, float* col_buff)
		{
			if (order_ == NCHW)
			{
				im2col_gpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
			}
			else if (order_ == NHWC)
			{
				im2col_gpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::conv_col2im_gpu(const float* col_buff, float* data)
		{
			col2im_gpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
				kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
		}

#endif

		void convolution::setup_internal_params()
		{
			kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
			group_ = 1;
			weight_offset_ = kernelSize_ * kernelSize_;
			isfirst = true;
		}

		void convolution::setup_internal_params(int group)
		{
			kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
			group_ = group;
			weight_offset_ = kernelSize_*kernelSize_;
			isfirst = true;
		}


		void convolution::forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
				col_buff = col_buffer_->cpu_data();

				//std::cout << "im2col:" << std::endl;
				//for (size_t i = 0; i < 10; i++)
				//{
				//	std::cout << col_buff[i] << " ";
				//}
				//std::cout << std::endl;

			}

			if (order_ == NCHW)
			{
				for (int g = 0; g < group_; ++g)
				{
					if (group_ == 1)
					{
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
							conv_out_spatial_dim_, kernel_dim_, 1.0f,
							weights, col_buff, 0.0f, output);
					}
					else
					{
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
							conv_out_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
							weights + kernelSize_ * kernelSize_ * g, col_buff + conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + conv_out_spatial_dim_ * g);
					}
				}

				std::cout << "before bias:" << std::endl;
				for (size_t i = 0; i < 10; i++)
				{
					std::cout << output[i] << " ";
				}
				std::cout << std::endl;
			}
			else if (order_ == NHWC)
			{
				for (int g = 0; g < group_; ++g)
				{
					if (group_ == 1)
					{
						//math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, conv_out_spatial_dim_, output_Channel_,
						//    kernel_dim_, 1.0f,
						//	col_buff, weights, 0.0f, output);
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, conv_out_spatial_dim_, output_Channel_,
							kernel_dim_, 1.0f,
							col_buff, weights, 0.0f, output);
					}
					else
					{
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, conv_out_spatial_dim_, output_Channel_ / group_,
							kernelSize_ * kernelSize_, 1.0f,
							col_buff + conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * g, weights + kernelSize_ * kernelSize_ * g, 0.0f, output + conv_out_spatial_dim_ * g);

						//math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
						//	conv_out_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
						//	weights + kernelSize_ * kernelSize_ * g, col_buff + conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + conv_out_spatial_dim_ * g);
						
						//float* temp_col_data = (float*)malloc(conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * sizeof(float));
						//for (size_t i = 0; i < conv_out_spatial_dim_; i++)
						//{
						//	memcpy(temp_col_data + i * kernelSize_ * kernelSize_, col_buff + i * kernelSize_ * kernelSize_ * input_Channel_ + g * kernelSize_ * kernelSize_, kernelSize_ * kernelSize_ * sizeof(float));
						//}

						//float* temp_weights_data = (float*)malloc((output_Channel_ / group_) * kernelSize_ * kernelSize_ * sizeof(float));
						//for (size_t i = 0; i < kernelSize_ * kernelSize_; i++)
						//{
						//	for (size_t j = 0; j < (output_Channel_ / group_); j++)
						//	{
						//		temp_weights_data[i * (output_Channel_ / group_) + j] = weights[kernelSize_ * kernelSize_ * g + i];
						//	}
						//}

						//
						//math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, conv_out_spatial_dim_, output_Channel_ / group_,
						//	kernelSize_ * kernelSize_, 1.0f,
						//	temp_col_data, temp_weights_data, 0.0f, output + output_Channel_ * g);

						//delete temp_col_data;
						//delete temp_weights_data;
					}
				}

				if (group_ != 1)
				{
					float* temp_data = (float*)malloc(conv_out_spatial_dim_ * output_Channel_ * sizeof(float));
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

					memcpy(output, temp_data, conv_out_spatial_dim_ * output_Channel_ * sizeof(float));
					delete temp_data;
				}

				std::cout << "before bias:" << std::endl;
				for (size_t i = 0; i < 10; i++)
				{
					std::cout << output[i * output_Channel_] << " ";
				}
				std::cout << std::endl;
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			
		}

		void convolution::forward_cpu_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					out_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
					1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, out_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}

		}


#ifdef USE_CUDA
		void convolution::forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;
			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_gpu(input, gpu_temp_col_buffer_);
				col_buff = col_buffer_->gpu_data();
			}
			for (int g = 0; g < group_; ++g)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
					conv_out_spatial_dim_, kernel_dim_, 1.0f, weights + weight_offset_ * g, col_buff + col_offset_ * g,
					0.0f, output + output_offset_ * g);
			}
		}

		void convolution::forward_gpu_bias(cublasHandle_t cublas_handle_, float* output, const float* bias)
		{
			math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
				out_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
		}

#endif

		void convolution::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			const int num = bottom->data_shape()[0];
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			order_ = bottom->order();

			//
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();


			if (order_ == NCHW)
			{
				output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
				out_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
				output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);
				for (int n = 0; n < num; n++)
				{
					forward_cpu_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_cpu_bias(top_data + n * top_dim, bias);
					}
				}
			}
			else if (order_ == NHWC)
			{
				output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
				out_spatial_dim_ = output_dim_w_*output_dim_h_;
				//col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
				//output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);
				for (int n = 0; n < num; n++)
				{
					forward_cpu_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_cpu_bias(top_data + n * top_dim, bias);
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


