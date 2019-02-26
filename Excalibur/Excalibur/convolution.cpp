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

		void convolution::conv_im2col_cpu(const float* data, float* col_buff, int num)
		{
			if (order_ == NCHW)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_, num);
			}
			else if (order_ == NHWC)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_, num);
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

			////winograd
			//BT{ 4, 0,-5, 0, 1, 0, 
			//	0,-4,-4, 1, 1, 0, 
			//	0, 4,-4,-1, 1, 0, 
			//	0,-2,-1, 2, 1, 0, 
			//	0, 2,-1,-2, 1, 0, 
			//	0, 4, 0,-5, 0, 1 };
			//G{   1.0f/4,   0.0f,    0.0f,
			//	-1.0f/6,-1.0f/6, -1.0f/6, 
			//	-1.0f/6, 1.0f/6, -1.0f/6, 
			//	1.0f/24, 1.0f/12, 1.0f/6,
			//	1.0f/24,-1.0f/12, 1.0f/6,
			//	   0.0f,    0.0f,   1.0f
			//};
			//AT{ 1,1,1,1,1,0,
			//	0,1,-1,2,-2,0,
			//	0,1,1,4,4,0,
			//	0,1,-1,8,-8,1 }

			//}
		}

		void convolution::setup_internal_params(int group)
		{
			kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
			group_ = group;
			weight_offset_ = kernelSize_*kernelSize_;
			isfirst = true;
		}


#ifdef USE_CUDA
		void convolution::forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col)
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
						conv_out_spatial_dim_, kernel_dim_, 1.0f, weights, col_buff, 0.0f, output);
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasTrans, CblasTrans, conv_out_spatial_dim_,
						output_Channel_, kernel_dim_, 1.0f, col_buff, weights, 0.0f, output);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::forward_gpu_bias(cublasHandle_t cublas_handle_, float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
					out_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, out_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#endif

		void convolution::forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
				col_buff = col_buffer_->cpu_data();
			}

			//std::cout << "col_buffer:" << std::endl;
			//for (size_t i = 0; i < 50; i++)
			//{
			//	std::cout << *(col_buff + i) << " ";
			//}
			//std::cout << std::endl;
			//std::cout << std::endl;

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
						conv_out_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
							conv_out_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
							weights + kernelSize_ * kernelSize_ * g, col_buff + conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + conv_out_spatial_dim_ * g);
					}
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasTrans, conv_out_spatial_dim_, output_Channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, conv_out_spatial_dim_, output_Channel_ / group_,
							kernelSize_ * kernelSize_, 1.0f,
							col_buff + conv_out_spatial_dim_ * kernelSize_ * kernelSize_ * g, weights + kernelSize_ * kernelSize_ * g, 0.0f, output + conv_out_spatial_dim_ * g);
					}

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
				col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_/**group_*/, output_dim_h_, output_dim_w_}, device_));
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
			else
			{
				NOT_IMPLEMENTED;
			}
		}


#ifdef USE_MKL
void convolution::forward_cpu_gemm_batch(const float* input, const float* weights, float* output, int top_dim, int num, bool skip_im2col)
{
	const float* col_buff = input;

	if ((kernelSize_ != 1) || (order_ == NHWC))
	{
		conv_im2col_cpu(input, col_buffer_->mutable_cpu_data(), num);
		col_buff = col_buffer_->cpu_data();
	}

	if (order_ == NCHW)
	{
		for (int g = 0; g < group_; ++g)
		{
			math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
				conv_out_spatial_dim_, kernel_dim_ / group_, 1.0f, 
				weights + g * kernel_dim_ / group_, 0,
				col_buff + g * conv_out_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_, 0.0f, 
				output + conv_out_spatial_dim_ * g, top_dim, num);
		}
	}
	else if (order_ == NHWC)
	{
		for (int g = 0; g < group_; ++g)
		{
			math_functions::cpu_batch_sgemm(CblasTrans, CblasTrans, conv_out_spatial_dim_,
				output_Channel_ / group_, kernel_dim_ / group_, 1.0f, 
				col_buff + g * conv_out_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_,
				weights + g * kernel_dim_ / group_, 0, 0.0f, 
				output + conv_out_spatial_dim_ * g, top_dim, num);
		}

		if (group_ > 1)
		{
			float* temp_data = (float*)malloc(conv_out_spatial_dim_ * output_Channel_ * sizeof(float));
			for (int n = 0; n < num; n++)
			{
				int n_offset = n * conv_out_spatial_dim_ * output_Channel_;
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
				memcpy(output + n_offset, temp_data, conv_out_spatial_dim_ * output_Channel_ * sizeof(float));
			}

			delete temp_data;
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}

void convolution::forward_cpu_bias_batch(float* output, const float* bias, int num)
{
	if (order_ == NCHW)
	{
		math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
			conv_out_spatial_dim_, 1, 1.0f, bias, 0,
			bias_multiplier_->cpu_data(), conv_out_spatial_dim_, 1.0f, output, conv_out_spatial_dim_ * output_Channel_, num);
	}
	else if (order_ == NHWC)
	{
		math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, out_spatial_dim_,
			output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), conv_out_spatial_dim_, bias, 0,
			 1.0f, output, conv_out_spatial_dim_ * output_Channel_, num);
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}

void convolution::Forward_cpu_batch(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	const int num = bottom->data_shape()[0];
	const float* bottom_data = bottom->cpu_data();
	const float* weights = weights_->cpu_data();
	const float* bias = bias_->cpu_data();
	order_ = bottom->order();
	intput_shape_.clear();
	intput_shape_ = bottom->data_shape();

	if (order_ == NCHW)
	{
		output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;

		top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
		//

		float* top_data = (top)->mutable_cpu_data();
		col_buffer_.reset(new tensor<float>(std::vector<int>{num*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
		bias_multiplier_.reset(new tensor<float>(std::vector<int>{num*output_dim_w_*output_dim_h_}, device_));
		conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
		out_spatial_dim_ = output_dim_w_*output_dim_h_;
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
		output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
		math_functions::cpu_set(num*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
		//
		int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
		int top_dim = (top)->count(1, 4);

		forward_cpu_gemm_batch(bottom_data, weights, top_data, top_dim, num);
		if (bias_term_)
		{
			forward_cpu_bias_batch(top_data, bias, num);
		}

	}
	else if (order_ == NHWC)
	{
		output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;

		top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
		//

		float* top_data = (top)->mutable_cpu_data();
		col_buffer_.reset(new tensor<float>(std::vector<int>{num*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
		bias_multiplier_.reset(new tensor<float>(std::vector<int>{num*output_dim_w_*output_dim_h_}, device_));
		conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
		out_spatial_dim_ = output_dim_w_*output_dim_h_;
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
		output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
		math_functions::cpu_set(num*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
		//
		int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
		int top_dim = (top)->count(1, 4);

		forward_cpu_gemm_batch(bottom_data, weights, top_data, top_dim, num);

		if (bias_term_)
		{
			forward_cpu_bias_batch(top_data, bias, num);
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}
#endif // !USE_MKL

	}
}


