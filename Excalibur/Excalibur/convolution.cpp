#include "convolution.hpp"
//#include "cblas.h"
#include <iostream>
#include <filesystem>

namespace excalibur
{
	convolution::convolution(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, int device)
	{
		input_Channel_ = input_Channel;
		output_Channel_ = output_Channel;
		kernelSize_ = kernelSize;
		stride_ = stride;
		pad_ = pad;
		device_ = device;
		weights_ = new tensor(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_}, device_);
		bias_ = new tensor(std::vector<int>{output_Channel_}, device_);
		setup_internal_params();
	}

	convolution::~convolution()
	{
		delete weights_;
		delete bias_;
		//delete bias_multiplier_;
		//delete col_buffer_;
	}

	void convolution::set_bias(float* bias)
	{
		bias_->set_cpu_data(bias);
	}

	void convolution::set_weights(float* weights)
	{
		weights_->set_cpu_data(weights);
	}

	void convolution::conv_im2col_cpu(const float* data, float* col_buff)
	{
		im2col_cpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
			kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff);
	}

	void convolution::conv_col2im_cpu(const float* col_buff, float* data)
	{
		col2im_cpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
			kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
	}

#ifdef USE_CUDA
	void convolution::conv_im2col_gpu(const float* data, float* col_buff)
	{
		im2col_gpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
			kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff);
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
		weight_offset_ = output_Channel_ * kernel_dim_ / group_;
	}


	void convolution::forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
	{
		const float* col_buff = input;
		if (kernelSize_!=1)
		{
			conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
			col_buff = col_buffer_->cpu_data();
		}
		//auto p0 = std::chrono::system_clock::now();
		for (int g = 0; g < group_; ++g)
		{
			math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_ ,
				conv_out_spatial_dim_, kernel_dim_, 1.0f,
				weights + weight_offset_ * g, col_buff + col_offset_ * g, 0.0f, output + output_offset_ * g);
		}
		/*auto p1 = std::chrono::system_clock::now();
		std::cout << "forward gemm time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl;*/
	}

	void convolution::forward_cpu_bias(float* output, const float* bias)
	{
		//auto p0 = std::chrono::system_clock::now();
		math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
			out_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
			1.0f, output);
		/*auto p1 = std::chrono::system_clock::now();
		std::cout << "forward bias time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl;*/
	}


#ifdef USE_CUDA
	void convolution::forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col)
	{
		const float* col_buff = input;
		if (kernelSize_ != 1)
		{
			conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
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

	void convolution::Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		const int num = bottom->data_shape()[0];
		const float* bottom_data = bottom->cpu_data();
		const float* weights = weights_->cpu_data();
		const float* bias = bias_->cpu_data();
		//
		intput_shape_.clear();
		intput_shape_ = bottom->data_shape();
		int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
		int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
		top.reset(new tensor(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_));
		//

		float* top_data = (top)->mutable_cpu_data();
		col_buffer_.reset(new tensor(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
		bias_multiplier_.reset(new tensor(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
		conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
		out_spatial_dim_ = output_dim_w_*output_dim_h_;
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
		output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
		math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
		//
		int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
		int top_dim = (top)->data_shape()[1] * (top)->data_shape()[2] * (top)->data_shape()[3];
		//auto p0 = std::chrono::system_clock::now();
		for (int n = 0; n < num; n++)
		{
			forward_cpu_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
			if (bias_term_)
			{
				forward_cpu_bias(top_data + n * top_dim, bias);
			}
		}
		/*auto p1 = std::chrono::system_clock::now();
		std::cout << "forward gemm time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl;*/
	}

}

