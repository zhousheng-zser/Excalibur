#include "convolution.hpp"
#include "cblas.h"
#include <iostream>

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
		delete bias_multiplier_;
		delete col_buffer_;
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


	void convolution::forward_cpu_gemm(const float* input, float* output, bool skip_im2col)
	{
		const float* col_buff = input;
		if (kernelSize_!=1)
		{
			conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
			col_buff = col_buffer_->cpu_data();
		}
		for (int g = 0; g < group_; ++g)
		{
			int M = output_Channel_ / group_;
			int N = conv_out_spatial_dim_;
			int K = kernel_dim_;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
				1.0f, weights_->cpu_data() + weight_offset_ * g, K, col_buff + col_offset_ * g, N, 0.0f, output + output_offset_ * g, N );
		}
		if (bias_term_)
		{
			int M = output_Channel_;
			int N = out_spatial_dim_;
			int K = 1;
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
				1.0f, bias_->cpu_data(), K, bias_multiplier_->cpu_data(), N, 1.0f, output, N);
		}
	}

#ifdef USE_CUDA
	void convolution::forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, float* output, bool skip_im2col)
	{
		const float* col_buff = input;
		if (kernelSize_ != 1)
		{
			conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
			col_buff = col_buffer_->gpu_data();
		}
		for (int g = 0; g < group_; ++g)
		{
			int M = output_Channel_ / group_;
			int N = conv_out_spatial_dim_;
			int K = kernel_dim_;
			cublasOperation_t cuTransA = CUBLAS_OP_N;
			cublasOperation_t cuTransB = CUBLAS_OP_N;
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CUBLAS_CHECK(cublasSgemm(cublas_handle_, cuTransA, cuTransB, N, M, K, &alpha, col_buff + col_offset_ * g,
				N, weights_->gpu_data() + weight_offset_ * g, K, &beta, output + output_offset_ * g, N));
		}
		if (bias_term_)
		{
			int M = output_Channel_;
			int N = out_spatial_dim_;
			int K = 1;
			cublasOperation_t cuTransA = CUBLAS_OP_N;
			cublasOperation_t cuTransB = CUBLAS_OP_N;
			const float alpha = 1.0f;
			const float beta = 1.0f;
			CUBLAS_CHECK(cublasSgemm(cublas_handle_, cuTransA, cuTransB, N, M, K, &alpha, bias_multiplier_->gpu_data(), 
				N, bias_->gpu_data(),	K, &beta, output, N));
		}
	}

#endif

	void convolution::Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		const int num = bottom->data_shape()[0];
		const float* bottom_data = bottom->cpu_data();
		//
		intput_shape_.clear();
		intput_shape_ = bottom->data_shape();
		int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
		int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
		top.reset(new tensor(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_));
		//

		float* top_data = (top)->mutable_cpu_data();
		if (col_buffer_ !=nullptr)
		{
			delete col_buffer_;
		}
		col_buffer_ = new tensor(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_);
		if (bias_multiplier_!=nullptr)
		{
			delete bias_multiplier_;
		}
		bias_multiplier_ = new tensor(std::vector<int>{output_dim_w_*output_dim_h_}, device_);
		conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
		out_spatial_dim_ = output_dim_w_*output_dim_h_;
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
		output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
		for (int i = 0; i < output_dim_w_*output_dim_h_; i++)
		{
			bias_multiplier_->mutable_cpu_data()[i] = 1.0f;
		}
		//
		int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
		int top_dim = (top)->data_shape()[1] * (top)->data_shape()[2] * (top)->data_shape()[3];
		for (int n = 0; n < num; n++)
		{
			forward_cpu_gemm(bottom_data + n * bottom_dim_, top_data + n * top_dim);
		}
	}

}

