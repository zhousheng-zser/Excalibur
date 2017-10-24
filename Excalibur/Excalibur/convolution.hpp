#pragma once
#ifndef _CONVOLUTION_HPP_
#define _CONVOLUTION_HPP_
#include "tensor.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"
#include <memory>

namespace excalibur
{
	class convolution
	{
		tensor* weights_;
		tensor* bias_;
		tensor* col_buffer_;
		int device_;
		/// parameters
		int input_Channel_;
		int output_Channel_;
		int kernelSize_;
		int stride_;
		int pad_;
		///
		std::vector<int> intput_shape_;
		std::vector<int> output_shape_;
		int group_;
		int out_spatial_dim_;
		int weight_offset_;
		bool bias_term_;
		///
		int num_kernels_im2col_;
		int num_kernels_col2im_;
		int conv_out_channels_;
		int conv_in_channels_;
		int conv_out_spatial_dim_;
		int kernel_dim_;
		int col_offset_;
		int output_offset_;
		tensor* bias_multiplier_;
		///
		inline void conv_im2col_cpu(const float* data, float* col_buff);
		inline void conv_col2im_cpu(const float* col_buff, float* data);
#ifdef USE_CUDA
		void conv_im2col_gpu(const float* data, float* col_buff);
		void conv_col2im_gpu(const float* col_buff, float* data);
#endif
		void setup_internal_params();
	public:
		convolution(int input_Channel, int output_Channel, int kernelSize,
			int stride, int pad, int device);
		~convolution();
		void set_weights(float* weights);
		void set_bias(float* bias);
		void set_bias_term(bool bias_term)
		{
			bias_term_ = bias_term;
		}
		void Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
#ifdef USE_CUDA
		void Forward_native_gpu(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
#endif
	private:
		void forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false);
		void forward_cpu_bias(float* output, const float* bias);
#ifdef USE_CUDA
		void forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false);
		void forward_gpu_bias(cublasHandle_t cublas_handle_, float* output, const float* bias);
#endif
	};
}


#endif //_CONVOLUTION_HPP_