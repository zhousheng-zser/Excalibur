#pragma once
#ifndef _CUDNN_CONVOLUTION_HPP_
#define _CUDNN_CONVOLUTION_HPP_
#include "convolution.hpp"
#ifdef USE_CUDNN
#include "cudnn.hpp"
namespace excalibur
{
	class cudnn_convolution
	{
		// init norm conv params
		std::shared_ptr<tensor<float>> weights_;
		std::shared_ptr<tensor<float>> bias_;

		int group_;
		int num_output_;
		int num_input_;
		int stride_;
		int pad_;
		int channels_;
		bool bias_term_;
		int kernel_size_;
		int device_;
		
		//cuDNN API:
		//cudnnHandle_t handle_;
		float one = 1.0, zero = 0.0;
		size_t size;
		cudnnTensorDescriptor_t xdesc;
		cudnnTensorDescriptor_t	ydesc;
		cudnnTensorDescriptor_t bdesc;
		cudnnFilterDescriptor_t wdesc;
		cudnnConvolutionDescriptor_t conv_desc;
		// algorithms for forward and backwards convolutions
		cudnnConvolutionFwdAlgo_t fwd_algo_;
		size_t workspace_limit_bytes = 8 * 1024 * 1024;
	public:
		cudnn_convolution(int input_Channel, int output_Channel, int kernelSize,
			int stride, int pad, bool bias_term, int device);
		~cudnn_convolution();
		void set_weights(float* weights);
		void set_bias(float* bias);
		//for compatiblity
		void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			//N/A
		}
		void Forward_cudnn_gpu(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
	};
}

#endif // USE_CUDNN
#endif //_CUDNN_CONVOLUTION_HPP_