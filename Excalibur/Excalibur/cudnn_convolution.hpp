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
		tensor* weights_;
		tensor* bias_;

		int num_;
		int group_;
		int num_output_;
		int num_input_;
		int stride_;
		int pad_;
		int channels_;
		std::shared_ptr<tensor> kernel_shape_;
		bool bias_term_;
		int kernel_size_;
		int device_;
		//
		int bottom_dim_;
		int top_dim_;
		int out_spatial_dim_;
		int kernel_dim_;
		int bottom_offset_, top_offset_, bias_offset_, weight_offset_;
		//cuDNN API:
		bool handles_setup_;
		cudnnHandle_t* handle_;
		cudaStream_t*  stream_;

		// algorithms for forward and backwards convolutions
		cudnnConvolutionFwdAlgo_t *fwd_algo_;

		std::vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
		cudnnTensorDescriptor_t bias_desc_;
		cudnnFilterDescriptor_t filter_desc_;
		std::vector<cudnnConvolutionDescriptor_t> conv_descs_;

		size_t *workspace_fwd_sizes_;
		tensor *workspaceDataBlob;  // hold the real data for workspace
		void *workspaceData; // underlying storage
		void **workspace;  // aliases into workspaceData
	public:
		cudnn_convolution(int input_Channel, int output_Channel, int kernelSize,
			int stride, int pad, bool bias_term, int device);
		~cudnn_convolution();
		void set_weights(float* weights);
		void set_bias(float* bias);
		//for compatiblity
		void Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
		{
			//N/A
		}
		void Forward_cudnn_gpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
	};
}

#endif // USE_CUDNN
#endif //_CUDNN_CONVOLUTION_HPP_