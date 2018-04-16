#include "cudnn_convolution.hpp"
#ifdef USE_CUDNN
#include <algorithm>

namespace excalibur
{
	cudnn_convolution::cudnn_convolution(int input_Channel, int output_Channel, int kernelSize,
		int stride, int pad, bool bias_term, int device)
	{
		// Initalize conv params
		num_input_ = input_Channel;
		channels_ = num_input_;
		num_output_ = output_Channel;
		kernel_size_ = kernelSize;
		stride_ = stride;
		pad_ = pad;
		group_ = 1;
		bias_term_ = bias_term;
		device_ = device;
		//
		weights_.reset(new tensor<float>(std::vector<int>{num_input_*num_output_*kernel_size_*kernel_size_}, device_));
		bias_.reset(new tensor<float>(std::vector<int>{num_output_}, device_));
		// init handle
		//CUDNN_CHECK(cudnnCreate(&handle_));
		// create descriptor
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc));		
		CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
		CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
		// set params descriptor
		CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			num_output_ / group_, num_input_ / group_, kernel_size_, kernel_size_));
		CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_, pad_, stride_, stride_,
			1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		if (bias_term_)
		{
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
				1, num_output_ / group_, 1, 1));
		}
	}


	cudnn_convolution::~cudnn_convolution()
	{
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xdesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(ydesc));
		CUDNN_CHECK(cudnnDestroyFilterDescriptor(wdesc));
		CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
		if (bias_term_)
		{
			CUDNN_CHECK(cudnnDestroyTensorDescriptor(bdesc));
		}
		//CUDNN_CHECK(cudnnDestroy(handle_));
	}

	void cudnn_convolution::set_weights(float* weights)
	{
		weights_->set_cpu_data(weights);
	}

	void cudnn_convolution::set_bias(float* bias)
	{
		bias_->set_cpu_data(bias);
	}

	void cudnn_convolution::Forward_cudnn_gpu(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		// calcu output parms
		const int height = bottom->height();
		const int width = bottom->width();
		const int num = bottom->num();
		int height_out = (height + 2 * pad_ - kernel_size_) / stride_ + 1;
		int width_out = (width + 2 * pad_ - kernel_size_) / stride_ + 1;
		top.reset(new tensor<float>(std::vector<int>{num, this->num_output_, height_out, width_out}, this->device_));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
			num, channels_, height, width));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
			num, num_output_, height_out, width_out));
		CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_,
			xdesc,
			wdesc,
			conv_desc,
			ydesc,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes,
			&fwd_algo_));
		CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_,
			xdesc,
			wdesc,
			conv_desc,
			ydesc,
			fwd_algo_,
			&(size)));
		float *extra;
		CUDA_CHECK(cudaMalloc((void **)&extra, size));
		const float* bottom_data = bottom->gpu_data();
		float* top_data = top->mutable_gpu_data();
		// FORWARD!
		for (int g = 0; g < group_; g++)
		{
			CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle_, &one,
				xdesc, bottom_data, wdesc, weights_->gpu_data(),
				conv_desc, fwd_algo_,
				extra, size, &zero,
				ydesc, top_data));
			if (bias_term_)
			{
				CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &one, bdesc, bias_->gpu_data(),
					&one, ydesc, top_data));
			}
		}
		cudaFree(extra);
	}
}
#endif // USE_CUDNN