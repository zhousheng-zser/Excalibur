#include "../../include/excalibur/conv_cudnn_gpu.hpp"
#include "../../include/excalibur/depthwise_conv_kernel.cuh"
#include <iostream>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
#ifdef USE_CUDNN
		void conv_cudnn_gpu::Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();

			if (order_ == NCHW)
			{
				if (xdesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc));
				}
				if (ydesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc));
				}
				if (wdesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
				}
				if (conv_desc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
				}

#if CUDNN_VERSION_MIN(7, 0, 0)
				CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group_));
#else
				if (group_>1)
				{
					order_ = bottom->order();
					const int height = bottom->height();
					const int width = bottom->width();
					const int num = bottom->num();
					int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
					int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;

					top.reset(new tensor<float>(std::vector<int>{num, this->output_Channel_, height_out, width_out}, this->device_, order_));
					const float* bottom_data = bottom->gpu_data();
					float* top_data = top->mutable_gpu_data();
					const float* weights = weights_->gpu_data();
					const int count = top->count();
					if (bias_term_) {
						const float* bias = bias_->gpu_data();
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
							count, bottom_data, num, input_Channel_,
							height, width, height_out, width_out, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights, bias, bias_term_, order_);
					}
					else {
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
							count, bottom_data, num, input_Channel_,
							height, width, height_out, width_out, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights, 0, bias_term_, order_);
					}

					return;
				}
#endif

				// set params descriptor
				CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
					output_Channel_, input_Channel_ / group_, kernelSize_, kernelSize_));
				CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_, pad_, stride_, stride_,
					1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
				if (bias_term_)
				{
					if (bdesc == nullptr)
					{
						CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc));
					}
					
					CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, output_Channel_, 1, 1));
				}
				current_size = 0;

				// calcu output parms
				const int height = bottom->height();
				const int width = bottom->width();
				const int num = bottom->num();
				int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
				int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, this->output_Channel_, height_out, width_out}, this->device_, order_));

				CUDNN_CHECK(cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					num, input_Channel_, height, width));
				CUDNN_CHECK(cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
					num, output_Channel_, height_out, width_out));
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
				/// To avoid malloc workspace every forward, this is a very costly operation
				if (size > current_size)
				{
					if (extra != nullptr)
					{
						CUDA_CHECK(cudaFree(extra));
					}
					CUDA_CHECK(cudaMalloc((void **)&extra, size));
					current_size = size;
				}
				
				const float* bottom_data = bottom->gpu_data();
				float* top_data = top->mutable_gpu_data();
				
				// FORWARD!
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
			else if (order_ == NHWC)
			{
				if (xdesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc));
				}
				if (ydesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc));
				}
				if (wdesc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
				}
				if (conv_desc == nullptr)
				{
					CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
				}

#if CUDNN_VERSION_MIN(7, 0, 0)
				CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group_));
#else
				if (group_>1)
				{
					order_ = bottom->order();
					const int height = bottom->height();
					const int width = bottom->width();
					const int num = bottom->num();
					int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
					int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;

					top.reset(new tensor<float>(std::vector<int>{num, height_out, width_out, this->output_Channel_}, this->device_, order_));
					const float* bottom_data = bottom->gpu_data();
					float* top_data = top->mutable_gpu_data();
					const float* weights = weights_->gpu_data();
					const int count = top->count();
					if (bias_term_) {
						const float* bias = bias_->gpu_data();
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
							count, bottom_data, num, input_Channel_,
							height, width, height_out, width_out, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights, bias, bias_term_, order_);
					}
					else {
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
							count, bottom_data, num, input_Channel_,
							height, width, height_out, width_out, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights, 0, bias_term_, order_);
					}

					return;
				}
#endif

				// set params descriptor
				CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
					output_Channel_, input_Channel_ / group_, kernelSize_, kernelSize_));
				CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_, pad_, stride_, stride_,
					1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
				if (bias_term_)
				{
					if (bdesc == nullptr)
					{
						CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc));
					}
					
					CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, output_Channel_, 1, 1));
				}

				current_size = 0;

				// calcu output parms
				const int height = bottom->height();
				const int width = bottom->width();
				const int num = bottom->num();
				int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
				int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, height_out, width_out, this->output_Channel_}, this->device_, order_));
				CUDNN_CHECK(cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
					num, input_Channel_, height, width));
				CUDNN_CHECK(cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
					num, output_Channel_, height_out, width_out));

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
				/// To avoid malloc workspace every forward, this is a very costly operation
				if (size > current_size)
				{
					if (extra != nullptr)
					{
						CUDA_CHECK(cudaFree(extra));
					}
					CUDA_CHECK(cudaMalloc((void **)&extra, size));
					current_size = size;
				}

				const float* bottom_data = bottom->gpu_data();
				float* top_data = top->mutable_gpu_data();

				// FORWARD!
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
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#endif //!USE_CUDNN
#endif //!USE_CUDA
	}
}