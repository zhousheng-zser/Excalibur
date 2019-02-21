#ifdef USE_CUDA
#include "convolution.hpp"
#include <filesystem>
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		__global__ void depthwise_conv_kernel(const int nthreads,
			const float* const bottom_data, const int num, const int channels,
			const int height, const int width, const int conved_height,
			const int conved_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, const float* const weight, const float* const bias, const bool bias_term_, orderType order)
		{
			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, nthreads)
				{
					const int pw = index % conved_width;
					const int ph = (index / conved_width) % conved_height;
					const int c = (index / conved_width / conved_height) % channels;
					const int n = index / conved_width / conved_height / channels;
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + kernel_h, height + pad_h);
					int wend = min(wstart + kernel_w, width + pad_w);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					hend = min(hend, height);
					wend = min(wend, width);
					float aveval = 0;
					const float* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					const float* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend<kernel_h ? kernel_h - hend : 0;
					int kwstart = wend<kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}
					if (bias_term_) {
						aveval += bias[c];
					}
					top_data[index] = aveval;
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, nthreads)
				{
					const int c = index % channels;
					const int pw = (index / channels) % conved_width;
					const int ph = (index / channels / conved_width) % conved_height;
					const int n = index / conved_width / conved_height / channels;
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + kernel_h, height + pad_h);
					int wend = min(wstart + kernel_w, width + pad_w);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					hend = min(hend, height);
					wend = min(wend, width);
					float aveval = 0;
					const float* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend<kernel_h ? kernel_h - hend : 0;
					int kwstart = wend<kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval += bottom_data[(n * height * width + h * width + w) * channels + c] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}
					if (bias_term_) {
						aveval += bias[c];
					}
					top_data[index] = aveval;
				}
			}
			else
			{
				return;
			}
		}

		void convolution::Forward_native_gpu(cublasHandle_t cublas_handle_,
			const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			if (group_>1)
			{
				forward_depthwise_native_gpu(bottom, top);
				return;
			}
			const int num = bottom->num();
			const float* bottom_data = bottom->gpu_data();
			const float* weights = weights_->gpu_data();
			const float* bias = bias_->gpu_data();
			order_ = bottom->order();
			//
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();

			if (order_ == NCHW)
			{
				int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				//

				float* top_data = (top)->mutable_gpu_data();
				if (isfirst)
				{
					last_height = bottom->height();
					last_width = bottom->width();
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
					gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
					out_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
					output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
					math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					isfirst = false;
				}
				else
				{
					if (last_height != bottom->height() || last_width != bottom->width())
					{
						last_height = bottom->height();
						last_width = bottom->width();
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
						gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
						conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
						out_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
						output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
				{
					forward_gpu_gemm(cublas_handle_, bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_gpu_bias(cublas_handle_, top_data + n * top_dim, bias);
					}
				}
			}
			else if (order_ == NHWC)
			{
				int output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				int output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				//

				float* top_data = (top)->mutable_gpu_data();
				if (isfirst)
				{
					last_height = bottom->height();
					last_width = bottom->width();
					col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
					gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
					out_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
					output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
					math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					isfirst = false;
				}
				else
				{
					if (last_height != bottom->height() || last_width != bottom->width())
					{
						last_height = bottom->height();
						last_width = bottom->width();
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
						gpu_temp_col_buffer_ = col_buffer_->mutable_gpu_data();
						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
						conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
						out_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
						output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
				{
					forward_gpu_gemm(cublas_handle_, bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_gpu_bias(cublas_handle_, top_data + n * top_dim, bias);
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::forward_depthwise_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();
			const int height = bottom->height();
			const int width = bottom->width();
			const int num = bottom->num();
			int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
			int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;

			if (order_ == NCHW)
			{
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
			}
			else if (order_ == NHWC)
			{
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
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#ifdef USE_CUDNN
		void convolution::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();

			if (order_ == NCHW)
			{
				if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
				{
					LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
				}
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc));
				CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
				CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
				// set params descriptor
				CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
					output_Channel_ / group_, input_Channel_ / group_, kernelSize_, kernelSize_));
				CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_, pad_, stride_, stride_,
					1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
				if (bias_term_)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc));
					CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, output_Channel_ / group_, 1, 1));
				}
				current_size = 0;

				if (group_>1)
				{
					forward_depthwise_native_gpu(bottom, top);
					return;
				}
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
						cudaFree(extra);
					}
					CUDA_CHECK(cudaMalloc((void **)&extra, size));
					current_size = size;
				}

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
			}
			else if (order_ == NHWC)
			{
				if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
				{
					LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
				}
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&xdesc));
				CUDNN_CHECK(cudnnCreateTensorDescriptor(&ydesc));
				CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
				CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
				// set params descriptor
				CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
					output_Channel_ / group_, input_Channel_ / group_, kernelSize_, kernelSize_));
				CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_, pad_, stride_, stride_,
					1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
				if (bias_term_)
				{
					CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdesc));
					CUDNN_CHECK(cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
						1, output_Channel_ / group_, 1, 1));
				}
				current_size = 0;

				if (group_>1)
				{
					forward_depthwise_native_gpu(bottom, top);
					return;
				}
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
						cudaFree(extra);
					}
					CUDA_CHECK(cudaMalloc((void **)&extra, size));
					current_size = size;
				}

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
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
#endif
	}
}

#endif