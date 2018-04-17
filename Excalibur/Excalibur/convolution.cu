#include "convolution.hpp"
#ifdef USE_CUDA
#include <filesystem>
#include <iostream>
namespace excalibur
{
	void convolution::Forward_native_gpu(cublasHandle_t cublas_handle_, 
		const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		
		const int num = bottom->num();
		const float* bottom_data = bottom->gpu_data();
		const float* weights = weights_->gpu_data();
		const float* bias = bias_->gpu_data();
		//
		intput_shape_.clear();
		intput_shape_ = bottom->data_shape();
		
		int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
		int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
		top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_));
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
			if (last_height!= bottom->height()||last_width!= bottom->width())
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

#ifdef USE_CUDNN
	void convolution::Forward_cudnn_gpu(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		// calcu output parms
		const int height = bottom->height();
		const int width = bottom->width();
		const int num = bottom->num();
		int height_out = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
		int width_out = (width + 2 * pad_ - kernelSize_) / stride_ + 1;
		top.reset(new tensor<float>(std::vector<int>{num, this->output_Channel_, height_out, width_out}, this->device_));
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
#endif
}
#endif