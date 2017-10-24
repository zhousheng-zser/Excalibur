#include "convolution.hpp"
#ifdef USE_CUDA
#include <filesystem>
#include <iostream>
namespace excalibur
{
	void convolution::Forward_native_gpu(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		
		const int num = bottom->data_shape()[0];
		const float* bottom_data = bottom->gpu_data();
		const float* weights = weights_->gpu_data();
		const float* bias = bias_->gpu_data();
		//
		intput_shape_.clear();
		intput_shape_ = bottom->data_shape();
		int output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
		int output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;
		top.reset(new tensor(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_));
		//
		//auto p0 = std::chrono::system_clock::now();
		float* top_data = (top)->mutable_gpu_data();
		/*auto p1 = std::chrono::system_clock::now();
		std::cout << "forward gpu gemm time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl;*/
		col_buffer_.reset(new tensor(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
		bias_multiplier_.reset(new tensor(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
		conv_out_spatial_dim_ = output_dim_w_*output_dim_h_;
		out_spatial_dim_ = output_dim_w_*output_dim_h_;
		col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
		output_offset_ = output_Channel_ * conv_out_spatial_dim_ / group_;
		math_functions::gpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_gpu_data());
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
}
#endif