#ifdef USE_CUDA
#include "conv_native_gpu.hpp"
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

		void conv_native_gpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			if (group_>1)
			{
				conv_depthwise_native_gpu depthwise_conv(input_Channel_, output_Channel_, kernelSize_, group_, stride_, pad_, bias_term_, device_);
				depthwise_conv.Forward(bottom, top);
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
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
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
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * output_spatial_dim_;
						output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
				{
					forward_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim, bias);
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
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					output_spatial_dim_ = output_dim_w_*output_dim_h_;
					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
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
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						output_spatial_dim_ = output_dim_w_*output_dim_h_;
						col_offset_ = kernel_dim_ * output_spatial_dim_;
						output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
						math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
					}
				}

				int bottom_dim_ = bottom->count(1, 4);
				int top_dim = top->count(1, 4);

				for (int n = 0; n < num; n++)
				{
					forward_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_bias(top_data + n * top_dim, bias);
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_gpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;
			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
				col_buff = col_buffer_->gpu_data();
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f, weights, col_buff, 0.0f, output);
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_, kernel_dim_, 1.0f, col_buff, weights, 0.0f, output);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_gpu::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

	}
}

#endif