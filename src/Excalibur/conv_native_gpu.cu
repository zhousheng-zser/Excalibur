#include "conv_native_gpu.hpp"
#include <iostream>
#include "Excalibur/depthwise_conv_kernel.cuh"
#include "device_launch_parameters.h"

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
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

		__global__ void depthwise_conv_kernel(const int nthreads,
			const signed char* const bottom_data, const int num, const int channels,
			const int height, const int width, const int conved_height,
			const int conved_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, const signed char* const weight, const float *scales, const float* const bias, const bool bias_term_, orderType order)
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
					int aveval_int32 = 0;
					float aveval = 0;
					const signed char* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					const signed char* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval_int32 += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}

					aveval = aveval_int32 / (scales[0] * scales[1 + c]);
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
					int aveval_int32 = 0;
					float aveval = 0;
					const signed char* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval_int32 += bottom_data[(n * height * width + h * width + w) * channels + c] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}

					aveval = aveval_int32 / (scales[0] * scales[1 + c]);
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


		__global__ void kernel_float32_to_int8(const float *fp32_data, float scale, signed char *int8_data)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			float tmp;
			float value = fp32_data[totalID] * scale;
			
			if (value >= 0.f)
			{
				tmp = value + 0.5;
			}
			else
			{
				tmp = value - 0.5;
			}

			if (tmp > 127)
			{
				tmp = 127;
			}
			else if (tmp < -128)
			{
				tmp = -128;
			}

			int8_data[totalID] = (signed char)tmp;
		}


		__global__ void kernel_int32_to_float32(const int *int32_data, const float* scales, float *fp32_data)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			fp32_data[totalID] = int32_data[totalID] / (scales[0] * scales[1]);
		}


		void conv_native_gpu::Forward(cublasHandle_t &cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			order_ = bottom->order();
			num_ = bottom->num();
			const int channel = bottom->channels();
			const int height = bottom->height();
			const int width = bottom->width();
			input_spatial_dim_ = width * height;
			bottom_dim_ = bottom->count(1, 4);
			bottom_data = bottom->gpu_data();
			input_shape_.clear();
			input_shape_ = bottom->data_shape();
			output_dim_h_ = (height + 2 * pad_ - kernelSize_) / stride_ + 1;
			output_dim_w_ = (width + 2 * pad_ - kernelSize_) / stride_ + 1;
			output_spatial_dim_ = output_dim_w_ * output_dim_h_;
			
			if (int8_quantization_)
			{
				bottom_int8_.reset(new tensor<signed char>(std::vector<int>{num_ * bottom_dim_}, 0));
				bottom_int8_data = bottom_int8_->mutable_gpu_data();

				const dim3 block_size(channel, 1, 1);
				const dim3 grid_size(width, height, num_);

				float bottom_scale;
				CUDA_CHECK(cudaMemcpy(&bottom_scale, scales_data, 1 * sizeof(float), cudaMemcpyDefault));
				kernel_float32_to_int8 << <grid_size, block_size>> > (bottom_data, bottom_scale, bottom_int8_data);
			}

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
			}
			else if (order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			top_data = top->mutable_gpu_data();
			top_dim_ = top->count(1, 4);
			int count = top->count();

			if (group_ > 1)
			{
				if (int8_quantization_)
				{
					if (bias_term_) 
					{
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
							count, bottom_int8_data, num_, input_Channel_,
							height, width, output_dim_h_, output_dim_w_, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights_int8_data, scales_data, bias_data, bias_term_, order_);
					}
					else 
					{
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
							count, bottom_int8_data, num_, input_Channel_,
							height, width, output_dim_h_, output_dim_w_, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights_int8_data, scales_data, 0, bias_term_, order_);
					}
				}
				else
				{
					if (bias_term_) 
					{
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
							count, bottom_data, num_, input_Channel_,
							height, width, output_dim_h_, output_dim_w_, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights_data, bias_data, bias_term_, order_);
					}
					else 
					{
						depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
							count, bottom_data, num_, input_Channel_,
							height, width, output_dim_h_, output_dim_w_, kernelSize_,
							kernelSize_, stride_, stride_, pad_, pad_, top_data, weights_data, 0, bias_term_, order_);
					}
				}
			}
			else if (group_ == 1)
			{
				if (order_ == NCHW)
				{
					if (int8_quantization_)
					{
						top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
						top_int32_data = top_int32_->mutable_gpu_data();
						col_buffer_int8_.reset(new tensor<signed char>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
						col_buffer_int8_data = col_buffer_int8_->mutable_gpu_data();
					}
					else
					{
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
						col_buffer_data = col_buffer_->mutable_gpu_data();
					}

					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_gpu_data();

					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
					math_functions::gpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);

					for (int n = 0; n < num_; n++)
					{
						if (int8_quantization_)
						{
							forward_gemm(cublas_handle_, bottom_int8_data + n * bottom_dim_, weights_int8_data, top_int32_data + n * top_dim_);

							const dim3 block_size(output_Channel_, 1, 1);
							const dim3 grid_size(output_dim_w_, output_dim_h_, 1);
							kernel_int32_to_float32<<<grid_size, block_size >>>(top_int32_data + n * top_dim_, scales_data, top_data + n * top_dim_);
						}
						else
						{
							forward_gemm(cublas_handle_, bottom_data + n * bottom_dim_, weights_data, top_data + n * top_dim_);
						}

						if (bias_term_)
						{
							forward_bias(cublas_handle_, top_data + n * top_dim_, bias_data);
						}
					}
				}
				else if (order_ == NHWC)
				{
					if (int8_quantization_)
					{
						top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
						top_int32_data = top_int32_->mutable_gpu_data();
						col_buffer_int8_.reset(new tensor<signed char>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
						col_buffer_int8_data = col_buffer_int8_->mutable_gpu_data();
					}
					else
					{
						col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_, output_dim_h_, output_dim_w_}, device_));
						col_buffer_data = col_buffer_->mutable_gpu_data();
					}

					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_gpu_data();
					output_spatial_dim_ = output_dim_w_ * output_dim_h_;
					col_offset_ = kernel_dim_ * output_spatial_dim_;
					output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
					math_functions::gpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);

					for (int n = 0; n < num_; n++)
					{
						if (int8_quantization_)
						{
							forward_gemm(cublas_handle_, bottom_int8_data + n * bottom_dim_, weights_int8_data, top_int32_data + n * top_dim_);

							const dim3 block_size(output_Channel_, 1, 1);
							const dim3 grid_size(output_dim_w_, output_dim_h_, 1);
							kernel_int32_to_float32 << <grid_size, block_size >> > (top_int32_data + n * top_dim_, scales_data, top_data + n * top_dim_);
						}
						else
						{
							forward_gemm(cublas_handle_, bottom_data + n * bottom_dim_, weights_data, top_data + n * top_dim_);
						}

						if (bias_term_)
						{
							forward_bias(cublas_handle_, top_data + n * top_dim_, bias_data);
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
		}


		void conv_native_gpu::forward_gemm(cublasHandle_t &cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col)
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

		void conv_native_gpu::forward_gemm(cublasHandle_t &cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col)
		{
			const signed char* col_buff = input;
			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_gpu(input, col_buffer_int8_->mutable_gpu_data());
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::gpu_gemmEx(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, weights, col_buffer_int8_data, output);
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_gemmEx(cublas_handle_, CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_, kernel_dim_, col_buffer_int8_data, weights, output);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_native_gpu::forward_bias(cublasHandle_t &cublas_handle_, float* output, const float* bias)
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
#endif //!USE_CUDA
	}
}
