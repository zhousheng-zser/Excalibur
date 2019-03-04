#ifdef USE_CUDA
#include "conv_depthwise_native_gpu.hpp"
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

		void conv_depthwise_native_gpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
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

		void conv_depthwise_native_gpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col) {}

		void conv_depthwise_native_gpu::forward_bias(float* output, const float* bias) {}
	}
}

#endif //!USE_CUDA