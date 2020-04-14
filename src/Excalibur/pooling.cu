#ifdef USE_CUDA
#include "../../include/excalibur/pooling.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <algorithm>
#include <cfloat>
#include <vector>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		__global__ void MaxPoolForward(const int nthreads,
			const float* const bottom_data, const int num, const int channels,
			const int height, const int width, const int pooled_height,
			const int pooled_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, orderType order) {

			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, nthreads) {
					const int pw = index % pooled_width;
					const int ph = (index / pooled_width) % pooled_height;
					const int c = (index / pooled_width / pooled_height) % channels;
					const int n = index / pooled_width / pooled_height / channels;
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					const int hend = min(hstart + kernel_h, height);
					const int wend = min(wstart + kernel_w, width);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					float maxval = -FLT_MAX;
					const float* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							if (bottom_slice[h * width + w] > maxval) {
								maxval = bottom_slice[h * width + w];
							}
						}
					}
					top_data[index] = maxval;
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, nthreads) {
					const int c = index % channels;
					const int pw = (index / channels) % pooled_width;
					const int ph = (index / channels / pooled_width) % pooled_height;
					const int n = index / pooled_width / pooled_height / channels;

					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					const int hend = min(hstart + kernel_h, height);
					const int wend = min(wstart + kernel_w, width);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					float maxval = -FLT_MAX;
					const float* const bottom_slice =
						bottom_data + n * height * width * channels;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							if (bottom_slice[(h * width + w) * channels + c] > maxval) {
								maxval = bottom_slice[(h * width + w) * channels + c];
							}
						}
					}
					top_data[index] = maxval;
				}
			}
			else
			{
				return;
			}
		}

		__global__ void AvePoolForward(const int nthreads,
			const float* const bottom_data, const int num, const int channels,
			const int height, const int width, const int pooled_height,
			const int pooled_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, orderType order) {

			if (order == NCHW)
			{
				CUDA_KERNEL_LOOP(index, nthreads) {
					const int pw = index % pooled_width;
					const int ph = (index / pooled_width) % pooled_height;
					const int c = (index / pooled_width / pooled_height) % channels;
					const int n = index / pooled_width / pooled_height / channels;
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + kernel_h, height + pad_h);
					int wend = min(wstart + kernel_w, width + pad_w);
					const int pool_size = (hend - hstart) * (wend - wstart);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					hend = min(hend, height);
					wend = min(wend, width);
					float aveval = 0;
					const float* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval += bottom_slice[h * width + w];
						}
					}
					top_data[index] = aveval / pool_size;
				}
			}
			else if (order == NHWC)
			{
				CUDA_KERNEL_LOOP(index, nthreads) {
					const int c = index % channels;
					const int pw = (index / channels) % pooled_width;
					const int ph = (index / channels / pooled_width) % pooled_height;
					const int n = index / pooled_width / pooled_height / channels;

					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + kernel_h, height + pad_h);
					int wend = min(wstart + kernel_w, width + pad_w);
					const int pool_size = (hend - hstart) * (wend - wstart);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					hend = min(hend, height);
					wend = min(wend, width);
					float aveval = 0;
					const float* const bottom_slice =
						bottom_data + n * height * width * channels;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval += bottom_slice[(h * width + w) * channels + c];
						}
					}
					top_data[index] = aveval / pool_size;
				}
			}
			else
			{
				return;
			}
		}

		__global__ void GlobalAvePoolForward(const int spatial_dim,
			const float* bottom_data, float* top_data) {
			__shared__ float buffer[CUDA_NUM_THREADS];
			unsigned int tid = threadIdx.x;
			buffer[tid] = 0;
			__syncthreads();

			for (int j = tid; j < spatial_dim; j += blockDim.x) {
				buffer[tid] += bottom_data[blockIdx.x * spatial_dim + j];
			}
			__syncthreads();

			for (int i = blockDim.x / 2; i > 0; i >>= 1) {
				if (tid < i) {
					buffer[threadIdx.x] += buffer[threadIdx.x + i];
				}
				__syncthreads();
			}

			if (tid == 0) {
				top_data[blockIdx.x] = buffer[0] / spatial_dim;
			}
		}


		void pooling::Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = bottom->num();
			channels_ = bottom->channels();
			height_ = bottom->height();
			width_ = bottom->width();
			order_ = bottom->order();
			pooled_height_ = static_cast<int>(ceil(static_cast<float>(
				height_ + 2 * pad_ - kernel_) / stride_)) + 1;
			pooled_width_ = static_cast<int>(ceil(static_cast<float>(
				width_ + 2 * pad_ - kernel_) / stride_)) + 1;
			int spatial_dim;

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, device_, order_));
				spatial_dim = bottom->count(2, 4);
			}
			else if(order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{num, pooled_height_, pooled_width_, channels_}, device_, order_));
				spatial_dim = bottom->count(1, 3);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			//
			const float* bottom_data = bottom->gpu_data();
			float* top_data = (top)->mutable_gpu_data();
			const int top_count = (top)->count(0, 4);
			switch (type_)
			{
			case MAX:
				// NOLINT_NEXT_LINE(whitespace/operators)
				MaxPoolForward << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> >(
					top_count, bottom_data, bottom->num(), channels_,
					height_, width_, pooled_height_, pooled_width_, kernel_,
					kernel_, stride_, stride_, pad_, pad_, top_data, order_);
				break;
			case AVE:
				if (order_ == NCHW && pooled_height_ == 1 && pooled_width_ == 1)
				{
					GlobalAvePoolForward << <num * channels_, CUDA_NUM_THREADS >> > (
						spatial_dim, bottom_data, top_data);
				}
				else
				{
					AvePoolForward << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (
						top_count, bottom_data, bottom->num(), channels_,
						height_, width_, pooled_height_, pooled_width_, kernel_,
						kernel_, stride_, stride_, pad_, pad_, top_data, order_);
				}

				break;
			default:
				LOG(FATAL) << "Unknown pooling method.";
			}
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		void pooling::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = bottom->num();
			channels_ = bottom->channels();
			height_ = bottom->height();
			width_ = bottom->width();
			order_ = bottom->order();
			pooled_height_ = static_cast<int>(ceil(static_cast<float>(
				height_ + 2 * pad_ - kernel_) / stride_)) + 1;
			pooled_width_ = static_cast<int>(ceil(static_cast<float>(
				width_ + 2 * pad_ - kernel_) / stride_)) + 1;

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, device_, order_));
				CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc_, CUDNN_DATA_FLOAT,
					num, channels_, height_, width_, width_ * height_ * channels_, width_ * height_, width_, 1));
				CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc_, CUDNN_DATA_FLOAT,
					num, channels_, pooled_height_, pooled_width_, pooled_width_ * pooled_height_ * channels_, pooled_width_ * pooled_height_, pooled_width_, 1));
			}
			else if (order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{num, pooled_height_, pooled_width_, channels_}, device_, order_));
				CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc_, CUDNN_DATA_FLOAT,
					num, channels_, height_, width_, width_ * height_ * channels_, 1, width_ * channels_, channels_));
				CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc_, CUDNN_DATA_FLOAT,
					num, channels_, pooled_height_, pooled_width_, pooled_width_ * pooled_height_ * channels_, 1, pooled_width_ * channels_, channels_));
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			
			const float* bottom_data = bottom->gpu_data();
			float* top_data = top->mutable_gpu_data();
			CUDNN_CHECK(cudnnPoolingForward(cudnn_handle_, pooling_desc_,
				&one,
				bottom_desc_, bottom_data,
				&zero,
				top_desc_, top_data));
		}
#endif
	}
}


#endif
