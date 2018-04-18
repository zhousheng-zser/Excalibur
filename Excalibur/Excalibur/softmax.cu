#include "math_functions.hpp"
#ifdef USE_CUDA
#include "softmax.hpp"

namespace excalibur
{
	__global__ void kernel_channel_max(const int num, const int channels,
		const int spatial_dim, const float* data, float* out) {
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
			int n = index / spatial_dim;
			int s = index % spatial_dim;
			float maxval = -FLT_MAX;
			for (int c = 0; c < channels; ++c) {
				maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
			}
			out[index] = maxval;
		}
	}

	__global__ void kernel_channel_subtract(const int count,
		const int num, const int channels,
		const int spatial_dim, const float* channel_max, float* data) {
		CUDA_KERNEL_LOOP(index, count) {
			int n = index / channels / spatial_dim;
			int s = index % spatial_dim;
			data[index] -= channel_max[n * spatial_dim + s];
		}
	}

	__global__ void kernel_exp(const int count, const float* data, float* out) {
		CUDA_KERNEL_LOOP(index, count) {
			out[index] = exp(data[index]);
		}
	}

	__global__ void kernel_channel_sum(const int num, const int channels,
		const int spatial_dim, const float* data, float* channel_sum) {
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
			int n = index / spatial_dim;
			int s = index % spatial_dim;
			float sum = 0;
			for (int c = 0; c < channels; ++c) {
				sum += data[(n * channels + c) * spatial_dim + s];
			}
			channel_sum[index] = sum;
		}
	}

	__global__ void kernel_channel_div(const int count,
		const int num, const int channels,
		const int spatial_dim, const float* channel_sum, float* data) {
		CUDA_KERNEL_LOOP(index, count) {
			int n = index / channels / spatial_dim;
			int s = index % spatial_dim;
			data[index] /= channel_sum[n * spatial_dim + s];
		}
	}

	__global__ void kernel_channel_dot(const int num, const int channels,
		const int spatial_dim, const float* data_1, const float* data_2,
		float* channel_dot) {
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
			int n = index / spatial_dim;
			int s = index % spatial_dim;
			float dot = 0;
			for (int c = 0; c < channels; ++c) {
				dot += (data_1[(n * channels + c) * spatial_dim + s]
					* data_2[(n * channels + c) * spatial_dim + s]);
			}
			channel_dot[index] = dot;
		}
	}

	void softmax::Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		outer_num_ = bottom->num();
		if (bottom->data_shape().size() <= 2)
		{
			inner_num_ = 1;
			scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1}, device_));
		}
		else
		{
			inner_num_ = bottom->height()*bottom->width();
			scale_.reset(new tensor<float>(std::vector<int>{outer_num_, 1, bottom->height(), bottom->width()}, device_));
		}
		top.reset(new tensor<float>(bottom->data_shape(), device_));
		//
		const float* bottom_data = bottom->gpu_data();
		float* top_data = top->mutable_gpu_data();
		float* scale_data = scale_->mutable_gpu_data();
		int count = bottom->count();
		int channels = bottom->channels();
		//
		math_functions::excalibur_copy(count, bottom_data, top_data, device_);
		kernel_channel_max << <CUDA_GET_BLOCKS(outer_num_ * inner_num_),
			CUDA_NUM_THREADS >> >(outer_num_, channels, inner_num_, top_data,
				scale_data);
		// subtract
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_channel_subtract << <CUDA_GET_BLOCKS(count),
			CUDA_NUM_THREADS >> >(count, outer_num_, channels, inner_num_,
				scale_data, top_data);
		// exponentiate
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_exp << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
			count, top_data, top_data);
		// sum after exp
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_channel_sum << <CUDA_GET_BLOCKS(outer_num_ * inner_num_),
			CUDA_NUM_THREADS >> >(outer_num_, channels, inner_num_, top_data,
				scale_data);
		// divide
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_channel_div << <CUDA_GET_BLOCKS(count),
			CUDA_NUM_THREADS >> >(count, outer_num_, channels, inner_num_,
				scale_data, top_data);
	}

#ifdef USE_CUDNN
	void softmax::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
	{
		outer_num_ = bottom->num();
		int c = bottom->channels();
		if (bottom->data_shape().size() <= 2)
		{
			inner_num_ = 1;
		}
		else
		{
			inner_num_ = bottom->height()*bottom->width();
		}
		int h = inner_num_;
		int w = 1;
		top.reset(new tensor<float>(bottom->data_shape(), device_));
		//
		CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc_, CUDNN_DATA_FLOAT,
			outer_num_, c, h, w, c*h*w, h*w, w, 1));
		CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc_, CUDNN_DATA_FLOAT,
			outer_num_, c, h, w, c*h*w, h*w, w, 1));
		const float* bottom_data = bottom->gpu_data();
		float* top_data = top->mutable_gpu_data();

		CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handle_, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&one,
			bottom_desc_, bottom_data,
			&zero,
			top_desc_, top_data));
	}
#endif
}
#endif