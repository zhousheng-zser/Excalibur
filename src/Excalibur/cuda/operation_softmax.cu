#include "../../../include/Excalibur/operation_softmax.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include <cfloat>
namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA

		__global__ void kernel_channel_max(const int num, const int channels,
			const int spatial_dim, const float* data, float* out, memory::orderType order) {

			if (order == memory::NCHW)
			{
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
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, num * spatial_dim) {
					int n = index / spatial_dim;
					int s = index % spatial_dim;
					float maxval = -FLT_MAX;
					for (int c = 0; c < channels; ++c) {
						maxval = max(data[index * channels + c], maxval);
					}
					out[index] = maxval;
				}
			}
			else
			{
				return;
			}
		}

		__global__ void kernel_channel_subtract(const int count,
			const int num, const int channels,
			const int spatial_dim, const float* channel_max, float* data, memory::orderType order) {

			if (order == memory::NCHW)
			{
				CUDA_KERNEL_LOOP(index, count) {
					int n = index / channels / spatial_dim;
					int s = index % spatial_dim;
					data[index] -= channel_max[n * spatial_dim + s];
				}
			}
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, count) {
					int n = index / channels / spatial_dim;
					int s = (index / channels) % spatial_dim;
					data[index] -= channel_max[n * spatial_dim + s];
				}
			}
			else
			{
				return;
			}
		}

		__global__ void kernel_exp(const int count, const float* data, float* out) {
			CUDA_KERNEL_LOOP(index, count) {
				out[index] = exp(data[index]);
			}
		}

		__global__ void kernel_channel_sum(const int num, const int channels,
			const int spatial_dim, const float* data, float* channel_sum, memory::orderType order) {

			if (order == memory::NCHW)
			{
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
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, num * spatial_dim) {
					int n = index / spatial_dim;
					int s = index % spatial_dim;
					float sum = 0;
					for (int c = 0; c < channels; ++c) {
						sum += data[index * channels + c];
					}
					channel_sum[index] = sum;
				}
			}
			else
			{
				return;
			}
		}

		__global__ void kernel_channel_div(const int count,
			const int num, const int channels,
			const int spatial_dim, const float* channel_sum, float* data, memory::orderType order) {

			if (order == memory::NCHW)
			{
				CUDA_KERNEL_LOOP(index, count) {
					int n = index / channels / spatial_dim;
					int s = index % spatial_dim;
					data[index] /= channel_sum[n * spatial_dim + s];
				}
			}
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, count) {
					int n = index / channels / spatial_dim;
					int s = (index / channels) % spatial_dim;
					data[index] /= channel_sum[n * spatial_dim + s];
				}
			}
			else
			{
				return;
			}
		}

		__global__ void kernel_channel_dot(const int num, const int channels,
			const int spatial_dim, const float* data_1, const float* data_2,
			float* channel_dot, memory::orderType order) {

			if (order == memory::NCHW)
			{
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
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, num * spatial_dim) {
					int n = index / spatial_dim;
					int s = index % spatial_dim;
					int offset = index * channels;
					float dot = 0;
					for (int c = 0; c < channels; ++c) {
						dot += (data_1[offset + c]
							* data_2[offset + c]);
					}
					channel_dot[index] = dot;
				}
			}
			else
			{
				return;
			}
		}

		template<typename Dtype>
		void operation_softmax<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				int outer_num_ = bottoms[i]->num();
				memory::orderType order_ = bottoms[i]->order();
				int inner_num_ = 0;
				if (bottoms[i]->data_shape().size() <= 2)
				{
					inner_num_ = 1;
					scale_.reset(new memory::tensor<float>(std::vector<int>{outer_num_, 1}, this->params_.device_, order_, bottoms[i]->allocator()));
				}
				else
				{
					inner_num_ = bottoms[i]->height() * bottoms[i]->width();
					scale_.reset(new memory::tensor<float>(std::vector<int>{outer_num_, 1, bottoms[i]->height(), bottoms[i]->width()}, this->params_.device_, order_, bottoms[i]->allocator()));//单通道，NHWC和NCHW无差别
				}
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), this->params_.device_, order_, bottoms[i]->allocator()));
				//
				const float* bottom_data = bottoms[i]->gpu_data();
				float* top_data = tops[i]->mutable_gpu_data();
				float* scale_data = scale_->mutable_gpu_data();
				int count = bottoms[i]->count();
				int channels = bottoms[i]->channels();
				//
				math_functions::excalibur_copy(count, bottom_data, top_data, this->params_.device_);
				kernel_channel_max << <CUDA_GET_BLOCKS(outer_num_ * inner_num_),
					CUDA_NUM_THREADS >> > (outer_num_, channels, inner_num_, top_data,
						scale_data, order_);
				// subtract
				// NOLINT_NEXT_LINE(whitespace/operators)
				kernel_channel_subtract << <CUDA_GET_BLOCKS(count),
					CUDA_NUM_THREADS >> > (count, outer_num_, channels, inner_num_,
						scale_data, top_data, order_);
				// exponentiate
				// NOLINT_NEXT_LINE(whitespace/operators)
				kernel_exp << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, top_data, top_data);
				// sum after exp
				// NOLINT_NEXT_LINE(whitespace/operators)
				kernel_channel_sum << <CUDA_GET_BLOCKS(outer_num_ * inner_num_),
					CUDA_NUM_THREADS >> > (outer_num_, channels, inner_num_, top_data,
						scale_data, order_);
				// divide
				// NOLINT_NEXT_LINE(whitespace/operators)
				kernel_channel_div << <CUDA_GET_BLOCKS(count),
					CUDA_NUM_THREADS >> > (count, outer_num_, channels, inner_num_,
						scale_data, top_data, order_);
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_softmax);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_softmax);
#endif

#endif
	}
}