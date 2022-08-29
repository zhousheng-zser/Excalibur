#include "../../../include/Excalibur/operation_pooling.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#include <cfloat>

#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#endif

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
		__global__ void MaxPoolForward(const int nthreads,
			const float* const bottom_data, const int num, const int channels,
			const int height, const int width, const int pooled_height,
			const int pooled_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			float* const top_data, memory::orderType order) {
 
			if (order == memory::NCHW)
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
			else if (order == memory::NHWC)
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
			float* const top_data, memory::orderType order) {

			if (order == memory::NCHW)
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
			else if (order == memory::NHWC)
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

		template<typename Dtype>
		void operation_pooling<Dtype>::forward_gpu_f32(
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
				int num = bottoms[i]->num();
				int channels_ = bottoms[i]->channels();
				int height_ = bottoms[i]->height();
				int width_ = bottoms[i]->width();
				memory::orderType order_ = bottoms[i]->order();

				if (this->global_pooling_)
				{
					this->kernel_size_w_ = width_;
					this->kernel_size_h_ = height_;
					this->pad_top_ = 0;
					this->pad_bottom_ = 0;
					this->pad_left_ = 0;
					this->pad_right_ = 0;
					this->stride_h_ = 1;
					this->stride_w_ = 1;
				}

				int pooled_height_ = 0, pooled_width_ = 0;
				if (pad_mode_ == 0)
				{
					pooled_height_ = static_cast<int>(ceil(static_cast<float>(
						height_ + pad_top_ + pad_bottom_ - kernel_size_h_) /
						stride_h_)) + 1;
					pooled_width_ = static_cast<int>(ceil(static_cast<float>(
						width_ + pad_left_ + pad_right_ - kernel_size_w_) /
						stride_w_)) + 1;
				}
				else if (pad_mode_ == 1)
				{
					pooled_height_ = static_cast<int>(floor(static_cast<float>(
						height_ + pad_top_ + pad_bottom_ - kernel_size_h_) /
						stride_h_)) + 1;
					pooled_width_ = static_cast<int>(floor(static_cast<float>(
						width_ + pad_left_ + pad_right_ - kernel_size_w_) /
						stride_w_)) + 1;
				}
				else
					NOT_IMPLEMENTED;

				int spatial_dim;

				if (order_ == memory::NCHW)
				{
					tops[i].reset(new memory::tensor<float>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
					spatial_dim = bottoms[i]->count(2, 4);
				}
				else if (order_ == memory::NHWC)
				{
					tops[i].reset(new memory::tensor<float>(std::vector<int>{num, pooled_height_, pooled_width_, channels_}, this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
					spatial_dim = bottoms[i]->count(1, 3);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				//
				const float* bottom_data = bottoms[i]->gpu_data();
				float* top_data = tops[i]->mutable_gpu_data();
				const int top_count = tops[i]->count(0, 4);
				switch (type_)
				{
				case MAX:
					// NOLINT_NEXT_LINE(whitespace/operators)
					MaxPoolForward << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (
						top_count, bottom_data, num, channels_,
						height_, width_, pooled_height_, pooled_width_, kernel_size_h_,
						kernel_size_w_, stride_h_, stride_w_, pad_top_, pad_left_, top_data, order_);
					break;
				case AVE:
					if (order_ == memory::NCHW && this->global_pooling_)
					{
						GlobalAvePoolForward << <num * channels_, CUDA_NUM_THREADS >> > (
							spatial_dim, bottom_data, top_data);
					}
					else
					{
						AvePoolForward << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (
							top_count, bottom_data, num, channels_,
							height_, width_, pooled_height_, pooled_width_, kernel_size_h_,
							kernel_size_w_, stride_h_, stride_w_, pad_top_, pad_left_, top_data, order_);
					}

					break;
				default:
					LOG(FATAL) << "Unknown pooling method.";
				}
			}
			CUDA_POST_KERNEL_CHECK;
		}

		__global__ void MaxPoolForward_f16(const int nthreads,
			const unsigned short* const bottom_data, const int num, const int channels,
			const int height, const int width, const int pooled_height,
			const int pooled_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			unsigned short* const top_data, memory::orderType order) {

			if (order == memory::NCHW)
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
					__half maxval = __float2half(-FLT_MAX);
					const unsigned short* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							__half val = __ushort_as_half(bottom_slice[h * width + w]);
							if (__hgt(val, maxval)) {
								maxval = val;
							}
						}
					}
					top_data[index] = __half_as_ushort(maxval);
				}
			}
			else if (order == memory::NHWC)
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
					__half maxval = __float2half(-FLT_MAX);
					const unsigned short* const bottom_slice =
						bottom_data + n * height * width * channels;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							__half val = __ushort_as_half(bottom_slice[(h * width + w) * channels + c]);
							if (__hgt(val, maxval)) {
								maxval = val;
							}
						}
					}
					top_data[index] = __half_as_ushort(maxval);
				}
			}
			else
			{
				return;
			}
		}

		__global__ void AvePoolForward_f16(const int nthreads,
			const unsigned short* const bottom_data, const int num, const int channels,
			const int height, const int width, const int pooled_height,
			const int pooled_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			unsigned short* const top_data, memory::orderType order) {

			if (order == memory::NCHW)
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
					__half aveval = __float2half(0.f);
					const unsigned short* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval = __hadd(aveval, __ushort_as_half(bottom_slice[h * width + w]));
						}
					}
					top_data[index] = __half_as_ushort(__hdiv(aveval, __int2half_rn(pool_size)));
				}
			}
			else if (order == memory::NHWC)
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
					__half aveval = __float2half(0.f);
					const unsigned short* const bottom_slice =
						bottom_data + n * height * width * channels;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							aveval = __hadd(aveval, __ushort_as_half(bottom_slice[(h * width + w) * channels + c]));
						}
					}
					top_data[index] = __half_as_ushort(__hdiv(aveval, __int2half_rn(pool_size)));
				}
			}
			else
			{
				return;
			}
		}

		__global__ void GlobalAvePoolForward_f16(const int spatial_dim,
			const unsigned short* bottom_data, unsigned short* top_data) {
			__shared__ __half buffer[CUDA_NUM_THREADS];
			unsigned int tid = threadIdx.x;
			buffer[tid] = __float2half(0.0f);
			__syncthreads();

			for (int j = tid; j < spatial_dim; j += blockDim.x) {
				buffer[tid] = __hadd(buffer[tid], __ushort_as_half(bottom_data[blockIdx.x * spatial_dim + j]));
			}
			__syncthreads();

			for (int i = blockDim.x / 2; i > 0; i >>= 1) {
				if (tid < i) {
					buffer[threadIdx.x] = __hadd(buffer[threadIdx.x], buffer[threadIdx.x + i]);
				}
				__syncthreads();
			}

			if (tid == 0) {
				top_data[blockIdx.x] = __half_as_ushort(__hdiv(buffer[0], __int2half_rn(spatial_dim)));
			}
		}

		template<typename Dtype>
		void operation_pooling<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				int num = bottoms[i]->num();
				int channels_ = bottoms[i]->channels();
				int height_ = bottoms[i]->height();
				int width_ = bottoms[i]->width();
				memory::orderType order_ = bottoms[i]->order();

				if (this->global_pooling_)
				{
					this->kernel_size_w_ = width_;
					this->kernel_size_h_ = height_;
					this->pad_top_ = 0;
					this->pad_bottom_ = 0;
					this->pad_left_ = 0;
					this->pad_right_ = 0;
					this->stride_h_ = 1;
					this->stride_w_ = 1;
				}

				int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
					height_ + pad_top_ + pad_bottom_ - kernel_size_h_) / stride_h_)) + 1;
				int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
					width_ + pad_left_ + pad_right_ - kernel_size_w_) / stride_w_)) + 1;

				int spatial_dim;

				if (order_ == memory::NCHW)
				{
					tops[i].reset(new memory::tensor<unsigned short>(std::vector<int>{num, channels_, pooled_height_, pooled_width_}, this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
					spatial_dim = bottoms[i]->count(2, 4);
				}
				else if (order_ == memory::NHWC)
				{
					tops[i].reset(new memory::tensor<unsigned short>(std::vector<int>{num, pooled_height_, pooled_width_, channels_}, this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
					spatial_dim = bottoms[i]->count(1, 3);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				//
				const unsigned short* bottom_data = bottoms[i]->gpu_data();
				unsigned short* top_data = tops[i]->mutable_gpu_data();
				const int top_count = tops[i]->count(0, 4);
				switch (type_)
				{
				case MAX:
					// NOLINT_NEXT_LINE(whitespace/operators)
					MaxPoolForward_f16 << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (
						top_count, bottom_data, num, channels_,
						height_, width_, pooled_height_, pooled_width_, kernel_size_h_,
						kernel_size_w_, stride_h_, stride_w_, pad_top_, pad_left_, top_data, order_);
					break;
				case AVE:
					if (order_ == memory::NCHW && pooled_height_ == 1 && pooled_width_ == 1)
					{
						GlobalAvePoolForward_f16 << <num * channels_, CUDA_NUM_THREADS >> > (
							spatial_dim, bottom_data, top_data);
					}
					else
					{
						AvePoolForward_f16 << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (
							top_count, bottom_data, num, channels_,
							height_, width_, pooled_height_, pooled_width_, kernel_size_h_,
							kernel_size_w_, stride_h_, stride_w_, pad_top_, pad_left_, top_data, order_);
					}

					break;
				default:
					LOG(FATAL) << "Unknown pooling method.";
				}
			}
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_pooling);
		INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_pooling);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_pooling);
		INSTANTIATE_OPERATION_CUDA_FWDF16(operation_pooling);
#endif

#endif
	}
}