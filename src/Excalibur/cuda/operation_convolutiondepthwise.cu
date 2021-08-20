#include "../../../include/Excalibur/operation_convolutiondepthwise.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#endif

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
			float* const top_data, const float* const weight, const float* const bias, const bool bias_term_, glasssix::memory::orderType order)
		{
			if (order == glasssix::memory::NCHW)
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
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) 
					{
						for (int w = wstart; w < wend; ++w) 
						{
							aveval += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}
					if (bias_term_) 
					{
						aveval += bias[c];
					}
					top_data[index] = aveval;
				}
			}
			else if (order == glasssix::memory::NHWC)
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
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h) 
					{
						for (int w = wstart; w < wend; ++w) 
						{
							aveval += bottom_data[(n * height * width + h * width + w) * channels + c] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
						}
					}
					if (bias_term_) 
					{
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

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_gpu_f32(
			cublasHandle_t &cublas_handle,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<glasssix::memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<glasssix::memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->output_channel_, this->group_);
			glasssix::memory::orderType order = bottoms[0]->order();
			this->num_ = bottoms[0]->num();
			this->input_dim_h_ = bottoms[0]->height();
			this->input_dim_w_ = bottoms[0]->width();
			this->input_channel_ = bottoms[0]->channels();
			CHECK_EQ(this->input_channel_, this->output_channel_);
			this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
			this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
			const float* bottom_data = bottoms[0]->gpu_data();
			const float* weights_data = this->weights_f32_[0]->gpu_data();
			const float* bias_data = nullptr;
			if (this->bias_term_)
			{
				bias_data = this->weights_f32_[1]->gpu_data();
			}
			switch (order)
			{
			case glasssix::memory::NCHW:
				tops[0].reset(new glasssix::memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
					this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			case glasssix::memory::NHWC:
				tops[0].reset(new glasssix::memory::tensor<float>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_},
					this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
			auto top_data = tops[0]->mutable_gpu_data();
			int count = tops[0]->count(0 ,4);
			depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
				count, bottom_data, this->num_, this->input_channel_,
				this->input_dim_h_, this->input_dim_w_, this->output_dim_h_, this->output_dim_w_, this->kernel_size_h_,
				this->kernel_size_w_, this->stride_h_, this->stride_w_, this->pad_left_, this->pad_bottom_, 
				top_data, weights_data, bias_data, this->bias_term_, order);
			CUDA_POST_KERNEL_CHECK;
		}

		__global__ void depthwise_conv_kernel_f16(const int nthreads,
			const unsigned short* const bottom_data, const int num, const int channels,
			const int height, const int width, const int conved_height,
			const int conved_width, const int kernel_h, const int kernel_w,
			const int stride_h, const int stride_w, const int pad_h, const int pad_w,
			unsigned short* const top_data, const unsigned short* const weight, const unsigned short* const bias, const bool bias_term_, glasssix::memory::orderType order)
		{
			if (order == glasssix::memory::NCHW)
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
					__half aveval = __float2half(0.0f);
					const unsigned short* const bottom_slice =
						bottom_data + (n * channels + c) * height * width;
					const unsigned short* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h)
					{
						for (int w = wstart; w < wend; ++w)
						{
							aveval = __hadd(aveval, __hmul(__ushort_as_half(bottom_slice[h * width + w]), __ushort_as_half(weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)])));
						}
					}
					if (bias_term_)
					{
						aveval = __hadd(aveval, __ushort_as_half(bias[c]));
					}
					top_data[index] = __half_as_ushort(aveval);
				}
			}
			else if (order == glasssix::memory::NHWC)
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
					__half aveval = 0;
					const unsigned short* const weight_slice =
						weight + c * kernel_h * kernel_w;
					int khstart = hend < kernel_h ? kernel_h - hend : 0;
					int kwstart = wend < kernel_w ? kernel_w - wend : 0;
					for (int h = hstart; h < hend; ++h)
					{
						for (int w = wstart; w < wend; ++w)
						{
							aveval = __hadd(aveval, __hmul(__ushort_as_half(bottom_data[(n * height * width + h * width + w) * channels + c]), __ushort_as_half(weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)])));
						}
					}
					if (bias_term_)
					{
						aveval = __hadd(aveval, __ushort_as_half(bias[c]));
					}
					top_data[index] = __half_as_ushort(aveval);
				}
			}
			else
			{
				return;
			}
		}

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<glasssix::memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<glasssix::memory::tensor<unsigned short>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->output_channel_, this->group_);
			glasssix::memory::orderType order = bottoms[0]->order();
			this->num_ = bottoms[0]->num();
			this->input_dim_h_ = bottoms[0]->height();
			this->input_dim_w_ = bottoms[0]->width();
			this->input_channel_ = bottoms[0]->channels();
			CHECK_EQ(this->input_channel_, this->output_channel_);
			this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
			this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
			const unsigned short* bottom_data = bottoms[0]->gpu_data();
			const unsigned short* weights_data = this->weights_f16_[0]->gpu_data();
			const unsigned short* bias_data = nullptr;
			if (this->bias_term_)
			{
				bias_data = this->weights_f16_[1]->gpu_data();
			}
			switch (order)
			{
			case glasssix::memory::NCHW:
				tops[0].reset(new glasssix::memory::tensor<unsigned short>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
					this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			case glasssix::memory::NHWC:
				tops[0].reset(new glasssix::memory::tensor<unsigned short>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_},
					this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
			auto top_data = tops[0]->mutable_gpu_data();
			int count = tops[0]->count(0, 4);
			depthwise_conv_kernel_f16 << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
				count, bottom_data, this->num_, this->input_channel_,
				this->input_dim_h_, this->input_dim_w_, this->output_dim_h_, this->output_dim_w_, this->kernel_size_h_,
				this->kernel_size_w_, this->stride_h_, this->stride_w_, this->pad_left_, this->pad_bottom_,
				top_data, weights_data, bias_data, this->bias_term_, order);
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_convolutiondepthwise);
		INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_convolutiondepthwise);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_convolutiondepthwise);
		INSTANTIATE_OPERATION_CUDA_FWDF16(operation_convolutiondepthwise);
#endif

#endif
	}
}