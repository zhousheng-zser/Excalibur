#include "../../../include/Excalibur/operation_convolutiondepthwise.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

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
				const int bottom_offset_n = num * channels * height * width;
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
						bottom_data + bottom_offset_n + (n * channels + c) * height * width;
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
					top_data[bottom_offset_n + index] = aveval;
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
			CHECK_EQ(output_channel_, group_);
			glasssix::memory::orderType order = bottoms[0]->order();
			num_ = bottoms[0]->num();
			input_dim_h_ = bottoms[0]->height();
			input_dim_w_ = bottoms[0]->width();
			input_channel_ = bottoms[0]->channels();
			CHECK_EQ(input_channel_, output_channel_);
			output_dim_h_ = (input_dim_h_ + pad_bottom_ + pad_top_ - kernel_size_h_) / stride_h_ + 1;
			output_dim_w_ = (input_dim_w_ + pad_left_ + pad_right_ - kernel_size_w_) / stride_w_ + 1;
			const float* bottom_data = bottoms[0]->gpu_data();
			const float* weights_data = weights_f32_[0]->gpu_data();
			const float* bias_data = nullptr;
			if (bias_term_)
			{
				bias_data = weights_f32_[1]->gpu_data();
			}
			switch (bottoms[0]->order())
			{
			case glasssix::memory::NCHW:
				tops[0].reset(new glasssix::memory::tensor<float>(std::vector<int>{num_, output_channel_, output_dim_h_, output_dim_w_},
					bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			case glasssix::memory::NHWC:
				tops[0].reset(new glasssix::memory::tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_channel_},
					bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
			auto top_data = tops[0]->mutable_gpu_data();
			int count = tops[0]->count(1 ,4);
			for (size_t n = 0; n < num_; n++)
			{
				depthwise_conv_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, n, input_channel_,
					input_dim_h_, input_dim_w_, output_dim_h_, output_dim_w_, kernel_size_h_,
					kernel_size_w_, stride_h_, stride_w_, pad_left_, pad_bottom_, top_data, weights_data, bias_data, bias_term_, bottoms[0]->order());
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_convolutiondepthwise);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_convolutiondepthwise);
#endif

#endif
	}
}