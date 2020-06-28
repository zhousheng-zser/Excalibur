#include "../../include/Excalibur/operation_convolution.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_sgemm(cublasHandle_t &cublas_handle,
			const float* input, const float* weights, float* output, memory::orderType order)
		{
			const float* col_buff = input;

			if (order == memory::NCHW)
			{
				if (group_ == 1)
				{
					im2col_gpu(input, input_channel_, input_dim_h_, input_dim_w_, kernel_size_h_,
						kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, col_buffer_data, order);
					col_buff = col_buffer_data;
					math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, output_channel_,
						output_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						int gistep = input_channel_ / group_;
						int gostep = output_channel_ / group_;
						im2col_gpu(input + input_dim_h_ * input_dim_w_ * gistep * g, gistep, input_dim_h_, input_dim_w_, kernel_size_h_,
							kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, col_buffer_data, order);
						col_buff = col_buffer_data;
						math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, gostep,
							output_spatial_dim_, kernel_size_h_ * kernel_size_w_ * gistep, 1.0f,
							weights + kernel_size_h_ * kernel_size_w_ * g * gostep,
							col_buff, 0.0f, output + output_spatial_dim_ * g * gostep);
					}
				}
			}
			else if (order == memory::NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle, CblasTrans, CblasTrans, output_spatial_dim_, output_channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					NOT_IMPLEMENTED;
					for (int g = 0; g < group_; ++g)
					{
						math_functions::gpu_sgemm(cublas_handle, CblasTrans, CblasTrans, output_spatial_dim_, output_channel_ / group_,
							kernel_size_h_ * kernel_size_w_, 1.0f,
							col_buff + output_spatial_dim_ * kernel_size_h_ * kernel_size_w_ * g,
							weights + kernel_size_h_ * kernel_size_w_ * g, 0.0f,
							output + output_spatial_dim_ * g);
					}

					std::shared_ptr<memory::tensor<float>> temp;
					temp.reset(new memory::tensor<float>(std::vector<int>{output_spatial_dim_ * output_channel_}));
					float* temp_data = temp->mutable_gpu_data();

					for (int ch = 0; ch < output_channel_; ++ch)
					{
						int channel_offset = ch * output_dim_h_ * output_dim_w_;
						for (int row = 0; row < output_dim_h_; ++row)
						{
							int row_offset = row * output_dim_w_;
							for (int col = 0; col < output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * output_channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, output_spatial_dim_ * output_channel_ * sizeof(float));
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_sbias(cublasHandle_t &cublas_handle,
			float* output, const float* bias, memory::orderType order)
		{

		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_f32(
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_convolution);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_convolution);
#endif

#endif //!USE_CUDA
	}
}