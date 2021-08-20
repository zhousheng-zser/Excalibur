#include "../../../include/Excalibur/operation_convolution.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/im2col.hpp"

#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#endif

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
				if (this->group_ == 1)
				{
					im2col_gpu(input, this->input_channel_, this->input_dim_h_, this->input_dim_w_, this->kernel_size_h_,
						this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_, 
						this->dilation_h_, this->dilation_w_, col_buffer_data, order);
					col_buff = col_buffer_data;
					math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_channel_,
						this->output_spatial_dim_, this->kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < this->group_; ++g)
					{
						int gistep = this->input_channel_ / this->group_;
						int gostep = this->output_channel_ / this->group_;
						im2col_gpu(input + this->input_dim_h_ * this->input_dim_w_ * gistep * g, gistep, this->input_dim_h_, this->input_dim_w_, this->kernel_size_h_,
							this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_, this->dilation_h_, this->dilation_w_, col_buffer_data, order);
						col_buff = col_buffer_data;
						math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, gostep,
							this->output_spatial_dim_, this->kernel_size_h_ * this->kernel_size_w_ * gistep, 1.0f,
							weights + this->kernel_size_h_ * this->kernel_size_w_ * g * gostep,
							col_buff, 0.0f, output + this->output_spatial_dim_ * g * gostep);
					}
				}
			}
			else if (order == memory::NHWC)
			{
				if (this->group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle, CblasTrans, CblasTrans, this->output_spatial_dim_, this->output_channel_,
						this->kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					NOT_IMPLEMENTED;
					for (int g = 0; g < this->group_; ++g)
					{
						math_functions::gpu_sgemm(cublas_handle, CblasTrans, CblasTrans, this->output_spatial_dim_, this->output_channel_ / this->group_,
							this->kernel_size_h_ * this->kernel_size_w_, 1.0f,
							col_buff + this->output_spatial_dim_ * this->kernel_size_h_ * this->kernel_size_w_ * g,
							weights + this->kernel_size_h_ * this->kernel_size_w_ * g, 0.0f,
							output + this->output_spatial_dim_ * g);
					}

					std::shared_ptr<memory::tensor<float>> temp;
					temp.reset(new memory::tensor<float>(std::vector<int>{this->output_spatial_dim_ * this->output_channel_}));
					float* temp_data = temp->mutable_gpu_data();

					for (int ch = 0; ch < this->output_channel_; ++ch)
					{
						int channel_offset = ch * this->output_dim_h_ * this->output_dim_w_;
						for (int row = 0; row < this->output_dim_h_; ++row)
						{
							int row_offset = row * this->output_dim_w_;
							for (int col = 0; col < this->output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * this->output_channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, this->output_spatial_dim_ * this->output_channel_ * sizeof(float));
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
			if (order == memory::NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_channel_,
					this->output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order == memory::NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_spatial_dim_,
					this->output_channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
		}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			memory::orderType order = bottoms[0]->order();
			this->num_ = bottoms[0]->num();
			const float* bottom_data = bottoms[0]->gpu_data();
			int bottom_dim = bottoms[0]->count(1, 4);
			const float* weights_data = this->weights_f32_[0]->gpu_data();
			const float* bias_data = nullptr;
			if (this->bias_term_)
				bias_data = this->weights_f32_[1]->gpu_data();

			this->input_channel_ = bottoms[0]->channels();
			this->input_dim_h_ = bottoms[0]->height();
			this->input_dim_w_ = bottoms[0]->width();
			this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
			this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
			this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
			this->kernel_dim_ = this->input_channel_ * this->kernel_size_h_ * this->kernel_size_w_;

			if (order == memory::NCHW)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, bottoms[0]->allocator()));
			}
			else if (order == memory::NHWC)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_}, this->params_.device_, order, bottoms[0]->allocator()));
			}
			else
			{
				LOG(FATAL) << "Un-supported data arrange.";
			}

			col_buffer_.reset(new memory::tensor<float>(std::vector<int>{this->kernel_dim_ / this->group_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, memory::NCHW, bottoms[0]->allocator()));
			col_buffer_data = col_buffer_->mutable_gpu_data();

			float* top_data = tops[0]->mutable_gpu_data();
			int top_dim = tops[0]->count(1, 4);

			bias_multiplier_.reset(new memory::tensor<float>(this->output_spatial_dim_, this->params_.device_, order, bottoms[0]->allocator()));
			bias_multiplier_data = bias_multiplier_->mutable_gpu_data();

			math_functions::gpu_set(this->output_spatial_dim_, 1.0f, bias_multiplier_data);
			for (int n = 0; n < this->num_; n++)
			{
				forward_gpu_sgemm(cublas_handle_, bottom_data + n * bottom_dim, weights_data, top_data + n * top_dim, order);
				if (this->bias_term_)
					forward_gpu_sbias(cublas_handle_, top_data + n * top_dim, bias_data, order);
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_hbias(cublasHandle_t& cublas_handle,
			unsigned short* output, const unsigned short* bias, memory::orderType order)
		{
			if (order == memory::NCHW)
			{
				math_functions::gpu_gemmEx(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_channel_,
					this->output_spatial_dim_, 1, float32_to_float16(1.0f), bias, bias_multiplier_f16_data_, float32_to_float16(1.0f), output);
			}
			else if (order == memory::NHWC)
			{
				math_functions::gpu_gemmEx(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_spatial_dim_,
					this->output_channel_, 1, float32_to_float16(1.0f), bias_multiplier_f16_data_, bias, float32_to_float16(1.0f), output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_hgemm(cublasHandle_t& cublas_handle,
			const unsigned short* input, const unsigned short* weights, unsigned short* output, memory::orderType order)
		{
			const unsigned short* col_buff = input;

			if (order == memory::NCHW)
			{
				if (this->group_ == 1)
				{
					im2col_gpu(input, this->input_channel_, this->input_dim_h_, this->input_dim_w_, this->kernel_size_h_,
						this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_,
						this->dilation_h_, this->dilation_w_, col_buffer_f16_data_, order);
					col_buff = col_buffer_f16_data_;
					math_functions::gpu_gemmEx(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_channel_,
						this->output_spatial_dim_, this->kernel_dim_, float32_to_float16(1.0f),
						weights, col_buff, float32_to_float16(0.0f), output);
				}
				else
				{
					for (int g = 0; g < this->group_; ++g)
					{
						int gistep = this->input_channel_ / this->group_;
						int gostep = this->output_channel_ / this->group_;
						im2col_gpu(input + this->input_dim_h_ * this->input_dim_w_ * gistep * g, gistep, this->input_dim_h_, this->input_dim_w_, this->kernel_size_h_,
							this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_, this->dilation_h_, this->dilation_w_, col_buffer_f16_data_, order);
						col_buff = col_buffer_f16_data_;
						math_functions::gpu_gemmEx(cublas_handle, CblasNoTrans, CblasNoTrans, gostep,
							this->output_spatial_dim_, this->kernel_size_h_ * this->kernel_size_w_ * gistep, float32_to_float16(1.0f),
							weights + this->kernel_size_h_ * this->kernel_size_w_ * g * gostep,
							col_buff, float32_to_float16(0.0f), output + this->output_spatial_dim_ * g * gostep);
					}
				}
			}
			else if (order == memory::NHWC)
			{
				if (this->group_ == 1)
				{
					math_functions::gpu_gemmEx(cublas_handle, CblasTrans, CblasTrans, this->output_spatial_dim_, this->output_channel_,
						this->kernel_dim_, float32_to_float16(1.0f),
						col_buff, weights, float32_to_float16(0.0f), output);
				}
				else
				{
					NOT_IMPLEMENTED;
					for (int g = 0; g < this->group_; ++g)
					{
						math_functions::gpu_gemmEx(cublas_handle, CblasTrans, CblasTrans, this->output_spatial_dim_, this->output_channel_ / this->group_,
							this->kernel_size_h_ * this->kernel_size_w_, float32_to_float16(1.0f),
							col_buff + this->output_spatial_dim_ * this->kernel_size_h_ * this->kernel_size_w_ * g,
							weights + this->kernel_size_h_ * this->kernel_size_w_ * g, float32_to_float16(0.0f),
							output + this->output_spatial_dim_ * g);
					}

					std::shared_ptr<memory::tensor<float>> temp;
					temp.reset(new memory::tensor<float>(std::vector<int>{this->output_spatial_dim_* this->output_channel_}));
					float* temp_data = temp->mutable_gpu_data();

					for (int ch = 0; ch < this->output_channel_; ++ch)
					{
						int channel_offset = ch * this->output_dim_h_ * this->output_dim_w_;
						for (int row = 0; row < this->output_dim_h_; ++row)
						{
							int row_offset = row * this->output_dim_w_;
							for (int col = 0; col < this->output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * this->output_channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, this->output_spatial_dim_ * this->output_channel_ * sizeof(float));
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			memory::orderType order = bottoms[0]->order();
			this->num_ = bottoms[0]->num();
			const unsigned short* bottom_data = bottoms[0]->gpu_data();
			int bottom_dim = bottoms[0]->count(1, 4);
			const unsigned short* weights_data = this->weights_f16_[0]->gpu_data();
			const unsigned short* bias_data = nullptr;
			if (this->bias_term_)
				bias_data = this->weights_f16_[1]->gpu_data();

			this->input_channel_ = bottoms[0]->channels();
			this->input_dim_h_ = bottoms[0]->height();
			this->input_dim_w_ = bottoms[0]->width();
			this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
			this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
			this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
			this->kernel_dim_ = this->input_channel_ * this->kernel_size_h_ * this->kernel_size_w_;

			if (order == memory::NCHW)
			{
				tops[0].reset(new memory::tensor<unsigned short>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, bottoms[0]->allocator()));
			}
			else if (order == memory::NHWC)
			{
				tops[0].reset(new memory::tensor<unsigned short>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_}, this->params_.device_, order, bottoms[0]->allocator()));
			}
			else
			{
				LOG(FATAL) << "Un-supported data arrange.";
			}

			col_buffer_f16_.reset(new memory::tensor<unsigned short>(std::vector<int>{this->kernel_dim_ / this->group_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, memory::NCHW, bottoms[0]->allocator()));
			col_buffer_f16_data_ = col_buffer_f16_->mutable_gpu_data();

			unsigned short* top_data = tops[0]->mutable_gpu_data();
			int top_dim = tops[0]->count(1, 4);

			bias_multiplier_f16_.reset(new memory::tensor<unsigned short>(this->output_spatial_dim_, this->params_.device_, order, bottoms[0]->allocator()));
			bias_multiplier_f16_data_ = bias_multiplier_f16_->mutable_gpu_data();

			math_functions::gpu_set(this->output_spatial_dim_, float32_to_float16(1.0f), bias_multiplier_f16_data_);
			for (int n = 0; n < this->num_; n++)
			{
				forward_gpu_hgemm(cublas_handle_, bottom_data + n * bottom_dim, weights_data, top_data + n * top_dim, order);
				if (this->bias_term_)
					forward_gpu_hbias(cublas_handle_, top_data + n * top_dim, bias_data, order);
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_convolution);
		INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_convolution);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_convolution);
		INSTANTIATE_OPERATION_CUDA_FWDF16(operation_convolution);
#endif

#endif //!USE_CUDA
	}
}