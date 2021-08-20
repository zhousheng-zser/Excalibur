#include "../../../include/Excalibur/operation_deconvolution.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include "../../../include/Excalibur/im2col.hpp"

namespace glasssix
{
	namespace excalibur
	{

#ifdef USE_CUDA
		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			this->num_ = bottoms[0]->num();
			this->bottom_dim_ = bottoms[0]->count(1, 4);
			const float* bottom_data = bottoms[0]->gpu_data();
			const float* weights = this->weights_f32_[0]->gpu_data();
			const float* bias = nullptr;

			memory::orderType order = bottoms[0]->order();

			if (order == memory::NCHW)
			{
				this->input_channel_ = bottoms[0]->channels();
				this->input_dim_h_ = bottoms[0]->height();
				this->input_dim_w_ = bottoms[0]->width();
				this->input_spatial_dim_ = this->input_dim_h_ * this->input_dim_w_;
				this->output_dim_h_ = (this->input_dim_h_ - 1) * this->stride_h_ + this->kernel_size_h_ - (this->pad_top_ + this->pad_bottom_);
				this->output_dim_w_ = (this->input_dim_w_ - 1) * this->stride_w_ + this->kernel_size_w_ - (this->pad_left_ + this->pad_right_);
				this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
				tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
					this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
				float* top_data = tops[0]->mutable_gpu_data();
				this->top_dim_ = tops[0]->count(1, 4);

				col_buffer_.reset(new memory::tensor<float>(std::vector<int>{1, this->output_channel_* this->kernel_size_h_* this->kernel_size_w_, this->input_dim_h_, this->input_dim_w_},
					this->params_.device_, order, nullptr));
				this->col_offset_ = this->input_spatial_dim_ * this->kernel_size_h_ * this->kernel_size_w_;
				bias_multiplier_.reset(new memory::tensor<float>(this->output_spatial_dim_, this->params_.device_, order, nullptr));
				this->output_offset_ = this->output_channel_ * this->output_spatial_dim_ / this->group_;

				if (this->bias_term_)
				{
					bias_multiplier_.reset(new memory::tensor<float>(this->output_spatial_dim_, this->params_.device_, order, nullptr));
					math_functions::gpu_set(this->output_dim_w_ * this->output_dim_h_, 1.0f, bias_multiplier_->mutable_gpu_data());
					bias = this->weights_f32_[1]->gpu_data();
				}

				for (int n = 0; n < this->num_; n++)
				{
					forward_sgemm(cublas_handle_, bottom_data + n * this->bottom_dim_, weights, top_data + n * this->top_dim_, order);
					if (this->bias_term_)
					{
						forward_sbias(cublas_handle_, top_data + n * this->top_dim_, bias, order);
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_sgemm(cublasHandle_t& cublas_handle_, const float* input, const float* weights, float* output, memory::orderType order)
		{
			col_buffer_data = col_buffer_->mutable_gpu_data();

			if (order == memory::NCHW)
			{
				for (int g = 0; g < this->group_; g++)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasTrans, CblasNoTrans, this->output_channel_ * this->kernel_size_h_ * this->kernel_size_w_ / this->group_,
						this->input_spatial_dim_, this->input_channel_ / this->group_, 1.0f,
						weights + this->kernel_size_h_ * this->kernel_size_w_ * g,
						input + g * this->input_channel_ * this->input_spatial_dim_ / this->group_,
						0.0f, col_buffer_data + this->col_offset_ * g);
				}

				col2im_gpu(col_buffer_data, this->output_channel_, this->output_dim_h_, this->output_dim_w_, this->kernel_size_h_,
					this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_, this->dilation_h_, this->dilation_w_, output);
			}
			else if (order == memory::NHWC)
			{
				NOT_IMPLEMENTED;
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_sbias(cublasHandle_t& cublas_handle_, float* output, const float* bias, memory::orderType order)
		{
			if (order == memory::NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, this->output_channel_,
					this->output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order == memory::NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, this->output_spatial_dim_,
					this->output_channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_deconvolution);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_deconvolution);
#endif

#endif
	}
}
