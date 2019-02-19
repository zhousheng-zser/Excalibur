#ifdef USE_CUDA
#include "inner_product.hpp"

namespace glasssix
{
	namespace excalibur
	{
		void inner_product::Forward_native_gpu(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			M_ = bottom->num();
			order_ = bottom->order();
			if (bias_term_)
			{
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{M_}, device_));
				math_functions::gpu_set(M_, 1.0f, bias_multiplier_->mutable_gpu_data());
			}

			if (order_ == NCHW)
			{
				top.reset(new tensor<float>(std::vector<int>{M_, N_, 1, 1}, device_, order_));
				//
				const float* bottom_data = bottom->gpu_data();
				float* top_data = top->mutable_gpu_data();
				const float* weight = weights_->gpu_data();
				//
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, M_, N_, K_, 1.0f,
					bottom_data, weight, 0.0f, top_data);
				if (bias_term_)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, M_, N_, 1,
						1.0f, bias_multiplier_->gpu_data(), bias_->gpu_data(), 1.0f, top_data);
				}
			}
			else if (order_ == NHWC)
			{
				top.reset(new tensor<float>(std::vector<int>{M_, 1, 1, N_}, device_, order_));
				//
				const float* bottom_data = bottom->gpu_data();
				float* top_data = top->mutable_gpu_data();
				const float* weight = weights_->gpu_data();

				if (bottom->height() != 1 || bottom->width() != 1)
				{
					std::shared_ptr<tensor<float>> col_buff;
					col_buff.reset(new tensor<float>(std::vector<int>{bottom->num(), bottom->height(), bottom->width(), bottom->channels()}, device_, order_));
					float* col_buff_data = col_buff->mutable_gpu_data();

					im2col_gpu<float>(bottom_data, bottom->channels(), bottom->height(), bottom->width(), 1,
						1, 0, 0, 1, 1, 1, 1, col_buff_data, order_, bottom->num());
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, M_, N_, K_, 1.0f,
						col_buff_data, weight, 0.0f, top_data);
				}
				else
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, M_, N_, K_, 1.0f,
						bottom_data, weight, 0.0f, top_data);
				}
				//

				if (bias_term_)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, M_, N_, 1,
						1.0f, bias_multiplier_->gpu_data(), bias_->gpu_data(), 1.0f, top_data);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}

#endif