#ifdef USE_CUDA
#include "inner_product.hpp"

namespace glasssix
{
	namespace excalibur
	{
		void inner_product::Forward_native_gpu(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			M_ = bottom->num();
			if (bias_term_)
			{
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{M_}, device_));
				math_functions::gpu_set(M_, 1.0f, bias_multiplier_->mutable_gpu_data());
			}
			top.reset(new tensor<float>(std::vector<int>{M_, N_}, device_));
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
	}
}

#endif