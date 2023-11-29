#pragma once
#ifndef _OPERATION_SHUFFLECHANNEL_HPP_
#define _OPERATION_SHUFFLECHANNEL_HPP_
#include "operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_shufflechannel : public operation<Dtype>
		{
		public:
			explicit operation_shufflechannel(const operation_param& param);

			virtual const char* type() const { return this->params_.type_.c_str(); }

			virtual ~operation_shufflechannel() {}


		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

#ifdef USE_CUDA
			virtual void forward_gpu_f32(
				cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
				const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);
#endif //!USE_CUDA

		private:
			int group_ = 1;
			int reverse_ = 0;
		};
	}
}
#endif // !_OPERATION_SHUFFLECHANNEL_HPP_

