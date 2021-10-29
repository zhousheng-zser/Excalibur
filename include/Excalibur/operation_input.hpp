#pragma once
#ifndef _OPERATION_INPUT_HPP_
#define _OPERATION_INPUT_HPP_
#include "operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_input : public operation<Dtype>
		{
		public:
			explicit operation_input(const operation_param& param);

			virtual const char* type() const { return this->params_.type_.c_str(); }

			virtual ~operation_input() {}

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_gpu_f32(
#ifdef USE_CUDA
				cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);


#ifdef USE_CUDA
			virtual void forward_gpu_f16(
				cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
				const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops);
#endif //!USE_CUDA

		private:
			int w_ = 0;
			int h_ = 0;
			int c_ = 0;
			std::vector<float> vars_;
			std::vector<float> means_;
		};
	}
}
#endif // !_OPERATION_INPUT_HPP_

