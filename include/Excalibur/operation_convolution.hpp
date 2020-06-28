#pragma once
#ifndef _OPERATION_CONVOLUTION_HPP_
#define _OPERATION_CONVOLUTION_HPP_

#include "operation_general_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_convolution : public operation_general_conv<Dtype>
		{
		public:
			operation_convolution(const operation_param& param);

#ifdef HARDCODE
			virtual void init_weights() {}
#else
			virtual int init_weights(FILE *fp);
#endif //!HARDCODE

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

		private:
			void forward_cpu_sgemm(const float* input, const float* weights, float* output, memory::orderType order /*= memory::NCHW*/);

			void forward_cpu_sbias(float* output, const float* bias, memory::orderType order/* = memory::NCHW*/);

#ifdef USE_CUDA
			void forward_gpu_sgemm(cublasHandle_t &cublas_handle, const float* input, const float* weights, float* output, memory::orderType order);

			void forward_gpu_sbias(cublasHandle_t &cublas_handle, float* output, const float* bias, memory::orderType order);
#endif //!USE_CUDA

			void forward_cpu_k1s1_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);

			std::shared_ptr<memory::tensor<signed char>> weights1x1_int8_;
			std::shared_ptr<memory::tensor<signed char>> col_buffer_int8_;
			std::shared_ptr<memory::tensor<signed char>> bottom_int8_;
			std::shared_ptr<memory::tensor<float>> weights1x1_;
			std::shared_ptr<memory::tensor<float>> col_buffer_;
			std::shared_ptr<memory::tensor<float>> bias_multiplier_;
			float* col_buffer_data;
			float* bias_multiplier_data;

			enum CONV_FWD_ALG { IMPLICIT_GEMM, 
				IMPLICIT_PRECOMP_GEMM,
				GEMM,
				DIRECT,
				WINOGRAD,
				WINOGRAD_NONFUSED
			};
		};
	}
}
#endif // !_OPERATION_CONVOLUTION_HPP_
