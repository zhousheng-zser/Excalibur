
#ifndef _DECONV_HPP_
#define _DECONV_HPP_
#include "base_conv.hpp"
#include "Primitives/tensor.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"
#include <memory>
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif

namespace glasssix
{
	namespace excalibur
	{
		class deconv : public baseconv
		{
		public:

			int kernel_length_;

			deconv() {}

			deconv(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device)
				: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device) 
			{
				kernel_length_ = kernelSize * kernelSize;
			}

			virtual ~deconv() {}

			void Forward(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override;

#ifdef USE_CUDA
			void Forward(cublasHandle_t &cublas_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override;
#ifdef USE_CUDNN
			void Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override;
#endif // USE_CUDNN
#endif // !USE_CUDA

		private:
			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;
			void forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) override;
			void forward_bias(float* output, const float* bias) override;

#ifdef USE_CUDA
			void forward_gemm(cublasHandle_t &cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) override;
			void forward_gemm(cublasHandle_t &cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) override;
			void forward_bias(cublasHandle_t &cublas_handle_, float* output, const float* bias) override;
#endif // !USE_CUDA

		};
	}
}
#endif //_DECONV_HPP_