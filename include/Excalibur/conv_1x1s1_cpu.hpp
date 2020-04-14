#ifndef _CONV_1X1S1_CPU_HPP_
#define _CONV_1X1S1_CPU_HPP_
#include "base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class conv_1x1s1_cpu : public baseconv
		{
		public:

			conv_1x1s1_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device = -1, bool int8_quantization = false);

			virtual ~conv_1x1s1_cpu() {};

			void Forward(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override;

		private:

			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;

			void forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) override;

			void forward_bias(float* output, const float* bias) override;

#ifdef USE_CUDA
			void Forward(cublasHandle_t &cublas_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) {}
			void forward_gemm(cublasHandle_t &cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) {}
			void forward_gemm(cublasHandle_t &cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) {}
			void forward_bias(cublasHandle_t &cublas_handle_, float* output, const float* bias) {}
#ifdef USE_CUDNN
			void Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) {}
#endif //!USE_CUDNN
#endif // !USE_CUDA
		};
	}
}


#endif // !_CONV_1X1S1_CPU_HPP_