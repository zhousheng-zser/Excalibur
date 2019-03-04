#ifdef USE_CUDA
#ifndef _CONV_DEPTHWISE_NATIVE_GPU_HPP_
#define _CONV_DEPTHWISE_NATIVE_GPU_HPP_
#include "base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class conv_depthwise_native_gpu : public baseconv
		{
		public:
			cublasHandle_t cublas_handle_;

			conv_depthwise_native_gpu::conv_depthwise_native_gpu(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device) 
				: baseconv(input_Channel, output_Channel, kernelSize, group, stride, pad, bias_term, device) {}

			virtual conv_depthwise_native_gpu::~conv_depthwise_native_gpu() {}

			void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) override;

			void forward_bias(float* output, const float* bias) override;

			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;
		};
	}
}

#endif // !_CONV_DEPTHWISE_NATIVE_GPU_HPP_
#endif // !USE_CUDA