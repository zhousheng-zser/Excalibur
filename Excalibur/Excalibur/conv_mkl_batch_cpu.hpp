#ifdef USE_MKL
#ifndef _CONV_MKL_BATCH_CPU_HPP_
#define _CONV_MKL_BATCH_CPU_HPP_
#include "base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class conv_mkl_batch_cpu : public baseconv
		{
		public:
			conv_mkl_batch_cpu::conv_mkl_batch_cpu(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, bool bias_term, int device) 
				: baseconv(input_Channel, output_Channel, kernelSize, stride, pad, bias_term, device) {}

			conv_mkl_batch_cpu::conv_mkl_batch_cpu(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device) 
				: baseconv(input_Channel, output_Channel, kernelSize, group, stride, pad, bias_term, device) {}

			virtual conv_mkl_batch_cpu::~conv_mkl_batch_cpu() {}

			void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) override;

			void forward_bias(float* output, const float* bias) override;

			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;
		};
	}
}
#endif // !_CONV_MKL_BATCH_CPU_HPP_
#endif //!USE_MKL