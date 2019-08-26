#ifndef _CONV_ARM_HPP_
#define _CONV_ARM_HPP_

#include "../base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class conv_arm : public baseconv
		{
		public:
			conv_arm(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization = false);
			virtual ~conv_arm() { }

			void set_bias(float* bias);
			void set_weights(float *weights);

			void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) override;
		private:
			std::shared_ptr<tensor<float>> weights_transformed_;
			std::shared_ptr<tensor<float>> weights_sgemm_;

			bool use_sgemm1x1;
			bool use_winograd3x3;
			bool conv3x3s2;

			void forward_gemm(const float* input, const float* weights_packed, float* output, bool skip_im2col = false) override {}
			void forward_gemm(const signed char * input, const signed char *weights, int *output, bool skip_im2col = false) override {}
			void forward_bias(float* output, const float* bias) override {}
		};
	}
}

#endif