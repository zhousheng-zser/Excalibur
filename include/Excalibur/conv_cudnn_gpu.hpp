#ifdef USE_CUDA
#ifndef _CONV_CUDNN_GPU_HPP_
#define _CONV_CUDNN_GPU_HPP_
#include "base_conv.hpp"
#include "conv_depthwise_native_gpu.hpp"
namespace glasssix
{
	namespace excalibur
	{
		class conv_cudnn_gpu : public baseconv
		{
		public:
			float one = 1.0, zero = 0.0;
			size_t size;
			cudnnHandle_t cudnn_handle_ = nullptr;
			cudnnTensorDescriptor_t xdesc;
			cudnnTensorDescriptor_t	ydesc;
			cudnnTensorDescriptor_t bdesc;
			cudnnFilterDescriptor_t wdesc;
			cudnnConvolutionDescriptor_t conv_desc;
			// algorithms for forward and backwards convolutions
			cudnnConvolutionFwdAlgo_t fwd_algo_;
			size_t workspace_limit_bytes = 8 * 1024 * 1024;
			float *extra = nullptr;
			size_t current_size;

			conv_cudnn_gpu(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, bool bias_term, int device) 
				: baseconv(input_Channel, output_Channel, kernelSize, stride, pad, bias_term, device) {}

			conv_cudnn_gpu(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device)
			    : baseconv(input_Channel, output_Channel, kernelSize, group, stride, pad, bias_term, device) {}

			virtual ~conv_cudnn_gpu() {}

			void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) override;

			void forward_bias(float* output, const float* bias) override;

			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;
		};
	}
}

#endif // !_CONV_CUDNN_GPU_HPP_
#endif // !USE_CUDA