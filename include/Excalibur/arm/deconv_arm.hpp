#ifndef _DECONV_ARM_HPP_
#define _DECONV_ARM_HPP_
#include "../base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class deconv_arm : public baseconv
		{
		public:
			deconv_arm(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device)
				: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device)
			{
				kernel_length_ = kernelSize * kernelSize;
			}

			virtual ~deconv_arm() {}

			void set_weights(float *weights)
			{
				weights_->set_cpu_data(weights);
				weights_data = weights_->cpu_data();
				const float *p = weights_data;

				if (input_Channel_ == group_ && output_Channel_ == group_)
				{
					weights_reversed_.reset(new memory::tensor<float>(std::vector<int>{input_Channel_*output_Channel_ * kernel_length_ / group_}, device_));
					float *pt = weights_reversed_->mutable_cpu_data();
					for (int i = 0; i < (input_Channel_ / group_)*(output_Channel_ / group_)*group_; i++)
					{
						for (int k = 0; k < kernel_length_; k++)
						{
							pt[kernel_length_ - 1 - k] = p[k];
						}

						p += kernel_length_;
						pt += kernel_length_;
					}

#ifdef __ARM_NEON
					if (input_Channel_ % 4 == 0)
					{
						weights_pack4_.reset(new memory::tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernel_length_ / group_}, device_));

						float *weights_pack4_data = weights_pack4_->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int i = 0; i < input_Channel_ / 4; i++)
						{
							float* outptr = weights_pack4_data + i * kernel_length_ * 4;

							for (int j = 0; j < kernel_length_; j++)
							{
								float* out_elem_ptr = outptr + j * 4;

								for (int k = 0; k < 4; k++)
								{
									int srcy = (i * 4 + k);
									if (srcy >= input_Channel_)
										break;

									const float* ptr = pt + srcy * kernel_length_;
									const float* elem_ptr = ptr + j;

									*(out_elem_ptr + k) = *elem_ptr;
								}
							}
						}
					}
#endif
				}
			}

			void Forward(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override;
		private:
			std::shared_ptr<memory::tensor<float>> weights_reversed_;
#ifdef __ARM_NEON
			std::shared_ptr<memory::tensor<float>> weights_pack4_;
#endif

			void forward_gemm(const float* input, const float* weights_packed, float* output, bool skip_im2col = false) override {}
			void forward_gemm(const signed char * input, const signed char *weights, int *output, bool skip_im2col = false) override {}
			void forward_bias(float* output, const float* bias) override {}

#ifdef USE_CUDA
			void Forward(cublasHandle_t &cublas_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override {}
			void forward_gemm(cublasHandle_t &cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) override {}
			void forward_gemm(cublasHandle_t &cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) override {}
			void forward_bias(cublasHandle_t &cublas_handle_, float* output, const float* bias) override {}
#ifdef USE_CUDNN
			void Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top) override {}
#endif //!USE_CUDNN
#endif // !USE_CUDA
		};
	}
}
#endif
