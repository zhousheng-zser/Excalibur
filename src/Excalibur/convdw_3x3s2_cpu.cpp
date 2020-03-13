#include "convdw_3x3s2_cpu.hpp"
#include "../../include/Excalibur/tensor_operation_cpu.hpp"
#include <iostream>
#include <fstream>

namespace glasssix
{
	namespace excalibur
	{
		convdw_3x3s2_cpu::convdw_3x3s2_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization)
			: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device, int8_quantization)
		{

		}

		void convdw_3x3s2_cpu::forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col) {}

		void convdw_3x3s2_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col) {}

		void convdw_3x3s2_cpu::forward_bias(float* output, const float* bias_data) {}

		void convdw_3x3s2_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 2);
			CHECK_GT(group_, 1);
			
			int num = bottom->num();
			int inch = bottom->channels();
			int inh = bottom->height();
			int inw = bottom->width();
			int bottom_cstep = inw * inh;
			const float* bottom_data = bottom->cpu_data();
			order_ = bottom->order();

			int outch = output_Channel_;
			int outh = (inh + 2 * pad_ - kernelSize_) / stride_ + 1;
			int outw = (inw + 2 * pad_ - kernelSize_) / stride_ + 1;
			top.reset(new tensor<float>(std::vector<int>{num, outch, outh, outw}));
			int top_cstep = outw * outh;
			float* top_data = top->mutable_cpu_data();			

			std::shared_ptr<tensor<float>> bottom_bordered;
			tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_, pad_, pad_);
			if (order_ == NHWC)
			{
				tensor_operation_cpu::nhwc2nchw_cpu(bottom_bordered, bottom_bordered);
			}
			inh = bottom_bordered->height();
			inw = bottom_bordered->width();
			bottom_cstep = inw * inh;
			bottom_data = bottom_bordered->cpu_data();

			const int tailstep = inw - 2 * outw + inw;

			for (int n = 0; n < num; n++)
			{
				const float* bottom_data_num = bottom_data + n * inch * bottom_cstep;
				float* top_data_num = top_data + n * outch * top_cstep;

#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int g = 0; g < group_; g++)
				{
					float *out = top_data_num + g * top_cstep;

					const float bias0 = bias_term_ ? bias_data[g] : 0.f;

					const float* kernel0 = weights_data + g * 9;

					float* outptr = out;

					const float* img0 = bottom_data_num + g * bottom_cstep;

					const float* r0 = img0;
					const float* r1 = img0 + inw;
					const float* r2 = img0 + inw * 2;

					const float* k0 = kernel0;
					const float* k1 = kernel0 + 3;
					const float* k2 = kernel0 + 6;

#if SIMD_TYPE >= SIMDTYPE_SSE
					__m128 k0_data = _mm_loadu_ps(k0);
					__m128 k1_data = _mm_loadu_ps(k1);
					__m128 k2_data = _mm_loadu_ps(k2);

					int i = 0;

					for (; i < outh; i++)
					{
						int remain = outw;

						for (; remain > 0; remain--)
						{
							__m128 r0_data = _mm_loadu_ps(r0);
							__m128 r1_data = _mm_loadu_ps(r1);
							__m128 r2_data = _mm_loadu_ps(r2);

							*outptr = mul_add_3x3_simd(r0_data, r1_data, r2_data, k0_data, k1_data, k2_data, bias0);

							r0 += 2;
							r1 += 2;
							r2 += 2;
							outptr++;
						}

						r0 += tailstep;
						r1 += tailstep;
						r2 += tailstep;
					}

#else
					int i = 0;

					for (; i < outh; i++)
					{
						int remain = outw;

						for (; remain > 0; remain--)
						{
							*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);

							r0 += 2;
							r1 += 2;
							r2 += 2;
							outptr++;
						}

						r0 += tailstep;
						r1 += tailstep;
						r2 += tailstep;
					}
#endif

				}

			}

			if (order_ == NHWC)
			{
				tensor_operation_cpu::nchw2nhwc_cpu(top, top);
			}
		}
	}
}