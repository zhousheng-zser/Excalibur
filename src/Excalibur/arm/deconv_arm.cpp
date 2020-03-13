#include "arm/deconv_arm.hpp"
#include "arm/conv_arm_func.hpp"

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif
#include <iostream>

//#define __ARM_NEON

void glasssix::excalibur::deconv_arm::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	int n = bottom->num();
	int w = bottom->width();
	int h = bottom->height();
	int bottom_cstep = w * h;

	int outw = (w - 1) * stride_ + kernelSize_;
	int outh = (h - 1) * stride_ + kernelSize_;
	int top_cstep = outw * outh;

	top.reset(new tensor<float>(std::vector<int> {n, output_Channel_, outh, outw }, -1, NCHW));

	order_ = bottom->order();
	if (!((order_ == NCHW) || (order_ == NHWC)))
	{
		NOT_IMPLEMENTED;
	}

	std::shared_ptr<tensor<float>> bottom_nchw;
	if (order_ == NHWC)
	{
		tensor_operation_cpu::nhwc2nchw_cpu(bottom, bottom_nchw);
	}
	else
		bottom_nchw = bottom;

	if (input_Channel_ == group_ && output_Channel_ == group_)
	{
#ifdef __ARM_NEON
		if (input_Channel_ % 4 == 0)
		{
			std::shared_ptr<tensor<float> > bottom_pack4 = std::make_shared<tensor<float> >(std::vector<int>{n, input_Channel_ / 4, h, w * 4}, -1, NCHW);
			for (int num_i = 0; num_i < n; num_i++)
			{
				const float *bottom_nchw_data = bottom_nchw->cpu_data() + num_i * input_Channel_ * bottom_cstep;
				float *bottom_pack4_data = bottom_pack4->mutable_cpu_data() + num_i * input_Channel_ * bottom_cstep;

#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int q = 0; q < input_Channel_ / 4; q++)
				{
					float *out = bottom_pack4_data + q * bottom_cstep * 4;

					for (int i = 0; i < h; i++)
					{
						float* outptr = out + i * w * 4;

						for (int j = 0; j < w; j++)
						{
							float* out_elem_ptr = outptr + j * 4;

							for (int k = 0; k < 4; k++)
							{
								int srcq = q * 4 + k;
								if (srcq >= input_Channel_)
									break;

								const float* ptr = bottom_nchw_data + srcq * bottom_cstep + i * w;
								const float* elem_ptr = ptr + j;

								*(out_elem_ptr + k) = *elem_ptr;
							}
						}
					}
				}
			}

			std::shared_ptr<tensor<float>> top_pack4 = std::make_shared<tensor<float> >(std::vector<int>{n, output_Channel_ / 4, outh, outw * 4}, -1, NCHW);

			const float *weight_data_pack4 = weights_pack4_->cpu_data();
			for (int num_i = 0; num_i < n; num_i++)
			{
				const float *bottom_pack4_data = bottom_pack4->cpu_data() + num_i * input_Channel_ * bottom_cstep;
				float top_pack4_data = top_pack4->mutable_cpu_data() + num_i * output_Channel_ * top_cstep;
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int g = 0; g < input_Channel_ / 4; g++)
				{
					float* outptr = top_pack4_data + g * top_cstep * 4;
					const float* kptr = weight_data_pack4 + kernel_length_ * g * 4;
					const float *inptr = bottom_pack4_data + g * bottom_cstep * 4;

					for (int i = 0; i < outh; i++)
					{
						for (int j = 0; j < outw; j++)
						{
							float32x4_t _sum = vdupq_n_f32(0.f);

							if (bias_term)
							{
								_sum = vld1q_f32(bias_data + g * 4);
							}

							for (int y = 0; y < kernelSize_; y++)
							{
								int sys = (i + y - (kernelSize_ - 1));
								if (sys < 0 || sys % stride_ != 0)
									continue;

								int sy = sys / stride_;
								if (sy >= h)
									continue;

								for (int x = 0; x < kernelSize_; x++)
								{
									int sxs = (j + x - (kernelSize_ - 1));
									if (sxs < 0 || sxs % stride_ != 0)
										continue;

									int sx = sxs / stride_;
									if (sx >= w)
										continue;

									const float* sptr = inptr + sy * w * 4 + sx * 4;

									float32x4_t _val = vld1q_f32(sptr);

									int k = y * kernelSize_ + x;

									float32x4_t _w = vld1q_f32(kptr + k * 4);

									_sum = vmlaq_f32(_sum, _val, _w);
								}
							}

							vst1q_f32(outptr + j * 4, _sum);
						}

						outptr += outw * 4;
					}
				}
			}

			for (int num_i = 0; num_i < n; num_i++)
			{
				const float *top_pack4_data = top_pack4->cpu_data() + num_i * output_Channel_ * top_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * output_Channel_ * top_cstep;

#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int q = 0; q < output_Channel_ / 4; q++)
				{
					const float *in = top_pack4_data + q * top_cstep * 4;

					for (int i = 0; i < h; i++)
					{
						float* inptr = in + i * w * 4;

						for (int j = 0; j < w; j++)
						{
							float* in_elem_ptr = inptr + j * 4;

							for (int k = 0; k < 4; k++)
							{
								int dstq = q * 4 + k;
								if (dstq >= output_Channel_)
									break;

								const float* ptr = top_data + dstq * top_cstep + i * w;
								const float* elem_ptr = ptr + j;

								*elem_ptr = *(in_elem_ptr + k);
							}
						}
					}
				}
			}
		}
		else
#endif
		{
			const float *weights_pack1_data = weights_reversed_->cpu_data();
			for (int num_i = 0; num_i < n; num_i++)
			{
				float *top_data = top->mutable_cpu_data() + num_i * output_Channel_ * top_cstep;
				const float *bottom_data = bottom_nchw->cpu_data() + num_i * input_Channel_ * bottom_cstep;
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int g = 0; g < input_Channel_; g++)
				{
					float* outptr = top_data + g * top_cstep;
					const float* kptr = (const float*)weights_pack1_data + kernel_length_ * g;
					const float *inptr = bottom_data + g * bottom_cstep;

					for (int i = 0; i < outh; i++)
					{
						for (int j = 0; j < outw; j++)
						{
							float sum = 0.f;

							if (bias_term_)
							{
								sum = bias_data[g];
							}

							for (int y = 0; y < kernelSize_; y++)
							{
								int sys = (i + y - (kernelSize_ - 1));
								if (sys < 0 || sys % stride_ != 0)
									continue;

								int sy = sys / stride_;
								if (sy >= h)
									continue;

								const float* sptr = inptr + sy * w;

								for (int x = 0; x < kernelSize_; x++)
								{
									int sxs = (j + x - (kernelSize_ - 1));
									if (sxs < 0 || sxs % stride_ != 0)
										continue;

									int sx = sxs / stride_;
									if (sx >= w)
										continue;

									float val = sptr[sx];

									int k = y * stride_ + x;

									float w = kptr[k];

									sum += val * w;
								}
							}

							outptr[j] = sum;
						}

						outptr += outw;
					}
				}
			}
		}
	}
	else if(group_ == 1)
	{
		NOT_IMPLEMENTED;
	}
	else
	{
		NOT_IMPLEMENTED;
	}

	if(pad_ > 0)
		tensor_operation_cpu::cut_border_cpu(top, top, pad_, pad_, pad_, pad_);
}
