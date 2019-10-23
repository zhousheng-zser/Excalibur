#include "arm/scale_arm.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

void glasssix::excalibur::scale_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
{
	orderType order = bottom->order();

	if (order == NCHW)
	{
		int num = bottom->num();

		int w = bottom->width();
		int h = bottom->height();
		int channels = bottom->channels();
		int size = w * h;

		const float *weights_data = weights_->cpu_data();

		for (int num_i = 0; num_i < num; num_i++)
		{
			float *bottom_data = bottom->mutable_cpu_data() + num_i * channels * size;
			if (bias_term_)
			{
				const float *bias_data = bias_->cpu_data();

				const float* weights_ptr = weights_data;
				const float* bias_ptr = bias_data;
#pragma omp parallel for
				for (int q = 0; q<channels; q++)
				{
					float* ptr = bottom_data + (q)* size;

					float s = weights_ptr[q];
					float bias = bias_ptr[q];

#if __ARM_NEON
					int nn = size >> 2;
					int remain = size - (nn << 2);
#else
					int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
					float32x4_t _s = vdupq_n_f32(s);
					float32x4_t _bias = vdupq_n_f32(bias);
					for (; nn>0; nn--)
					{
						float32x4_t _p = vld1q_f32(ptr);
						_p = vmlaq_f32(_bias, _p, _s);
						vst1q_f32(ptr, _p);

						ptr += 4;
					}
#endif // __ARM_NEON

					for (; remain>0; remain--)
					{
						*ptr = *ptr * s + bias;

						ptr++;
					}
				}
			}
			else
			{
				const float* weights_ptr = weights_data;
#pragma omp parallel for
				for (int q = 0; q<channels; q++)
				{
					float* ptr = bottom_data + (q)* size;

					float s = weights_ptr[q];

#if __ARM_NEON
					int nn = size >> 2;
					int remain = size - (nn << 2);
#else
					int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
					float32x4_t _s = vdupq_n_f32(s);
					for (; nn>0; nn--)
					{
						float32x4_t _p = vld1q_f32(ptr);
						_p = vmulq_f32(_p, _s);
						vst1q_f32(ptr, _p);

						ptr += 4;
					}
#endif // __ARM_NEON

					for (; remain>0; remain--)
					{
						*ptr *= s;

						ptr++;
					}
				}
			}
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}
