#include "arm/sigmoid_arm.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm/neon_mathfun.hpp"
#endif // __ARM_NEON

void glasssix::excalibur::sigmoid_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
{
	int num = bottom->num();
	int w = bottom->width();
	int h = bottom->height();
	int channels = bottom->channels();
	int size = w * h;

	for (int num_i = 0; num_i < num; num_i++)
	{
		float *bottom_data = bottom->mutable_cpu_data() + num_i * channels * size;
#pragma omp parallel for
		for (int q = 0; q<channels; q++)
		{
			float* ptr = bottom_data + (q)* size;

#if __ARM_NEON
			int nn = size >> 2;
			int remain = size - (nn << 2);
#else
			int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
			float32x4_t _one = vdupq_n_f32(1.f);
			for (; nn>0; nn--)
			{
				float32x4_t _p = vld1q_f32(ptr);
				_p = vnegq_f32(_p);
				_p = exp_ps(_p);
				_p = vaddq_f32(_p, _one);
				float32x4_t _outp = vrecpeq_f32(_p);
				_outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
				//             _outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
				vst1q_f32(ptr, _outp);

				ptr += 4;
		}
#endif // __ARM_NEON
			for (; remain>0; remain--)
			{
				*ptr = 1.f / (1.f + exp(-*ptr));

				ptr++;
			}
		}
	}
}
