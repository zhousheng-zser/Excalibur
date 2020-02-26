#include "arm/prelu_arm.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

void glasssix::excalibur::prelu_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
{
	
	orderType order_ = bottom->order();

	if (order_ == NCHW)
	{
		int num = bottom->num();
		int w = bottom->width();
		int h = bottom->height();
		int channels = bottom->channels();
		int size = w * h;

		const float *slope_data = slope_data_->cpu_data();

		for (int num_i = 0; num_i < num; num_i++)
		{
			float *bottom_data = bottom->mutable_cpu_data() + num_i * channels * size;

#pragma omp parallel for
			for (int q = 0; q<channels; q++)
			{
				float* ptr = bottom_data + (q)* size;
				float slope = slope_data[0];
				if (!is_shared_)
				{
					slope = slope_data[q];
				}
				

#if __ARM_NEON
				int nn = size >> 2;
				int remain = size - (nn << 2);
#else
				int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
				float32x4_t _zero = vdupq_n_f32(0.f);
				float32x4_t _slope = vdupq_n_f32(slope);
				for (; nn>0; nn--)
				{
					float32x4_t _p = vld1q_f32(ptr);
					uint32x4_t _lemask = vcleq_f32(_p, _zero);
					float32x4_t _ps = vmulq_f32(_p, _slope);
					_p = vbslq_f32(_lemask, _ps, _p);
					vst1q_f32(ptr, _p);

					ptr += 4;
				}
#else
				if (nn > 0)
				{
					asm volatile(
						"veor       q1, q0, q0          \n"
						"vdup.f32   q2, %4              \n"
						"0:                             \n"
						"pld        [%1, #128]          \n"
						"vld1.f32   {d0-d1}, [%1 :128]  \n"
						"vcle.f32   q3, q0, q1          \n"
						"vmul.f32   q4, q0, q2          \n"
						"vbit.32    q0, q4, q3          \n"
						"subs       %0, #1              \n"
						"vst1.f32   {d0-d1}, [%1 :128]! \n"
						"bne        0b                  \n"
						: "=r"(nn),     // %0
						"=r"(ptr)     // %1
						: "0"(nn),
						"1"(ptr),
						"r"(slope)    // %4
						: "cc", "memory", "q0", "q1", "q2", "q3", "q4"
						);
				}
#endif // __aarch64__
#endif // __ARM_NEON
				for (; remain>0; remain--)
				{
					if (*ptr < 0)
						*ptr *= slope;

					ptr++;
				}
			}
		}
	}
	else
	{
		prelu::Forward_cpu(bottom);
	}
}
