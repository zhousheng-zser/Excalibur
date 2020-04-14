#include "arm/batchnorm_arm.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using namespace glasssix::memory;

void glasssix::excalibur::batchnorm_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
{
	orderType order = bottom->order();

	if (order == NCHW)
	{
		int num = bottom->num();

		int w = bottom->width();
		int h = bottom->height();
		int channels = bottom->channels();
		int size = w * h;

		const float *a_data = a_->cpu_data();
		const float *b_data = b_->cpu_data();

		for (int num_i = 0; num_i < num; num_i++)
		{
			float *bottom_data = bottom->mutable_cpu_data() + num_i * channels * size;

#pragma omp parallel for
			for (int q = 0; q<channels; q++)
			{
				float* ptr = bottom_data + (q)* size;

				float a = a_data[q];
				float b = b_data[q];

#if __ARM_NEON
				int nn = size >> 2;
				int remain = size - (nn << 2);
#else
				int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
				if (nn > 0)
				{
					asm volatile(
						"dup        v1.4s, %w4             \n"
						"dup        v2.4s, %w5             \n"
						"0:                                \n"
						"prfm       pldl1keep, [%1, #128]  \n"
						"ld1        {v0.4s}, [%1]          \n"
						"orr        v3.16b, v1.16b, v1.16b \n"
						"fmla       v3.4s, v0.4s, v2.4s    \n"
						"subs       %w0, %w0, #1           \n"
						"st1        {v3.4s}, [%1], #16     \n"
						"bne        0b                     \n"
						: "=r"(nn),     // %0
						"=r"(ptr)     // %1
						: "0"(nn),
						"1"(ptr),
						"r"(b),       // %4
						"r"(a)        // %5
						: "cc", "memory", "v0", "v1", "v2", "v3"
						);
				}
#else
				if (nn > 0)
				{
					asm volatile(
						"vdup.f32   q1, %4              \n"
						"vdup.f32   q2, %5              \n"
						"0:                             \n"
						"pld        [%1, #128]          \n"
						"vld1.f32   {d0-d1}, [%1 :128]  \n"
						"vorr.32    q3, q1, q1          \n"
						"vmla.f32   q3, q0, q2          \n"
						"subs       %0, #1              \n"
						"vst1.f32   {d6-d7}, [%1 :128]! \n"
						"bne        0b                  \n"
						: "=r"(nn),     // %0
						"=r"(ptr)     // %1
						: "0"(nn),
						"1"(ptr),
						"r"(b),       // %4
						"r"(a)        // %5
						: "cc", "memory", "q0", "q1", "q2", "q3"
						);
				}
#endif // __aarch64__
#endif // __ARM_NEON
				for (; remain>0; remain--)
				{
					*ptr = a * *ptr + b;

					ptr++;
				}
			}
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}
