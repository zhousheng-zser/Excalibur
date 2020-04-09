#include "arm/eltwise_arm.hpp"
#include <cstring>
#include <cfloat>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void glasssix::excalibur::eltwise_arm::Forward_cpu(const std::vector<std::shared_ptr<memory::tensor<float>>> bottom, std::shared_ptr<memory::tensor<float>>& top)
{
	if (bottom[0]->order() == memory::NCHW)
	{
		for (int i = 1; i < bottom.size(); ++i) {
			CHECK(bottom[i]->data_shape() == bottom[0]->data_shape());
		}

		top.reset(new memory::tensor<float>(bottom[0]->data_shape(), device_, bottom[0]->order()));
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int w = bottom[0]->width();
		int h = bottom[0]->height();
		int size = w * h;
		
		float *top_ptr = top->mutable_cpu_data();
		const int count = (top)->count(0, (top)->data_shape().size());

		switch (type_)
		{
		case SUM:
		{
			memset(top_ptr, 0, count * sizeof(float));
			coeffs_ = std::vector<float>(bottom.size(), 1);
			const float *coeffs_ptr = coeffs_.data();
			for (int i = 0; i < num; i++)
			{
				float *top_data = top_ptr + i * channels * size;
				const float *bottom_data0 = bottom[0]->cpu_data() + i * channels * size;
				const float *bottom_data1 = bottom[1]->cpu_data() + i * channels * size;
				float coeff0 = coeffs_ptr[0];
				float coeff1 = coeffs_ptr[1];
#pragma omp parallel for
				for (int q = 0; q < channels; q++)
				{
					const float* ptr = bottom_data0 + q * size;
					const float* ptr1 = bottom_data1 + q * size;
					float* outptr = top_data + q * size;

#ifdef __ARM_NEON
					int nn = size >> 2;
					int remain = size - (nn << 2);
#else
					int remain = size;
#endif // __ARM_NEON

#ifdef __ARM_NEON
					float32x4_t _coeff0 = vdupq_n_f32(coeff0);
					float32x4_t _coeff1 = vdupq_n_f32(coeff1);
#if __aarch64__
					if (nn > 0)
					{
						asm volatile(
							"0:                               \n"
							"prfm       pldl1keep, [%1, #128] \n"
							"prfm       pldl1keep, [%2, #128] \n"
							"ld1        {v0.4s}, [%1], #16    \n"
							"ld1        {v1.4s}, [%2], #16    \n"
							"fmul       v0.4s, v0.4s, %8.4s   \n"
							"fmla       v0.4s, v1.4s, %9.4s   \n"
							"subs       %w0, %w0, #1          \n"
							"st1        {v0.4s}, [%3], #16    \n"
							"bne        0b                    \n"
							: "=r"(nn),     // %0
							"=r"(ptr),    // %1
							"=r"(ptr1),   // %2
							"=r"(outptr)  // %3
							: "0"(nn),
							"1"(ptr),
							"2"(ptr1),
							"3"(outptr),
							"w"(_coeff0), // %8
							"w"(_coeff1)  // %9
							: "cc", "memory", "v0", "v1"
							);
					}
#else
					if (nn > 0)
					{
						asm volatile(
							"0:                             \n"
							"pld        [%1, #128]          \n"
							"pld        [%2, #128]          \n"
							"vld1.f32   {d0-d1}, [%1 :128]! \n"
							"vld1.f32   {d2-d3}, [%2 :128]! \n"
							"vmul.f32   q0, q0, %q8         \n"
							"vmla.f32   q0, q1, %q9         \n"
							"subs       %0, #1              \n"
							"vst1.f32   {d0-d1}, [%3 :128]! \n"
							"bne        0b                  \n"
							: "=r"(nn),     // %0
							"=r"(ptr),    // %1
							"=r"(ptr1),   // %2
							"=r"(outptr)  // %3
							: "0"(nn),
							"1"(ptr),
							"2"(ptr1),
							"3"(outptr),
							"w"(_coeff0), // %8
							"w"(_coeff1)  // %9
							: "cc", "memory", "q0", "q1"
							);
					}
#endif // __aarch64__
#endif // __ARM_NEON
					for (; remain>0; remain--)
					{
						*outptr = *ptr * coeff0 + *ptr1 * coeff1;

						ptr++;
						ptr1++;
						outptr++;
					}
				}

				for (size_t b = 2; b<bottom.size(); b++)
				{
					const float *bottom_datab = bottom[b]->cpu_data() + i * channels * size;
					float coeff = coeffs_ptr[b];
#pragma omp parallel for
					for (int q = 0; q<channels; q++)
					{
						const float* ptr = bottom_datab + q * size;
						float* outptr = top_data + q * size;

#ifdef __ARM_NEON
						int nn = size >> 2;
						int remain = size - (nn << 2);
#else
						int remain = size;
#endif // __ARM_NEON

#ifdef __ARM_NEON
						float32x4_t _coeff = vdupq_n_f32(coeff);
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"0:                               \n"
								"prfm       pldl1keep, [%1, #128] \n"
								"prfm       pldl1keep, [%2, #128] \n"
								"ld1        {v0.4s}, [%1], #16    \n"
								"ld1        {v1.4s}, [%2]         \n"
								"fmla       v1.4s, v0.4s, %6.4s   \n"
								"subs       %w0, %w0, #1          \n"
								"st1        {v1.4s}, [%2], #16    \n"
								"bne        0b                    \n"
								: "=r"(nn),     // %0
								"=r"(ptr),    // %1
								"=r"(outptr)  // %2
								: "0"(nn),
								"1"(ptr),
								"2"(outptr),
								"w"(_coeff)   // %6
								: "cc", "memory", "v0", "v1"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                             \n"
								"pld        [%1, #128]          \n"
								"pld        [%2, #128]          \n"
								"vld1.f32   {d0-d1}, [%1 :128]! \n"
								"vld1.f32   {d2-d3}, [%2 :128]  \n"
								"vmla.f32   q1, q0, %q6         \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d2-d3}, [%2 :128]! \n"
								"bne        0b                  \n"
								: "=r"(nn),     // %0
								"=r"(ptr),    // %1
								"=r"(outptr)  // %2
								: "0"(nn),
								"1"(ptr),
								"2"(outptr),
								"w"(_coeff)   // %6
								: "cc", "memory", "q0", "q1"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain>0; remain--)
						{
							*outptr += *ptr * coeff;

							ptr++;
							outptr++;
						}
					}
				}
			}
		}
		break;
		case MAX:
		{
			memset(top_ptr, static_cast<float>(-FLT_MAX), count * sizeof(float));
			for (int i = 0; i < num; i++)
			{
				float *top_data = top_ptr + i * channels * size;
				const float *bottom_data0 = bottom[0]->cpu_data() + i * channels * size;
				const float *bottom_data1 = bottom[1]->cpu_data() + i * channels * size;

#pragma omp parallel for
				for (int q = 0; q<channels; q++)
				{
					const float* ptr = bottom_data0 + q * size;
					const float* ptr1 = bottom_data1 + q * size;
					float* outptr = top_data + q * size;

#ifdef __ARM_NEON
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
							"0:                               \n"
							"prfm       pldl1keep, [%1, #128] \n"
							"prfm       pldl1keep, [%2, #128] \n"
							"ld1        {v0.4s}, [%1], #16    \n"
							"ld1        {v1.4s}, [%2], #16    \n"
							"fmax       v0.4s, v0.4s, v1.4s   \n"
							"subs       %w0, %w0, #1          \n"
							"st1        {v0.4s}, [%3], #16    \n"
							"bne        0b                    \n"
							: "=r"(nn),     // %0
							"=r"(ptr),    // %1
							"=r"(ptr1),   // %2
							"=r"(outptr)  // %3
							: "0"(nn),
							"1"(ptr),
							"2"(ptr1),
							"3"(outptr)
							: "cc", "memory", "v0", "v1"
							);
					}
#else
					if (nn > 0)
					{
						asm volatile(
							"0:                             \n"
							"pld        [%1, #128]          \n"
							"pld        [%2, #128]          \n"
							"vld1.f32   {d0-d1}, [%1 :128]! \n"
							"vld1.f32   {d2-d3}, [%2 :128]! \n"
							"vmax.f32   q0, q0, q1          \n"
							"subs       %0, #1              \n"
							"vst1.f32   {d0-d1}, [%3 :128]! \n"
							"bne        0b                  \n"
							: "=r"(nn),     // %0
							"=r"(ptr),    // %1
							"=r"(ptr1),   // %2
							"=r"(outptr)  // %3
							: "0"(nn),
							"1"(ptr),
							"2"(ptr1),
							"3"(outptr)
							: "cc", "memory", "q0", "q1"
							);
					}
#endif // __aarch64__
#endif // __ARM_NEON
					for (; remain>0; remain--)
					{
						*outptr = std::max(*ptr, *ptr1);

						ptr++;
						ptr1++;
						outptr++;
					}
				}

				for (size_t b = 2; b<bottom.size(); b++)
				{
					const float *bottom_datab = bottom[b]->cpu_data() + i * channels * size;
#pragma omp parallel for
					for (int q = 0; q<channels; q++)
					{
						const float* ptr = bottom_datab + q * size;
						float* outptr = top_data + q * size;

#ifdef __ARM_NEON
						int nn = size >> 2;
						int remain = size - (nn << 2);
#else
						int remain = size;
#endif // __ARM_NEON

#ifdef __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"0:                               \n"
								"prfm       pldl1keep, [%1, #128] \n"
								"prfm       pldl1keep, [%2, #128] \n"
								"ld1        {v0.4s}, [%1], #16    \n"
								"ld1        {v1.4s}, [%2]         \n"
								"fmax       v0.4s, v0.4s, v1.4s   \n"
								"subs       %w0, %w0, #1          \n"
								"st1        {v0.4s}, [%2], #16    \n"
								"bne        0b                    \n"
								: "=r"(nn),     // %0
								"=r"(ptr),    // %1
								"=r"(outptr)  // %2
								: "0"(nn),
								"1"(ptr),
								"2"(outptr)
								: "cc", "memory", "v0", "v1"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                             \n"
								"pld        [%1, #128]          \n"
								"pld        [%2, #128]          \n"
								"vld1.f32   {d0-d1}, [%1 :128]! \n"
								"vld1.f32   {d2-d3}, [%2 :128]  \n"
								"vmax.f32   q0, q0, q1          \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d1}, [%2 :128]! \n"
								"bne        0b                  \n"
								: "=r"(nn),     // %0
								"=r"(ptr),    // %1
								"=r"(outptr)  // %2
								: "0"(nn),
								"1"(ptr),
								"2"(outptr)
								: "cc", "memory", "q0", "q1"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain>0; remain--)
						{
							*outptr = std::max(*ptr, *outptr);

							ptr++;
							outptr++;
						}
					}
				}
			}
		}
		break;
		default:
			LOG(FATAL) << "Unknown elementwise operation.";
			break;
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}
