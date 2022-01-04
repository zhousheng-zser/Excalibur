#include "../../include/Excalibur/arm/operation_eltwise_arm.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_eltwise_arm<Dtype>::operation_eltwise_arm(const operation_param & param) : operation<Dtype>(param)
		{
			coeffs_ = std::vector<float>(param.input_count_, 1.0f);
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					type_ = (eltwise_type)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23301")
				{
					auto coeff_str = split_string(split_string(attrs[i], "=")[1], ",");
					CHECK_EQ((eltwise_type)type_, eltwise_type::SUM);
					for (size_t j = 1; j < atoi(coeff_str[0].c_str()); j++)
					{
						coeffs_[j - 1] = atof(coeff_str[j].c_str());
					}
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Eltwise Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}
		template<typename Dtype>
		void operation_eltwise_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			for (int i = 0; i < bottoms.size(); ++i)
			//for (int i = 1; i < bottoms.size(); ++i)
			{
				//CHECK(bottoms[i]->data_shape() == bottoms[0]->data_shape());
				if (bottoms[i]->order() == memory::NHWC)
					bottoms[i]->convert_order();
			}

			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			int num = bottoms[0]->num();
			int channels = bottoms[0]->channels();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			int size = w * h;

			float *top_ptr = tops[0]->mutable_cpu_data();
			const int count = tops[0]->count();

			switch (type_)
			{
			case SUM:
			{
				memset(top_ptr, 0, count * sizeof(float));
				const float *coeffs_ptr = coeffs_.data();
				for (int i = 0; i < num; i++)
				{
					float *top_data = top_ptr + i * channels * size;
					const float *bottom_data_a = bottoms[0]->cpu_data() + i * channels * size;
					const float *bottom_data_b = bottoms[1]->cpu_data() + i * channels * size;
					float coeff0 = coeffs_ptr[0];
					float coeff1 = coeffs_ptr[1];
#pragma omp parallel for num_threads(2) 
					for (int q = 0; q < channels; q++)
					{
						const float* ptr = bottom_data_a + q * size;
						const float* ptr1 = bottom_data_b + q * size;
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
								"vld1.f32   {d0-d1}, [%1]! \n"
								"vld1.f32   {d2-d3}, [%2]! \n"			
								"vmul.f32   q0, q0, %q8         \n"
								"vmla.f32   q0, q1, %q9         \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d1}, [%3]! \n"
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
						for (; remain > 0; remain--)
						{
							*outptr = *ptr * coeff0 + *ptr1 * coeff1;

							ptr++;
							ptr1++;
							outptr++;
						}
					}

					for (size_t b = 2; b < bottoms.size(); b++)
					{
						const float *bottom_datab = bottoms[b]->cpu_data() + i * channels * size;
						float coeff = coeffs_ptr[b];
#pragma omp parallel for num_threads(2) 
						for (int q = 0; q < channels; q++)
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
									"vld1.f32   {d0-d1}, [%1]! \n"
									"vld1.f32   {d2-d3}, [%2]  \n"
									"vmla.f32   q1, q0, %q6         \n"
									"subs       %0, #1              \n"
									"vst1.f32   {d2-d3}, [%2]! \n"
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
							for (; remain > 0; remain--)
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
					const float *bottom_data0 = bottoms[0]->cpu_data() + i * channels * size;
					const float *bottom_data1 = bottoms[1]->cpu_data() + i * channels * size;

#pragma omp parallel for num_threads(2) 
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
								"vld1.f32   {d0-d1}, [%1]! \n"
								"vld1.f32   {d2-d3}, [%2]! \n"
								"vmax.f32   q0, q0, q1          \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d1}, [%3]! \n"
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
						for (; remain > 0; remain--)
						{
							*outptr = std::max(*ptr, *ptr1);

							ptr++;
							ptr1++;
							outptr++;
						}
					}

					for (size_t b = 2; b < bottoms.size(); b++)
					{
						const float *bottom_datab = bottoms[b]->cpu_data() + i * channels * size;
#pragma omp parallel for num_threads(2) 
						for (int q = 0; q < channels; q++)
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
									"vld1.f32   {d0-d1}, [%1]! \n"
									"vld1.f32   {d2-d3}, [%2]  \n"
									"vmax.f32   q0, q0, q1          \n"
									"subs       %0, #1              \n"
									"vst1.f32   {d0-d1}, [%2]! \n"
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
							for (; remain > 0; remain--)
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

		INSTANCE_CLASS(operation_eltwise_arm);
		REGISTE(operation_eltwise_arm);
	}
}