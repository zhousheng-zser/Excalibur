#include "../../include/Excalibur/operation_general_conv.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_general_conv<Dtype>::operation_general_conv(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					output_channel_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					kernel_size_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					kernel_size_h_ = kernel_size_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					dilation_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					dilation_h_ = dilation_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					stride_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					stride_h_ = stride_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "4")
				{
					pad_left_ = atoi(split_string(attrs[i], "=")[1].c_str());
					pad_right_ = pad_left_;
					pad_top_ = pad_left_;
					pad_bottom_ = pad_left_;
				}
				else if (split_string(attrs[i], "=")[0] == "5")
				{
					bias_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "6")
				{
					weight_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "7")
				{
					group_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "8")
				{
					int8_scale_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
					this->params_.set_int8_quantization(int8_scale_term_);
				}
				else if (split_string(attrs[i], "=")[0] == "9")
				{
					activation_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "10")
				{
					NOT_IMPLEMENTED;
				}
				else if (split_string(attrs[i], "=")[0] == "11")
				{
					kernel_size_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "12")
				{
					dilation_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "13")
				{
					stride_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "14")
				{
					pad_top_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "15")
				{
					pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "16")
				{
					pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "17")
				{
					impl_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "18")
				{
					//pad_value_ = (Dtype)atof(split_string(attrs[i], "=")[1].c_str());
					output_pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
					output_pad_bottom_ = output_pad_right_;
				}
				else if (split_string(attrs[i], "=")[0] == "19")
				{
					output_pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "20")
				{
					output_dim_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
					output_dim_w_ = output_dim_h_;
				}
				else if (split_string(attrs[i], "=")[0] == "21")
				{
					output_dim_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Convolution Attribution " << split_string(attrs[i], "=")[0];
				}
			}

			CHECK_EQ(output_channel_ % group_, 0);
			
		}

		template<typename Dtype>
		void operation_general_conv<Dtype>::suffix_activation_cpu_f32(std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			if (this->activation_type_ == 0)
			{
				return;
			}
			else if (this->activation_type_ == 1)
			{
				//ReLU
				for (size_t i = 0; i < tops.size(); i++)
				{
					float* top_data = tops[i]->mutable_cpu_data();
					const int count = tops[i]->count();
#if (SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_SSE4_2_VERSION) //SSE
					int simd_times = (count - count % 4) / 4;
					__m128 zero = _mm_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < simd_times; j++)
					{
						__m128 d = _mm_load_ps(top_data + 4 * j);
						d = _mm_max_ps(zero, d);
						_mm_store_ps(top_data + 4 * j, d);
					}
					for (int j = 4 * simd_times; j < count; j++)
					{
						top_data[j] = top_data[j] >= 0.0f ? top_data[j] : 0.0f;
					}
#elif (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
					int simd_times = (count - count % 8) / 8;
					__m256 zero = _mm256_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < simd_times; j++)
					{
						__m256 d = _mm256_load_ps(top_data + 8 * j);
						d = _mm256_max_ps(zero, d);
						_mm256_store_ps(top_data + 8 * j, d);
					}
					for (int j = 8 * simd_times; j < count; j++)
					{
						top_data[j] = top_data[j] >= 0.0f ? top_data[j] : 0.0f;
					}
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < count; j++)
					{
						top_data[j] = top_data[j] >= 0.0f ? top_data[j] : 0.0f;
					}
#endif
				}
			}
			else if (this->activation_type_ == 2)
			{
				//Leaky-ReLU
				NOT_IMPLEMENTED;
			}
			else if (this->activation_type_ == 3)
			{
				//ReLU6
				for (size_t i = 0; i < tops.size(); i++)
				{
					float* top_data = tops[i]->mutable_cpu_data();
					const int count = tops[i]->count();
#if (SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_SSE4_2_VERSION) //SSE
					int simd_times = (count - count % 4) / 4;
					__m128 zero = _mm_setzero_ps();
					__m128 v6 = _mm_set1_ps(6.0f);
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < simd_times; j++)
					{
						__m128 d = _mm_load_ps(top_data + 4 * j);
						d = _mm_max_ps(zero, d);
						d = _mm_and_ps(_mm_cmple_ps(d, v6), d);
						_mm_store_ps(top_data + 4 * j, d);
					}
					for (int j = 4 * simd_times; j < count; j++)
					{
						top_data[j] = (top_data[j] >= 0.0f && top_data[j] <= 6.0f) ? top_data[j] : 0.0f;
					}
#elif (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
					int simd_times = (count - count % 8) / 8;
					__m256 zero = _mm256_setzero_ps();
					__m256 v6 = _mm256_set1_ps(6.0f);
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < simd_times; j++)
					{
						__m256 d = _mm256_load_ps(top_data + 8 * j);
						d = _mm256_max_ps(zero, d);
						d = _mm256_and_ps(_mm256_cmp_ps(d, v6, 2), d);
						_mm256_store_ps(top_data + 8 * j, d);
					}
					for (int j = 8 * simd_times; j < count; j++)
					{
						top_data[j] = (top_data[j] >= 0.0f && top_data[j] <= 6.0f) ? top_data[j] : 0.0f;
					}
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int j = 0; j < count; j++)
					{
						top_data[j] = (top_data[j] >= 0.0f && top_data[j] <= 6.0f) ? top_data[j] : 0.0f;
					}
#endif
				}
			}
			else if (this->activation_type_ == 4)
			{
				//Sigmoid
				for (size_t i = 0; i < tops.size(); i++)
				{
					float* top_data = tops[i]->mutable_cpu_data();
					const int count = tops[i]->count();
					for (int i = 0; i < count; ++i)
					{
						top_data[i] = 1. / (1. + exp(-top_data[i]));
					}
				}
			}
			else if (this->activation_type_ == 5)
			{
				//Mish
				for (size_t i = 0; i < tops.size(); i++)
				{
					float* top_data = tops[i]->mutable_cpu_data();
					const int count = tops[i]->count();
					for (int i = 0; i < count; ++i)
					{
						top_data[i] = top_data[i] * tanh(log(1 + exp(top_data[i])));
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_general_conv<Dtype>::quantize_float32_to_int8(const std::shared_ptr<memory::tensor<float>>& src,
			std::shared_ptr<memory::tensor<signed char>>& dst)
		{
			int num = src->num();
			int w = src->width();
			int h = src->height();
			int channels = src->channels();
			int size = w * h;

			dst.reset(new memory::tensor<signed char>(src->data_shape(), this->params_.device_, src->order(), nullptr));

			float scale = this->featmap_scaletable_i8_[0];

			for (size_t n = 0; n < num; n++)
			{
				const float* bottom_data = src->cpu_data() + n * size * channels;
				signed char* bottom_int8_data = dst->mutable_cpu_data() + n * size * channels;
#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
				for (int q = 0; q < channels; q++)
				{
					const float* ptr = bottom_data + q * size;
					signed char* outptr = bottom_int8_data + q * size;
#if __ARM_NEON
					int nn = size >> 3;
					int remain = size & 7;
#else
					int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
					if (nn > 0)
					{
						asm volatile(
							"dup    v2.4s, %w6                   \n" //scale
							"0:                                  \n"
							"prfm   pldl1keep, [%1, #128]        \n"
							"ld1    {v0.4s, v1.4s}, [%1], #32    \n" //data
							// bottom_f32 = bottom_f32 * scale
							"fmul   v3.4s, v0.4s, v2.4s          \n"
							"fmul   v4.4s, v1.4s, v2.4s          \n"
							// top_f32 -> top_s32
							"fcvtas v5.4s, v3.4s                 \n"
							"fcvtas v6.4s, v4.4s                 \n"
							// top_s32 -> top_s16
							"sqxtn  v7.4h, v5.4s                 \n"
							"sqxtn2 v7.8h, v6.4s                 \n"
							// top_s16 -> top_s8
							"sqxtn  v8.8b, v7.8h                 \n"
							// save top_s8
							"st1    {v8.8b}, [%2], #8            \n"
							"subs   %w0, %w0, #1                 \n"
							"bne    0b                           \n"
							: "=r"(nn),    // %0
							"=r"(ptr),   // %1
							"=r"(outptr) // %2
							: "0"(nn),
							"1"(ptr),
							"2"(outptr),
							"r"(scale) // %6
							: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
					}
#else
					if (nn > 0)
					{
						asm volatile(
							"pld        [%1, #256]          \n"
							"vld1.f32   {d0-d3}, [%1]!      \n"
							"vdup.32    q10, %6             \n"

							"0:                             \n"
							"vmul.f32   q0,q0,q10           \n"
							"vmul.f32   q1,q1,q10           \n"

							"vcvtr.s32.f32 s0,s0            \n"
							"vcvtr.s32.f32 s1,s1            \n"
							"vcvtr.s32.f32 s2,s2            \n"
							"vcvtr.s32.f32 s3,s3            \n"
							"vcvtr.s32.f32 s4,s4            \n"
							"vcvtr.s32.f32 s5,s5            \n"
							"vcvtr.s32.f32 s6,s6            \n"
							"vcvtr.s32.f32 s7,s7            \n"

							"vqmovn.s32 d4,q0               \n"
							"vqmovn.s32 d5,q1               \n"

							"pld        [%1, #256]          \n"
							"vld1.f32   {d0-d3}, [%1]!      \n"

							"vqmovn.s16 d4, q2              \n"
							"vst1.8     {d4}, [%2]!         \n"

							"subs       %0, #1              \n"
							"bne        0b                  \n"

							"sub        %1, #32             \n"
							: "=r"(nn),    // %0
							"=r"(ptr),   // %1
							"=r"(outptr) // %2
							: "0"(nn),
							"1"(ptr),
							"2"(outptr),
							"r"(scale) // %6
							: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11");
					}
#endif // __aarch64__
#endif // __ARM_NEON
					for (; remain > 0; remain--)
					{
						*outptr = float32_to_int8(*ptr * scale);

						ptr++;
						outptr++;
					}
				}
			}
		}

		template<typename Dtype>
		void operation_general_conv<Dtype>::dequantize_int32_to_float32(std::shared_ptr<memory::tensor<int>>& src,
			std::shared_ptr<memory::tensor<float>>& dst)
		{
			int num = src->num();
			int w = src->width();
			int h = src->height();
			int channels = src->channels();
			int size = w * h;
			const float* bias_data = nullptr;
			if (this->bias_term_)
				bias_data = this->weights_f32_[1]->cpu_data();

			for (size_t n = 0; n < num; n++)
			{
				const int* top_int32_data = src->cpu_data() + n * size * channels;
				float* top_f32_data = dst->mutable_cpu_data() + n * size * channels;

				if (this->bias_term_)
				{
#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
					for (int q = 0; q < channels; q++)
					{
						float scale = (std::fabs(this->weights_scaletable_i8_[q]) <= 1e-6) ? 0.f : (1.0f / (this->weights_scaletable_i8_[q] * this->featmap_scaletable_i8_[0]));
						const int* intptr = top_int32_data + q * size;
						float* ptr = top_f32_data + q * size;
						float bias = bias_data[q];

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"dup    v2.4s, %w6                   \n" // scale
								"dup    v3.4s, %w7                   \n" // bias
								"0:                                  \n"
								"prfm   pldl1keep, [%1, #128]        \n"
								"ld1    {v0.4s, v1.4s}, [%1], #32    \n" // data
								// top_s32 -> top_f32
								"scvtf  v5.4s, v0.4s                 \n"
								"scvtf  v6.4s, v1.4s                 \n"
								// top_f32 = top_f32 * scale_out
								"fmul   v5.4s, v5.4s, v2.4s          \n"
								"fmul   v6.4s, v6.4s, v2.4s          \n"
								// top_f32 = top_f32 + bias_tm
								"fadd   v5.4s, v5.4s, v3.4s          \n"
								"fadd   v6.4s, v6.4s, v3.4s          \n"
								// save top_f32
								"st1    {v5.4s, v6.4s}, [%2], #32    \n"
								"subs   %w0, %w0, #1                 \n"
								"bne    0b                           \n"
								: "=r"(nn),     // %0
								"=r"(intptr), // %1
								"=r"(ptr)     // %2
								: "0"(nn),
								"1"(intptr),
								"2"(ptr),
								"r"(scale), // %6
								"r"(bias)   // %7
								: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%1, #256]          \n"
								"vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
								"vdup.f32   q10, %6             \n" //q10 scale
								"vdup.f32   q12, %7             \n" //q12 bias

								"0:                             \n"
								"vcvt.f32.s32 q0, q0            \n"
								"vcvt.f32.s32 q1, q1            \n"

								"vmul.f32   q0,q0,q10           \n"
								"vmul.f32   q1,q1,q10           \n"

								"vadd.f32   q2,q0,q12           \n"
								"vadd.f32   q3,q1,q12           \n"

								"pld        [%1, #256]          \n"
								"vld1.s32   {d0-d3}, [%1]!      \n"
								"vst1.f32   {d4-d7}, [%2]!      \n"

								"subs       %0, #1              \n"
								"bne        0b                  \n"

								"sub        %1, #32             \n"
								: "=r"(nn),     // %0
								"=r"(intptr), // %1
								"=r"(ptr)     // %2
								: "0"(nn),
								"1"(intptr),
								"2"(ptr),
								"r"(scale), // %6
								"r"(bias)   // %7
								: "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12");
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							*ptr = *intptr * scale + bias;

							intptr++;
							ptr++;
						}
					}
				}
				else
				{
#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
					for (int q = 0; q < channels; q++)
					{
						float scale = (std::fabs(this->weights_scaletable_i8_[q]) <= 1e-6) ? 0.f : (1.0f / (this->weights_scaletable_i8_[q] * this->featmap_scaletable_i8_[0]));
						const int* intptr = top_int32_data + q * size;
						float* ptr = top_f32_data + q * size;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"dup    v2.4s, %w6                   \n" // scale
								"0:                                  \n"
								"prfm   pldl1keep, [%1, #128]      \n"
								"ld1    {v0.4s, v1.4s}, [%1], #32    \n" // data
								// top_s32 -> top_f32
								"scvtf  v5.4s, v0.4s                 \n"
								"scvtf  v6.4s, v1.4s                 \n"
								// top_f32 = top_f32 * scale_out
								"fmul   v5.4s, v5.4s, v2.4s          \n"
								"fmul   v6.4s, v6.4s, v2.4s          \n"
								// save top_f32
								"st1    {v5.4s, v6.4s}, [%2], #32    \n"
								"subs   %w0, %w0, #1                 \n"
								"bne    0b                           \n"
								: "=r"(nn),     // %0
								"=r"(intptr), // %1
								"=r"(ptr)     // %2
								: "0"(nn),
								"1"(intptr),
								"2"(ptr),
								"r"(scale) // %6
								: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%1, #256]          \n"
								"vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
								"vdup.f32   q10, %6             \n" //q10 scale

								"0:                             \n"
								"vcvt.f32.s32 q0, q0            \n"
								"vcvt.f32.s32 q1, q1            \n"

								"vmul.f32   q2,q0,q10           \n"
								"vmul.f32   q3,q1,q10           \n"

								"pld        [%1, #256]          \n"
								"vld1.s32   {d0-d3}, [%1]!      \n"
								"vst1.f32   {d4-d7}, [%2]!      \n"

								"subs       %0, #1              \n"
								"bne        0b                  \n"

								"sub        %1, #32             \n"
								: "=r"(nn),     // %0
								"=r"(intptr), // %1
								"=r"(ptr)     // %2
								: "0"(nn),
								"1"(intptr),
								"2"(ptr),
								"r"(scale) // %6
								: "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12");
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							*ptr = *intptr * scale;

							intptr++;
							ptr++;
						}
					}
				}
			}
		}

		INSTANCE_CLASS(operation_general_conv);
	}
}