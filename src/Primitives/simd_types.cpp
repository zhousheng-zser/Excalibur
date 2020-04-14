#include "simd_types.hpp"
#include "cpu.hpp"

namespace glasssix
{
	// for CUDA, provide device function: unsigned short f16 = __float2half_rn( value );
	// convert float to half precision floating point
	unsigned short float32_to_float16(float value)
	{
		// 1 : 8 : 23
		union
		{
			unsigned int u;
			float f;
		} tmp;

		tmp.f = value;

		// 1 : 8 : 23
		unsigned short sign = (tmp.u & 0x80000000) >> 31;
		unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
		unsigned int significand = tmp.u & 0x7FFFFF;

		//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

			// 1 : 5 : 10
		unsigned short fp16;
		if (exponent == 0)
		{
			// zero or denormal, always underflow
			fp16 = (sign << 15) | (0x00 << 10) | 0x00;
		}
		else if (exponent == 0xFF)
		{
			// infinity or NaN
			fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
		}
		else
		{
			// normalized
			short newexp = exponent + (-127 + 15);
			if (newexp >= 31)
			{
				// overflow, return infinity
				fp16 = (sign << 15) | (0x1F << 10) | 0x00;
			}
			else if (newexp <= 0)
			{
				// underflow
				if (newexp >= -10)
				{
					// denormal half-precision
					unsigned short sig = (significand | 0x800000) >> (14 - newexp);
					fp16 = (sign << 15) | (0x00 << 10) | sig;
				}
				else
				{
					// underflow
					fp16 = (sign << 15) | (0x00 << 10) | 0x00;
				}
			}
			else
			{
				fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
			}
		}

		return fp16;
	}

	// for CUDA, provide device function: float f32 = __half2float( value );
	// convert half precision floating point to float
	float float16_to_float32(unsigned short value)
	{
		// 1 : 5 : 10
		unsigned short sign = (value & 0x8000) >> 15;
		unsigned short exponent = (value & 0x7c00) >> 10;
		unsigned short significand = value & 0x03FF;

		//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

			// 1 : 8 : 23
		union
		{
			unsigned int u;
			float f;
		} tmp;
		if (exponent == 0)
		{
			if (significand == 0)
			{
				// zero
				tmp.u = (sign << 31);
			}
			else
			{
				// denormal
				exponent = 0;
				// find non-zero bit
				while ((significand & 0x200) == 0)
				{
					significand <<= 1;
					exponent++;
				}
				significand <<= 1;
				significand &= 0x3FF;
				tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
			}
		}
		else if (exponent == 0x1F)
		{
			// infinity or NaN
			tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
		}
		else
		{
			// normalized
			tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
		}

		return tmp.f;
	}

	unsigned short float32_to_bfloat16(float value)
	{
		// 16 : 16
		union { unsigned int u; float f; } tmp;
		tmp.f = value;
		return tmp.u >> 16;
	}

	float bfloat16_to_float32(unsigned short value)
	{
		// 16 : 16
		union { unsigned int u; float f; } tmp;
		tmp.u = value << 16;
		return tmp.f;
	}

	signed char float32_to_int8(float value)
	{
		float tmp;
		if (value >= 0.f) tmp = value + 0.5;
		else tmp = value - 0.5;

		if (tmp > 127)
			return 127;
		if (tmp < -128)
			return -128;

		return tmp;
	}

	void int8_to_float(const signed char* int8_data, const float* scales, float* floats, int num, int group)
	{
		if (num % group != 0)
		{
			LOG(FATAL) << "int8_data does not match group!!!";
			return;
		}
		int offset = num / group;

		for (int i = 0; i < offset; i++)
		{
			for (int j = 0; j < group; j++)
			{
				floats[j * offset + i] = int8_data[j * offset + i] / scales[j];
			}
		}
	}

	void float2half(const float* floats, unsigned short* halfs, int length)
	{
		int f16c = (int)support_F16C();//check f16c instruction support
		const int restl = (length - length % mm_align_size) * f16c;
		const int partl = length - restl;
		for (int i = 0; i < partl; i = i + mm_align_size)
		{
#if SIMD_X86_INSTR_SET > SIMD_X86_AVX2_VERSION
			const __m512 float_vector = _mm512_load_ps(floats + i);
			const __m256i half_vector = _mm512_cvtps_ph(float_vector, 0);
			_mm256_store_si256((__m256i*)(halfs + i), half_vector);
#elif SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION
			const __m256 float_vector = _mm256_load_ps(floats + i);
			const __m128i half_vector = _mm256_cvtps_ph(float_vector, 0);
			_mm_store_si128((__m128i*)(halfs + i), half_vector);
#elif SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION
			for (int j = 0; j < mm_align_size; j++)
			{
				halfs[i + j] = float32_to_float16(floats[i + j]);
			}
#endif
		}
		for (int i = partl; i < length; i++)
		{
			halfs[i] = float32_to_float16(floats[i]);
		}
	}

	void half2float(const unsigned short* halfs, float* floats, int length)
	{
		int f16c = (int)support_F16C();//check f16c instruction support
		const int restl = (length - length % mm_align_size) * f16c;
		const int partl = length - restl;
		for (int i = 0; i < partl; i = i + mm_align_size)
		{
#if SIMD_X86_INSTR_SET > SIMD_X86_AVX2_VERSION
			const __m256i half_vector = _mm256_load_si256((__m256i*)(halfs + i));
			const __m256 float_vector = _mm512_cvtph_ps(half_vector);
			_mm512_store_ps(floats + i, float_vector);
#elif SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION
			const __m128i half_vector = _mm_load_si128((__m128i*)(halfs + i));
			const __m256 float_vector = _mm256_cvtph_ps(half_vector);
			_mm256_store_ps(floats + i, float_vector);
#elif SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION
			for (int j = 0; j < mm_align_size; j++)
			{
				floats[i + j] = float16_to_float32(halfs[i + j]);
			}
#endif
		}
		for (int i = partl; i < length; i++)
		{
			floats[i] = float16_to_float32(halfs[i]);
		}
	}

	float mul_add_3x3_native(const float *r0, const float *r1, const float *r2, const float *k0, const float *k1, const float *k2, float bias)
	{
		float sum = bias;
		sum += r0[0] * k0[0];
		sum += r0[1] * k0[1];
		sum += r0[2] * k0[2];
		sum += r1[0] * k1[0];
		sum += r1[1] * k1[1];
		sum += r1[2] * k1[2];
		sum += r2[0] * k2[0];
		sum += r2[1] * k2[1];
		sum += r2[2] * k2[2];

		return sum;
	}
}