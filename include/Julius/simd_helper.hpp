#pragma once
#ifndef _SIMD_HELPER_HPP_
#define _SIMD_HELPER_HPP_
#include <glasssix/accelerator.hpp>

namespace glasssix
{
#define NATIVE_CODE_WARNING LOG(WARNING) << "Unhandled Scenario: fall back to native code with extremely low performance."


#if SIMD_TYPE >= SIMDTYPE_SSE
#define mm_load_ps _mm_load_ps
#define mm_store_ps _mm_store_ps
#define mm_set1_ps _mm_set1_ps
#define mm_setzero_ps _mm_setzero_ps
#define mm_add_ps _mm_add_ps
#define mm_sub_ps _mm_sub_ps
#define mm_mul_ps _mm_mul_ps
#define mm_type __m128
#define mm_typei __m128i
#define mm_round_ps _mm_round_ps
#define mm_load_si _mm_load_si128
#define mm_cvtepi32_ps _mm_cvtepi32_ps
#define mm_cvtps_epi32	_mm_cvtps_epi32
#define mm_cvtepu8_epi32 _mm_cvtepu8_epi32
#define mm_cvtepi8_epi32 _mm_cvtepi8_epi32
#define mm_cvtepi16_epi32 _mm_cvtepi16_epi32
#define mm_mullo_epi32 _mm_mullo_epi32
#define mm_store_si _mm_store_si128
#define mm_setzero_si _mm_setzero_si128
#define mm_add_epi32 _mm_add_epi32
#define mm_align_size 4
#define mm_align_size2 8
#define mm_align_size3 12
#define mm_align_size4 16
#define mm_align_size5 20
#define mm_align_size6 24
#define mm_align_size7 28
#define mm_align_size8 32
#define simd_registers 16
	__forceinline __m128i _mm_madd_epi16_epi32(const __m128i a, const __m128i b, __m128i c)
{
	const __m128i a_int16 = _mm_cvtepi8_epi16(a);
	const __m128i b_int16 = _mm_cvtepi8_epi16(b);
	const __m128i product_int16 = _mm_mullo_epi16(a_int16, b_int16);
	short product_int16_array[8];
	_mm_store_si128((__m128i*)product_int16_array, product_int16);
	const __m128i product_h = _mm_load_si128((__m128i*)product_int16_array);
	const __m128i product_l = _mm_load_si128((__m128i*)(product_int16_array + 4));
	const __m128i product_h_int32 = _mm_cvtepi16_epi32(product_h);
	const __m128i product_l_int32 = _mm_cvtepi16_epi32(product_l);
	return _mm_add_epi32(_mm_add_epi32(product_l_int32, product_h_int32), c);
}
	
	__forceinline int _mm_sumall_epi32(const __m128i re)
{
	int temp_sum[4];
	_mm_store_si128((__m128i*)temp_sum, re);
	return temp_sum[0] + temp_sum[1] + temp_sum[2] + temp_sum[3];
}
	union union_type_s_mm128 
{
	double d[2];
	float s[4];
	__m128 v;
	__m64 t[2];
};
#define q_type \
	union union_type_s_mm128
#define store_to_q(x,y)\
	_mm_store_ps(x,y)
#define mm_final_ssum_quarter(q) (q.s[0])
#define mm_final_ssum_half(q) (q.s[0]+q.s[1])
#define mm_final_ssum_all(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3])
	__forceinline float _mm_sumall_ps(__m128 r)
	{
		union_type_s_mm128 q = { 0 };
		_mm_store_ps(q.s, r);
		return (q.s[0] + q.s[1]) + (q.s[2] + q.s[3]);
	}
#define mm_sumall_ps _mm_sumall_ps
#if USE_FMADD128
#define mm_fmadd_ps _mm_fmadd_ps
#else
#define mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#endif

#if SIMD_TYPE >= SIMDTYPE_AVX
#define mm_load_ps _mm256_load_ps
#define mm_store_ps _mm256_store_ps
#define mm_set1_ps _mm256_set1_ps
#define mm_setzero_ps _mm256_setzero_ps
#define mm_add_ps _mm256_add_ps
#define mm_sub_ps _mm256_sub_ps
#define mm_mul_ps _mm256_mul_ps
#define mm_type __m256
#define mm_typei __m256i
#define mm_round_ps _mm256_round_ps
#define mm_load_si _mm256_load_si256
#define mm_cvtepi32_ps _mm256_cvtepi32_ps
#define mm_cvtps_epi32	_mm256_cvtps_epi32
#define mm_cvtepu8_epi32 _mm256_cvtepu8_epi32
#define mm_cvtepi8_epi32 _mm256_cvtepi8_epi32
#define mm_cvtepi16_epi32 _mm256_cvtepi16_epi32
#define mm_mullo_epi32 _mm256_mullo_epi32
#define mm_store_si _mm256_store_si256
#define mm_setzero_si _mm256_setzero_si256
#define mm_add_epi32 _mm256_add_epi32
#define mm_align_size 8
#define mm_align_size2 16
#define mm_align_size3 24
#define mm_align_size4 32
#define mm_align_size5 40
#define mm_align_size6 48
#define mm_align_size7 56
#define mm_align_size8 64
#define simd_registers 16

	__forceinline __m256i _mm256_madd_epi16_epi32(const __m128i a, const __m128i b, __m256i c)
{
	const __m256i a_int16 = _mm256_cvtepi8_epi16(a);
	const __m256i b_int16 = _mm256_cvtepi8_epi16(b);
	const __m256i product_int16 = _mm256_mullo_epi16(a_int16, b_int16);
	/*const __m128i product_h = _mm256_extractf128_si256(product_int16, 1);
	const __m128i product_l = _mm256_extractf128_si256(product_int16, 0);*/
	short product_int16_array[16];
	_mm256_store_si256((__m256i*)product_int16_array, product_int16);
	const __m128i product_h = _mm_load_si128((__m128i*)product_int16_array);
	const __m128i product_l = _mm_load_si128((__m128i*)(product_int16_array + 8));
	const __m256i product_h_int32 = _mm256_cvtepi16_epi32(product_h);
	const __m256i product_l_int32 = _mm256_cvtepi16_epi32(product_l);
	return _mm256_add_epi32(_mm256_add_epi32(product_l_int32, product_h_int32), c);
}
	
	__forceinline int _mm256_sumall_epi32(const __m256i re)
{
	int temp_sum[8];
	_mm256_store_si256((__m256i*)temp_sum, re);
	return temp_sum[0] + temp_sum[1] + temp_sum[2] + temp_sum[3] + temp_sum[4] + temp_sum[5] + temp_sum[6] + temp_sum[7];
}
	
	union union_type_s_mm256 
{
	double d[4];
	float s[8];
	__m256 v;
	__m128 p[2];
	__m64 t[4];
};
#define q_type \
	union union_type_s_mm256

#define store_to_q(x,y)\
	_mm256_store_ps(x,y)

#define mm_final_ssum_quarter(q) (q.s[0]+q.s[1])
#define mm_final_ssum_half(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3])
#define mm_final_ssum_all(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3]+q.s[4]+q.s[5]+q.s[6]+q.s[7])
	__forceinline float _mm256_sumall_ps(__m256 r)
	{
		__m128 h = _mm256_extractf128_ps(r, 1);
		__m128 l = _mm256_extractf128_ps(r, 0);
		h = _mm_add_ps(h, l);
		return _mm_sumall_ps(h);
	}
#define mm_sumall_ps _mm256_sumall_ps
#if USE_FMADD256
#define mm_fmadd_ps _mm256_fmadd_ps
#else
#define mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif// !USE_FMADD256
#endif

#if SIMD_TYPE >= SIMDTYPE_AVX512
#define mm_load_ps _mm512_load_ps
#define mm_store_ps _mm512_store_ps
#define mm_set1_ps _mm512_set1_ps
#define mm_setzero_ps _mm512_setzero_ps
#define mm_add_ps _mm512_add_ps
#define mm_mul_ps _mm512_mul_ps
#define mm_type __m512
#define mm_align_size 16
#define mm_align_size2 32
#define mm_align_size3 48
#define mm_align_size4 64
#define mm_align_size5 80
#define mm_align_size6 96
#define mm_align_size7 112
#define mm_align_size8 128
#define simd_registers 32
	union union_type_s_mm512 
	{
		double d[8];
		float s[16];
		__m512 r;
		__m256 v[2];
		__m128 p[4];
		__m64 t[8];
	};
#define q_type \
	union union_type_s_mm512

#define store_to_q(x,y)\
	_mm512_store_ps(x,y)

#define mm_final_ssum_quarter(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3])
#define mm_final_ssum_half(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3]+q.s[4]+q.s[5]+q.s[6]+q.s[7])
#define mm_final_ssum_all(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3]+q.s[4]+q.s[5]+q.s[6]+q.s[7] + \
	q.s[8]+q.s[9]+q.s[10]+q.s[11]+q.s[12]+q.s[13]+q.s[14]+q.s[15])
	__forceinline float _mm512_sumall_ps(__m512 r)
	{
		__m256 h = _mm512_extractf32x8_ps(r, 1);
		__m256 l = _mm512_extractf32x8_ps(r, 0);
		h = _mm256_add_ps(h, l);
		return _mm256_sumall_ps(h);
	}
#define mm_sumall_ps _mm512_sumall_ps
#define mm_fmadd_ps _mm512_fmadd_ps
#endif

#ifndef mm_align_size
#define mm_align_size 1
#endif // !mm_align_size

	// for CUDA, provide device function: unsigned short f16 = __float2half_rn( value );
	// convert float to half precision floating point
	static unsigned short float32_to_float16(float value)
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
	static float float16_to_float32(unsigned short value)
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

	// round to nearest
	static signed char float32_to_int8(float value)
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

	inline void float2half(const float* floats, unsigned short* halfs, int length)
	{
		const int restl = length - length % mm_align_size;
		const int partl = length - restl;
		for (int i = 0; i < partl; i = i + mm_align_size)
		{
#if SIMD_TYPE >= SIMDTYPE_AVX512
			const __m512 float_vector = _mm512_load_ps(floats + i);
			const __m256i half_vector = _mm512_cvtps_ph(float_vector, 0);
			_mm256_store_si256((__m256i*)(halfs + i), half_vector);
#elif SIMD_TYPE >= SIMDTYPE_AVX
			const __m256 float_vector = _mm256_load_ps(floats + i);
			const __m128i half_vector = _mm256_cvtps_ph(float_vector, 0);
			_mm_store_si128((__m128i*)(halfs + i), half_vector);
#elif SIMD_TYPE >= SIMDTYPE_SSE
			const __m128 float_vector = _mm_load_ps(floats + i);
			const __m128i half_vector = _mm_cvtps_ph(float_vector, 0);
			_mm_store_si128((__m128i*)(halfs + i), half_vector);
#endif
		}
		for (int i = partl; i < length; i++)
		{
			halfs[i] = float32_to_float16(floats[i]);
		}
	}

	inline void half2float(const unsigned short* halfs, float* floats, int length)
	{
		const int restl = length - length % mm_align_size;
		const int partl = length - restl;
		for (int i = 0; i < partl; i = i + mm_align_size)
		{
#if SIMD_TYPE >= SIMDTYPE_AVX512
			const __m256i half_vector = _mm256_load_si256((__m256i*)(halfs + i));
			const __m256 float_vector = _mm512_cvtph_ps(half_vector);
			_mm512_store_ps(floats + i, float_vector);
#elif SIMD_TYPE >= SIMDTYPE_AVX
			const __m128i half_vector = _mm_load_si128((__m128i*)(halfs + i));
			const __m256 float_vector = _mm256_cvtph_ps(half_vector);
			_mm256_store_ps(floats + i, float_vector);
#elif SIMD_TYPE >= SIMDTYPE_SSE
			const __m128i half_vector = _mm_load_si128((__m128i*)(halfs + i));
			const __m128 float_vector = _mm_cvtph_ps(half_vector);
			_mm_store_ps(floats + i, float_vector);
#endif
		}
		for (int i = partl; i < length; i++)
		{
			floats[i] = float16_to_float32(halfs[i]);
		}
	}

	inline void int8_to_float(const signed char* int8_data, const float* scales, float* floats, int num, int group)
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
}
#endif // !_SIMD_HELPER_HPP_
