#pragma once
#ifndef _SIMD_HELPER_HPP_
#define _SIMD_HELPER_HPP_
#include <glasssix\accelerator.hpp>

namespace glasssix
{
#define NATIVE_CODE_WARNING LOG(WARNING) << "Unhandled Scenario: fall back to native code with extremely low performance."

#if SIMD_TYPE >= SIMDTYPE_SSE
#define mm_load_ps _mm_load_ps
#define mm_store_ps _mm_store_ps
#define mm_set1_ps _mm_set1_ps
#define mm_setzero_ps _mm_setzero_ps
#define mm_add_ps _mm_add_ps
#define mm_mul_ps _mm_mul_ps
#define mm_type __m128
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
	NOT_IMPLEMENTED;
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
#define mm_mul_ps _mm256_mul_ps
#define mm_type __m256
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
	const __m128i product_h = _mm256_extractf128_si256(product_int16, 1);
	const __m128i product_l = _mm256_extractf128_si256(product_int16, 0);
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
}
#endif // !_SIMD_HELPER_HPP_
