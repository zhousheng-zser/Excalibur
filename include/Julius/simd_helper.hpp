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
	union union_type_s_mm128 {
		float s[4];
		__m128 v;
		__m64 t[2];
	};
#define q_type \
	union union_type_s_mm128
#define store_to_q(x,y)\
	_mm_store_ps(x,y)
#define mm_final_sum_quarter(q) (q.s[0])
#define mm_final_sum_half(q) (q.s[0]+q.s[1])
#define mm_final_sum_all(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3])
inline float _mm_sumall_ps(__m128 r)
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
	union union_type_s_mm256 {
		float s[8];
		__m256 v;
		__m128 p[2];
		__m64 t[4];
	};
#define q_type \
	union union_type_s_mm256

#define store_to_q(x,y)\
	_mm256_store_ps(x,y)

#define mm_final_sum_quarter(q) (q.s[0]+q.s[1])
#define mm_final_sum_half(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3])
#define mm_final_sum_all(q) (q.s[0]+q.s[1]+q.s[2]+q.s[3]+q.s[4]+q.s[5]+q.s[6]+q.s[7])
inline float _mm256_sumall_ps(__m256 r)
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
#define simd_registers 32
	union union_type_s_mm512 {
		float s[16];
		__m512 r;
		__m256 v[2];
		__m128 p[4];
		__m64 t[8];
	};
#endif
}
#endif // !_SIMD_HELPER_HPP_
