#pragma once
#ifndef _SIMD_HELPER_HPP_
#define _SIMD_HELPER_HPP_
#include <glasssix\accelerator.hpp>

namespace glasssix
{
#if SIMD_TYPE >= SIMDTYPE_SSE
	union union_type_s_mm128 {
		float s[4];
		__m128 v;
		__m64 t[2];
	};
#endif

#if SIMD_TYPE >= SIMDTYPE_AVX
	union union_type_s_mm256 {
		float s[8];
		__m256 v;
		__m128 p[2];
		__m64 t[4];
	};
#endif

#if SIMD_TYPE >= SIMDTYPE_AVX512
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
