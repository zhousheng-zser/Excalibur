#pragma once
#ifndef _JULIUS_SAXPBY_HPP_
#define _JULIUS_SAXPBY_HPP_

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			inline void cblas_saxpby(const int N, const float alpha, const float* X,
				const int incX, const float beta, float* Y, const int incY)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				NOT_IMPLEMENTED;
#else SIMD_TYPE == SIMDTYPE_NONE
				// Fall back to native code
				for (size_t i = 0; i < N; i++)
				{
					Y[i * incY] = alpha * X[i * incX] + beta * Y[i * incY];
				}
#endif 
			}
		}
	}
}
#endif // !_JULIUS_SAXPBY_HPP_
