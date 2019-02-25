#pragma once
#ifndef _JULIUS_GEMV_HPP_
#define _JULIUS_GEMV_HPP_
#include "simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			inline void cblas_sgemv_AnoTrans(const int M, const int N, const float alpha, const float  *A, const int lda, 
				const float  *x, const int incx, const float beta, float  *y, const int incy)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				NOT_IMPLEMENTED;
#else SIMD_TYPE == SIMDTYPE_NONE
				// Fall back to native code
				for (size_t j = 0; j < M; j++)
				{
					y[j * incy] = beta * y[j * incy];
					for (size_t i = 0; i < N; i++)
					{
						y[j * incy] += alpha * A[j * lda + i] * x[j * incx];
					}
				}
#endif 
			}

			inline void cblas_sgemv_ATrans(const int M, const int N, const float alpha, const float  *A, const int lda,
				const float  *x, const int incx, const float beta, float  *y, const int incy)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				NOT_IMPLEMENTED;
#else SIMD_TYPE == SIMDTYPE_NONE
				// Fall back to native code
				for (size_t j = 0; j < N; j++)
				{
					y[j * incy] = beta * y[j * incy];
					for (size_t i = 0; i < M; i++)
					{
						y[j * incy] += alpha * A[i * lda + j] * x[i * incx];
					}
				}
#endif 
			}
		}
	}
}
#endif // !_JULIUS_GEMV_HPP_
