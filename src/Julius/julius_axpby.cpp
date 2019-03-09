#include "julius_axpby.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void cblas_saxpby(const int N, const float alpha, const float* X,
				const int incX, const float beta, float* Y, const int incY)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define UNHANDLED
				NATIVE_CODE_WARNING;
#else 
#define UNHANDLED
				NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				for (int i = 0; i < N; i++)
				{
					Y[i * incY] = alpha * X[i * incX] + beta * Y[i * incY];
				}
#undef UNHANDLED
#endif
			}

			void cblas_daxpby(const int N, const double alpha, const double* X,
				const int incX, const double beta, double* Y, const int incY)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define UNHANDLED
				NATIVE_CODE_WARNING;
#else 
#define UNHANDLED
				NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				for (int i = 0; i < N; i++)
				{
					Y[i * incY] = alpha * X[i * incX] + beta * Y[i * incY];
				}
#undef UNHANDLED
#endif
			}
		}
	}
}