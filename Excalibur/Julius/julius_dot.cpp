#include "julius_dot.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace  juliusblas
		{
			float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				NOT_IMPLEMENTED;
#else SIMD_TYPE == SIMDTYPE_NONE
				// Fall back to native code
				float sum = 0;
				for (size_t i = 0; i < n; i++)
				{
					sum += x[i * incx] * y[i * incy];
				}
				return sum;
#endif 
			}

			double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				NOT_IMPLEMENTED;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				NOT_IMPLEMENTED;
#else SIMD_TYPE == SIMDTYPE_NONE
				// Fall back to native code
				double sum = 0;
				for (size_t i = 0; i < n; i++)
				{
					sum += x[i * incx] * y[i * incy];
				}
				return sum;
#endif 
			}
		}
	}
}