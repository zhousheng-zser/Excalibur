#include "julius_sdot.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			float  cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy)
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
					sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
				}
				return static_cast<float>(sum +static_cast<double>(alpha));
#endif 
			}

			double cblas_dsdot(const int n, const float *x, const int incx, const float *y, const int incy)
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
					sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
				}
				return sum;
#endif 
			}
		}
	}
}