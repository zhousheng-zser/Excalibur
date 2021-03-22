#include "julius_sdot.hpp"
#include "Primitives/simd_types.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			float  cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
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
				double sum = 0;
				for (size_t i = 0; i < n; i++)
				{
					sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
				}
				return static_cast<float>(sum + static_cast<double>(alpha));
#undef UNHANDLED
#endif
			}

			double cblas_dsdot(const int n, const float *x, const int incx, const float *y, const int incy)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
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
				double sum = 0;
				for (size_t i = 0; i < n; i++)
				{
					sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
				}
				return sum;
#undef UNHANDLED
#endif
			}
		}
	}
}