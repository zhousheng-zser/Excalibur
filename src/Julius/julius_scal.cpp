#include "julius_scal.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void cblas_sscal(const int n, const float alpha, float *x, const int incx)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				mm_type alphas = mm_set1_ps(alpha);
				if (incx == 1)
				{
					for (int i = 0; i < partn; i += mm_align_size)
					{
						mm_type* scal_val = (mm_type*)(x + i);
						*scal_val = mm_mul_ps(*scal_val, alphas);
					}
					for (int i = partn; i < n; i++)
					{
						x[i * incx] = alpha * x[i * incx];
					}
				}
				else
				{
					for (int i = 0; i < n; i++)
					{
						x[i * incx] = alpha * x[i * incx];
					}
				}
#else 
#define UNHANDLED
				NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				for (int i = 0; i < n; i++)
				{
					x[i * incx] = alpha * x[i * incx];
				}
#undef UNHANDLED
#endif
			}

			void cblas_dscal(const int n, const double alpha, double *x, const int incx)
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
				for (int i = 0; i < n; i++)
				{
					x[i * incx] = alpha * x[i * incx];
				}
#undef UNHANDLED
#endif
			}
		}
	}
}