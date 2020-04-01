#include "julius_dot.hpp"
#include "../../include/Primitives/simd_types.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace  juliusblas
		{
			float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				const int restN = n % mm_align_size;
				const int partN = n - restN;
				mm_type re = mm_setzero_ps();
				float sum = 0.0f;
				for (int i = 0; i < partN / mm_align_size; i++)
				{
					mm_type mx;
					mm_type my;
					int offset = i * mm_align_size;
					if (incx == 1)
					{
						mx = mm_load_ps(x + offset);
					}
					else
					{
						mx = _mm256_set_ps(x[(offset + 7) * incx], x[(offset + 6) * incx],
							x[(offset + 5) * incx], x[(offset + 4) * incx], x[(offset + 3) * incx],
							x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
					}
					if (incy == 1)
					{
						my = mm_load_ps(y + offset);
					}
					else
					{
						my = _mm256_set_ps(y[(offset + 7) * incy], y[(offset + 6) * incy],
							y[(offset + 5) * incy], x[(offset + 4) * incy], y[(offset + 3) * incy],
							y[(offset + 2) * incy], y[(offset + 1) * incy], y[(offset + 0) * incy]);
					}
					re = mm_fmadd_ps(mx, my, re);
					sum = _mm256_sumall_ps(re);
				}
				for (int i = partN; i < n; i++)
				{
					sum += x[i * incx] * y[i * incy];
				}
				return sum;
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define UNHANDLED
				NATIVE_CODE_WARNING;
#else 
#define UNHANDLED
				NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				float sum = 0;
				for (int i = 0; i < n; i++)
				{
					sum += x[i * incx] * y[i * incy];
				}
				return sum;
#undef UNHANDLED
#endif
			}

			double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy)
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
				for (int i = 0; i < n; i++)
				{
					sum += x[i * incx] * y[i * incy];
				}
				return sum;
#undef UNHANDLED
#endif
			}
		}
	}
}