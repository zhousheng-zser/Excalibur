#include "julius_asum.hpp"
#include "../../include/Primitives/simd_types.hpp"
#include <cmath>

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			float cblas_sasum(const int n, const float *x, const int incx)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
				float sum = 0.0f;
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				mm_type re = mm_setzero_ps();
				for (int i = 0; i < partn; i += mm_align_size)
				{
					mm_type val_x;
					if (incx == 1)
					{
						val_x = mm_load_ps(x + i);
					}
					else
					{
						const int offset = i * mm_align_size;
						val_x = _mm512_set_ps(x[(offset + 15) * incx], x[(offset + 14) * incx],
							x[(offset + 13) * incx], x[(offset + 12) * incx], x[(offset + 11) * incx],
							x[(offset + 10) * incx], x[(offset + 9) * incx], x[(offset + 8) * incx],
							x[(offset + 7) * incx], x[(offset + 6) * incx], x[(offset + 5) * incx],
							x[(offset + 4) * incx], x[(offset + 3) * incx], x[(offset + 2) * incx],
							x[(offset + 1) * incx], x[(offset + 0) * incx]);
					}
					re = _mm512_add_ps(_mm512_abs_ps(val_x), re);
				}
				sum = mm_sumall_ps(re);
				for (int i = partn; i < n; i++)
				{
					sum += x[i * incx] > 0 ? x[i * incx] : -1.0f * x[i * incx];
				}
				return sum;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				float sum = 0.0f;
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				mm_type re = mm_setzero_ps();
				for (int i = 0; i < partn; i += mm_align_size)
				{
					const mm_type mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
					mm_type val_x;
					if (incx == 1)
					{
						val_x = mm_load_ps(x + i);
					}
					else
					{
						const int offset = i * mm_align_size;
						val_x = _mm256_set_ps(x[(offset + 7) * incx], x[(offset + 6) * incx],
							x[(offset + 5) * incx], x[(offset + 4) * incx], x[(offset + 3) * incx],
							x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
					}
					// MSVC and INTEL COMPILER is BAD when inlining '_mm256_andnot_ps' into loops?
					// ref: https://stackoverflow.com/questions/32408665/fastest-way-to-compute-absolute-value-using-sse
					re = _mm256_add_ps(_mm256_andnot_ps(mask, val_x), re);
				}
				sum = mm_sumall_ps(re);
				for (int i = partn; i < n; i++)
				{
					sum += x[i * incx] > 0 ? x[i * incx] : -1.0f * x[i * incx];
				}
				return sum;
#elif SIMD_TYPE >= SIMDTYPE_SSE
				float sum = 0.0f;
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				mm_type re = mm_setzero_ps();
				for (int i = 0; i < partn; i += mm_align_size)
				{
					const mm_type mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
					mm_type val_x;
					if (incx == 1)
					{
						val_x = mm_load_ps(x + i);
					}
					else
					{
						const int offset = i * mm_align_size;
						val_x = _mm_set_ps(x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
					}
					// MSVC and INTEL COMPILER is BAD when inlining '_mm256_andnot_ps' into loops?
					// ref: https://stackoverflow.com/questions/32408665/fastest-way-to-compute-absolute-value-using-sse
					re = _mm_add_ps(_mm_andnot_ps(mask, val_x), re);
				}
				sum = mm_sumall_ps(re);
				for (int i = partn; i < n; i++)
				{
					sum += x[i * incx] > 0 ? x[i * incx] : -1.0f * x[i * incx];
				}
				return sum;
#else 
#define UNHANDLED
				NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				float sum = 0.0f;
				for (int i = 0; i < n; i++)
				{
					sum += std::abs(x[i * incx]);
				}
				return sum;
#undef UNHANDLED
#endif
			}

			double cblas_dasum(const int n, const double *x, const int incx)
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
				double sum = 0.0f;
				for (int i = 0; i < n; i++)
				{
					sum += std::abs(x[i * incx]);
				}
				return sum;
#undef UNHANDLED
#endif
			}
		}
	}
}