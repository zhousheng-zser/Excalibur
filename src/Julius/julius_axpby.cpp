#include "julius_axpby.hpp"
#include "../../include/Primitives/simd_types.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void cblas_saxpby(const int n, const float alpha, const float* x,
				const int incx, const float beta, float* y, const int incy)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				const mm_type alphas = mm_set1_ps(alpha);
				const mm_type betas = mm_set1_ps(beta);
				if (incy == 1)
				{
					if (incx == 1)
					{
						for (int i = 0; i < partn; i += mm_align_size)
						{
							const mm_type v_x = mm_load_ps(x + i);
							mm_type* v_y = (mm_type*)(y + i);
							*v_y = mm_add_ps(mm_mul_ps(v_x, alphas), mm_mul_ps(*v_y, betas));
						}
						for (int i = partn; i < n; i++)
						{
							y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
						}
					}
					else
					{
						for (int i = 0; i < partn; i += mm_align_size)
						{
							const int offset = i * mm_align_size;
							const mm_type v_x = _mm256_set_ps(x[(offset + 7) * incx], x[(offset + 6) * incx],
								x[(offset + 5) * incx], x[(offset + 4) * incx], x[(offset + 3) * incx],
								x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
							mm_type* v_y = (mm_type*)(y + i);
							*v_y = mm_add_ps(mm_mul_ps(v_x, alphas), mm_mul_ps(*v_y, betas));
						}
						for (int i = partn; i < n; i++)
						{
							y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
						}
					}
				}
				else
				{
					for (int i = 0; i < n; i++)
					{
						y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
					}
				}
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int restn = n % mm_align_size;
				const int partn = n - restn;
				const mm_type alphas = mm_set1_ps(alpha);
				const mm_type betas = mm_set1_ps(beta);
				if (incy == 1)
				{
					if (incx == 1)
					{
						for (int i = 0; i < partn; i += mm_align_size)
						{
							const mm_type v_x = mm_load_ps(x + i);
							mm_type* v_y = (mm_type*)(y + i);
							*v_y = mm_add_ps(mm_mul_ps(v_x, alphas), mm_mul_ps(*v_y, betas));
						}
						for (int i = partn; i < n; i++)
						{
							y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
						}
					}
					else
					{
						for (int i = 0; i < partn; i += mm_align_size)
						{
							const int offset = i * mm_align_size;
							const mm_type v_x = _mm_set_ps(x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
							mm_type* v_y = (mm_type*)(y + i);
							*v_y = mm_add_ps(mm_mul_ps(v_x, alphas), mm_mul_ps(*v_y, betas));
						}
						for (int i = partn; i < n; i++)
						{
							y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
						}
					}
				}
				else
				{
					for (int i = 0; i < n; i++)
					{
						y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
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
					y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
				}
#undef UNHANDLED
#endif
			}

			void cblas_daxpby(const int n, const double alpha, const double* x,
				const int incx, const double beta, double* y, const int incy)
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
					y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];
				}
#undef UNHANDLED
#endif
			}
		}
	}
}