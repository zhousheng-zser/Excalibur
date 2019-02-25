#pragma once
#ifndef _JULIUS_GEMM_HPP_
#define _JULIUS_GEMM_HPP_

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			inline void cblas_sgemm_AnoTrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
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
					for (size_t j = 0; j < M; j++)
					{
						C[j * ldc + i] = beta * C[j * ldc + i];
						for (size_t k = 0; k < K; k++)
						{
							C[j * ldc + i] += alpha * A[j * lda + k] * B[k * ldb + i];
						}
					}
				}
#endif 
			}

			inline void cblas_sgemm_ATrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
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
					for (size_t j = 0; j < M; j++)
					{
						C[j * ldc + i] = beta * C[j * ldc + i];
						for (size_t k = 0; k < K; k++)
						{
							C[j * ldc + i] += alpha * A[k * lda + j] * B[k * ldb + i];
						}
					}
				}
#endif 
			}

			inline void cblas_sgemm_AnoTrans_BTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
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
					for (size_t j = 0; j < M; j++)
					{
						C[j * ldc + i] = beta * C[j * ldc + i];
						for (size_t k = 0; k < K; k++)
						{
							C[j * ldc + i] += alpha * A[j * lda + k] * B[i * ldb + k];
						}
					}
				}
#endif 
			}

			inline void cblas_sgemm_ATrans_BTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
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
					for (size_t j = 0; j < M; j++)
					{
						C[j * ldc + i] = beta * C[j * ldc + i];
						for (size_t k = 0; k < K; k++)
						{
							C[j * ldc + i] += alpha * A[k * lda + j] * B[i * ldb + k];
						}
					}
				}
#endif 
			}
		};
	}
}
 
#endif // !_JULIUS_GEMM_HPP_
