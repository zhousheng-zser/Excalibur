#include "julius_gemm.hpp"
#include "julius_gemm_align.hpp"
#include "julius_axpby.hpp"
#include "Primitives/simd_types.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			template<typename Dtype>
			void packnoTransedA(const int M, const int K, const int padK, const Dtype* A, const int lda, Dtype* packedA)
			{
				memset(packedA, 0, M * padK * sizeof(Dtype));
				for (int i = 0; i < M; i++)
				{
					memcpy(packedA + i * padK, A + i * lda, K * sizeof(Dtype));
				}
			}

			template<typename Dtype>
			void packTransedA(const int M, const int K, const int padK, const Dtype* A, const int lda, Dtype* packedA)
			{
				memset(packedA, 0, M * padK * sizeof(Dtype));
				for (int j = 0; j < K; j++)
				{
					const int offsetA = j * lda;
					for (int i = 0; i < M; i++)
					{
						packedA[i * padK + j] = A[offsetA + i];
					}
				}
			}

			template<typename Dtype>
			void packnoTransedB(const int N, const int K, const int padK, const Dtype* B, const int ldb, Dtype* packedB)
			{
				memset(packedB, 0, N * padK * sizeof(Dtype));
				for (int j = 0; j < K; j++)
				{
					const int offsetB = j * ldb;
					for (int i = 0; i < N; i++)
					{
						packedB[i * padK + j] = B[offsetB + i];
					}
				}
			}

			template<typename Dtype>
			void packTransedB(const int N, const int K, const int padK, const Dtype* B, const int ldb, Dtype* packedB)
			{
				memset(packedB, 0, N * padK * sizeof(Dtype));
				for (int j = 0; j < N; j++)
				{
					memcpy(packedB + j * padK, B + j * ldb, K * sizeof(Dtype));
				}
			}

			template<typename Dtype>
			void packC(const int M, const int N, const int padN, Dtype* C, const int ldc, Dtype* packedC)
			{
				memset(packedC, 0, M * padN * sizeof(Dtype));
				for (int i = 0; i < M; i++)
				{
					memcpy(packedC + i * padN, C + i * ldc, N * sizeof(Dtype));
				}
			}

			template<typename Dtype>
			void unpackC(const int M, const int N, const int padN, Dtype* C, const int ldc, Dtype* packedC)
			{
				for (int i = 0; i < M; i++)
				{
					memcpy(C + i * ldc, packedC + i * padN, N * sizeof(Dtype));
				}
			}


			void cblas_sgemm_AnoTrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = (K + 7) >> 3 << 3;
				const int padN = (N + 7) >> 3 << 3;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				float* packedC = new float[M * padN];
				float* packed_copy_C = new float[M * padN];
				packnoTransedA(M, K, padK, A, lda, packedA);
				packnoTransedB(N, K, padK, B, ldb, packedB);
				packC(M, N, padN, C, ldc, packedC);
				memcpy(packed_copy_C, packedC, M * padN * sizeof(float));
				sgemm_AnoTrans_Btrans_auto(M, N, K, packedA, padK, packedB, padK, packedC, padN);
				cblas_saxpby(M * padN, alpha, packedC, 1, beta, packed_copy_C, 1);
				unpackC(M, N, padN, C, ldc, packed_copy_C);
				delete[] packedA;
				delete[] packedB;
				delete[] packedC;
				delete[] packed_copy_C;
#else 
#define UNHANDLED
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				for (int i = 0; i < N; i++)
				{
					for (int j = 0; j < M; j++)
					{
						C[j * ldc + i] = beta * C[j * ldc + i];
						for (int k = 0; k < K; k++)
						{
							C[j * ldc + i] += alpha * A[j * lda + k] * B[k * ldb + i];
						}
					}
				}
#undef UNHANDLED
#endif
			}

			void cblas_sgemm_ATrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = (K + 7) >> 3 << 3;
				const int padN = (N + 7) >> 3 << 3;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				float* packedC = new float[M * padN];
				float* packed_copy_C = new float[M * padN];
				packTransedA(M, K, padK, A, lda, packedA);
				packnoTransedB(N, K, padK, B, ldb, packedB);
				packC(M, N, padN, C, ldc, packedC);
				memcpy(packed_copy_C, packedC, M * padN * sizeof(float));
				sgemm_AnoTrans_Btrans_auto(M, N, K, packedA, padK, packedB, padK, packedC, padN);
				cblas_saxpby(M * padN, alpha, packedC, 1, beta, packed_copy_C, 1);
				unpackC(M, N, padN, C, ldc, packed_copy_C);
				delete[] packedA;
				delete[] packedB;
				delete[] packedC;
				delete[] packed_copy_C;
#else 
#define UNHANDLED
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}

			void cblas_sgemm_AnoTrans_BTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = (K + 7) >> 3 << 3;
				const int padN = (N + 7) >> 3 << 3;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				float* packedC = new float[M * padN];
				float* packed_copy_C = new float[M * padN];
				packnoTransedA(M, K, padK, A, lda, packedA);
				packTransedB(N, K, padK, B, ldb, packedB);
				packC(M, N, padN, C, ldc, packedC);
				memcpy(packed_copy_C, packedC, M * padN * sizeof(float));
				sgemm_AnoTrans_Btrans_auto(M, N, K, packedA, padK, packedB, padK, packedC, padN);
				cblas_saxpby(M * padN, alpha, packedC, 1, beta, packed_copy_C, 1);
				unpackC(M, N, padN, C, ldc, packed_copy_C);
				delete[] packedA;
				delete[] packedB;
				delete[] packedC;
				delete[] packed_copy_C;
#else 
#define UNHANDLED
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}

			void cblas_sgemm_ATrans_BTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				//NATIVE_CODE_WARNING;
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = (K + 7) >> 3 << 3;
				const int padN = (N + 7) >> 3 << 3;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				float* packedC = new float[M * padN];
				float* packed_copy_C = new float[M * padN];
				packTransedA(M, K, padK, A, lda, packedA);
				packTransedB(N, K, padK, B, ldb, packedB);
				packC(M, N, padN, C, ldc, packedC);
				memcpy(packed_copy_C, packedC, M * padN * sizeof(float));
				sgemm_AnoTrans_Btrans_auto(M, N, K, packedA, padK, packedB, padK, packedC, padN);
				cblas_saxpby(M * padN, alpha, packedC, 1, beta, packed_copy_C, 1);
				unpackC(M, N, padN, C, ldc, packed_copy_C);
				delete[] packedA;
				delete[] packedB;
				delete[] packedC;
				delete[] packed_copy_C;
#else 
#define UNHANDLED
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}

			void cblas_fgemm_AnoTrans_BnoTrans(const int M, const int N, const int K, const float alpha, const signed char* A, const int lda,
				const signed char* B, const int ldb, const float beta, int* C, const int ldc)
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
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
				// Fall back to native code
				for (int i = 0; i < N; i++)
				{
					for (int j = 0; j < M; j++)
					{
						C[j * ldc + i] = static_cast<int>(beta * C[j * ldc + i]);
						for (int k = 0; k < K; k++)
						{
							C[j * ldc + i] += static_cast<int>( alpha * A[j * lda + k] * B[k * ldb + i]);
						}
					}
				}
#undef UNHANDLED
#endif
			}

			void cblas_fgemm_ATrans_BnoTrans(const int M, const int N, const int K, const float alpha, const signed char* A, const int lda,
				const signed char* B, const int ldb, const float beta, int* C, const int ldc)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
#define UNHANDLED
				NATIVE_CODE_WARNING;
#else 
#define UNHANDLED
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}

			void cblas_fgemm_AnoTrans_BTrans(const int M, const int N, const int K, const float alpha, const signed char* A, const int lda,
				const signed char* B, const int ldb, const float beta, int* C, const int ldc)
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
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}

			void cblas_fgemm_ATrans_BTrans(const int M, const int N, const int K, const float alpha, const signed char* A, const int lda,
				const signed char* B, const int ldb, const float beta, int* C, const int ldc)
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
				//NATIVE_CODE_WARNING;
#endif 
#ifdef UNHANDLED
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
#undef UNHANDLED
#endif
			}
		}
	}
}