#include "julius_gemm.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void packAnoTrans(const int M, const int K, const int padK, const float* A, const int lda, float* packedA)
			{
				memset(packedA, 0, M * padK * sizeof(float));
				for (int i = 0; i < M; i++)
				{
					memcpy(packedA + i * padK, A + i * lda, K * sizeof(float));
				}
			}

			void packBnoTrans(const int N, const int K, const int padK, const float* B, const int ldb, float* packedB)
			{
				memset(packedB, 0, N * padK * sizeof(float));
				for (int j = 0; j < K; j++)
				{
					const int offsetB = j * ldb;
					for (int i = 0; i < N; i++)
					{
						packedB[i * padK + j] = B[offsetB + i];
					}
				}
			}

			inline void adddot1x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, const float beta, float* C_ptr, const int ldc)
			{
				mm_type re = mm_setzero_ps();
				for (int j = 0; j < padK; j += mm_align_size)
				{
					const mm_type a = mm_load_ps(packedA_ptr + j);
					const mm_type b = mm_load_ps(packedB_ptr + j);
					re = mm_fmadd_ps(a, b, re);
				}
#if SIMD_TYPE == SIMDTYPE_SSE
				*C_ptr = alpha * _mm_sumall_ps(re) + beta * *C_ptr;
#endif
#if SIMD_TYPE >= SIMDTYPE_AVX
				* C_ptr = alpha * _mm256_sumall_ps(re) + beta * *C_ptr;
#endif
			}

			inline void adddot1x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr + padK, beta, C_ptr + 1, ldc);
			}

			inline void adddot2x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x1(padK, alpha,  packedA_ptr + padK, packedB_ptr, beta, C_ptr + ldc, ldc);
			}

			inline void adddot1x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr + padK * 2, beta, C_ptr + 2, ldc);
			}

			inline void adddot2x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x2(padK, alpha,  packedA_ptr + padK, packedB_ptr, beta, C_ptr + ldc, ldc);
			}

			inline void adddot4x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot2x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot2x1(padK, alpha,  packedA_ptr + padK * 2, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			}

			inline void adddot1x8(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x4(padK, alpha,  packedA_ptr, packedB_ptr + padK * 4, beta, C_ptr + 4, ldc);
			}

			inline void adddot2x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr + padK * 2, beta, C_ptr + 2, ldc);
			}

			inline void adddot4x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot2x2(padK, alpha,  packedA_ptr + padK * 2, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			}

			inline void adddot8x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot4x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot4x1(padK, alpha,  packedA_ptr + padK * 4, packedB_ptr, beta, C_ptr + 4 * ldc, ldc);
			}

			inline void adddot1x16(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot1x8(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot1x8(padK, alpha,  packedA_ptr, packedB_ptr + padK * 8, beta, C_ptr + 8, ldc);
			}

			inline void adddot2x8(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr + padK * 4, beta, C_ptr + 4, ldc);
			}

			inline void adddot4x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot2x4(padK, alpha,  packedA_ptr + 2 * padK, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			}

			inline void adddot8x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot4x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot4x2(padK, alpha,  packedA_ptr + padK * 4, packedB_ptr, beta, C_ptr + 4 * ldc, ldc);
			}

			inline void adddot16x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				adddot8x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
				adddot8x1(padK, alpha,  packedA_ptr + padK * 8, packedB_ptr, beta, C_ptr + 8 * ldc, ldc);
			}

			void cblas_sgemm_AnoTrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
				const int padK = K % mm_align_size ? K - K % mm_align_size + mm_align_size: K;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				packAnoTrans(M, K, padK, A, lda, packedA);
				packBnoTrans(N, K, padK, B, ldb, packedB);

				bool flag_M = true, flag_N = true;
				const int num_M1 = M / 4, num_N1 = N / 4, remain_M1 = M % 4, remain_N1 = N % 4;
				int num_M2, num_N2, remain_M2, remain_N2;
				if (remain_M1 != 0) { flag_M = false; }
				if (remain_N1 != 0) { flag_N = false; }
				if(flag_M && flag_N) 
				{
					for (int n_row = 0; n_row < num_M1; n_row ++)
					{
						for (int n_col = 0; n_col < num_N1; n_col ++)
						{
							adddot4x4(padK, alpha, packedA + n_row * 4 * padK, packedB + n_col * 4 * padK, beta, C + n_row * 4 * ldc + n_col * 4, ldc);
						}
					}
				}
				else if(flag_M && !flag_N)
				{
					num_N2 = remain_N1 / 2;
					remain_N2 = remain_N1 % 2;
					flag_N = true;
					if (remain_N2 != 0) { flag_N = false; }

					for (int n_row = 0; n_row < num_M1; n_row++)
					{
						for (int n_col = 0; n_col < num_N1; n_col++)
						{
							adddot4x4(padK, alpha, packedA + n_row * 4 * padK, packedB + n_col * 4 * padK, beta, C + n_row * 4 * ldc + n_col * 4, ldc);
						}
					}
					//deal with remain_N1
					{
						int row = 0;
						for (; row + 8 < M; row += 8)
						{
							if (!num_N2)
							{
								adddot8x2(padK, alpha, packedA + row * padK, packedB + num_N1 * 4 * padK, beta, C + row * ldc + num_N1 * 4, ldc);
							}
						}
						if(row < M - 1 && !num_N2)
						{
							adddot4x2(padK, alpha, packedA + row * padK, packedB + num_N1 * 4 * padK, beta, C + row * ldc + num_N1 * 4, ldc);
						}

						if (!flag_N)
						{
							row = 0;
							for (; row + 16 < M; row += 16)
							{
								adddot16x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}
							for (; row + 8 < M; row += 8)
							{
								adddot8x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}

							if(row < M-1)
							{
								adddot4x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}
						}
					}
				}
				else if(!flag_M && flag_N)
				{
					num_M2 = remain_M1 / 2;
					remain_M2 = remain_M1 % 2;
					flag_M = true;
					if (remain_M2 != 0) { flag_M = false; }

					for (int n_row = 0; n_row < num_M1; n_row++)
					{
						for (int n_col = 0; n_col < num_N1; n_col++)
						{
							adddot4x4(padK, alpha, packedA + n_row * 4 * padK, packedB + n_col * 4 * padK, beta, C + n_row * 4 * ldc + n_col * 4, ldc);
						}
					}

					//deal with remain_M1
					{
						int col = 0;
						for (; col + 8 < N; col += 8)
						{
							if(num_M2)
							{
								adddot2x8(padK, alpha, packedA + num_M1 * 4 * padK, packedB + col * padK, beta, C + num_M1 * 4 * ldc + col, ldc);
							}
						}

						if(num_M2)
						{
							adddot2x4(padK, alpha, packedA + num_M1 * 4 * padK, packedB + col * padK, beta, C + num_M1 * 4 * ldc + col, ldc);
						}

						if (!flag_M)
						{
							col = 0;
							for (; col + 16 < N; col += 16)
							{
								adddot1x16(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}

							for (; col + 8 < N; col += 8)
							{
								adddot1x8(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}

							if (col < N - 1)
							{
								adddot1x4(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}
						}
					}
				}
				else if (!flag_M && !flag_N)
				{
					num_M2 = remain_M1 / 2;
					remain_M2 = remain_M1 % 2;
					num_N2 = remain_N1 / 2;
					remain_N2 = remain_N1 % 2;
					flag_M = flag_N = true;
					if (remain_M2 != 0) { flag_M = false; }
					if (remain_N2 != 0) { flag_N = false; }

					for(int n_row=0;n_row< num_M1;n_row++)
					{
						for(int n_col=0;n_col< num_N1;n_col++)
						{
							adddot4x4(padK, alpha, packedA + n_row * 4 * padK, packedB + n_col * 4 * padK, beta, C + n_row * 4 * ldc + n_col * 4, ldc);
						}
					}
					//deal with remain_M1
					{
						int col = 0;
						for (; col + 8 < N - remain_N1 - 1; col += 8)
						{
							if (num_M2)
							{
								adddot2x8(padK, alpha, packedA + num_M1 * 4 * padK, packedB + col * padK, beta, C + num_M1 * 4 * ldc + col, ldc);
							}
						}

						if(col < N - remain_N1 - 1 && num_M2)
						{
							adddot2x4(padK, alpha, packedA + num_M1 * 4 * padK, packedB + col * padK, beta, C + num_M1 * 4 * ldc + col, ldc);
						}

						if (!flag_M)
						{
							col = 0;
							for (; col + 16 < N - remain_N1 - 1; col += 16)
							{
								adddot1x16(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}

							for (; col + 8 < N - remain_N1 - 1; col += 8)
							{
								adddot1x8(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}

							if(col < N - remain_N1 - 1)
							{
								adddot1x4(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + col * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + col, ldc);
							}
						}
					}

					//deal with remain_N1
					{
						int row = 0;
						for (; row + 8 < M - remain_M1; row += 8)
						{
							for (int n = 0; n < num_N2; n++)
							{
								adddot8x2(padK, alpha, packedA + row * padK, packedB + num_N1 * 4 * padK, beta, C + row * ldc + num_N1 * 4, ldc);
							}
						}
						if (row < M - remain_M1 - 1 && num_N2)
						{
							adddot4x2(padK, alpha, packedA + row * padK, packedB + num_N1 * 4 * padK, beta, C + row * ldc + num_N1 * 4, ldc);
						}
						if (!flag_N)
						{
							row = 0;
							for (; row + 16 < M - remain_M1; row += 16)
							{
								adddot16x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}
							for (; row + 8 < M - remain_M1; row += 8)
							{
								adddot8x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}
							if (row < M - remain_M1 - 1)
							{
								adddot4x1(padK, alpha, packedA + row * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + row * ldc + num_N1 * 4 + num_N2 * 2, ldc);
							}
						}
					}

					//deal with remain_M2 and remain_N2 
					{
						if (!flag_M && !flag_N)
						{
							adddot2x2(padK, alpha, packedA + num_M1 * 4 * padK, packedB + num_N1 * 4 * padK, beta, C + num_M1 * 4 * ldc + num_N1 * 4, ldc);
							adddot1x2(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + num_N1 * 4 * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + num_N1 * 4, ldc);
							adddot2x1(padK, alpha, packedA + num_M1 * 4 * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + num_M1 * 4 * ldc + (num_N1 * 4 + num_N2 * 2), ldc);
							adddot1x1(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + (num_N1 * 4 + num_N2 * 2), ldc);
						}
						else if (flag_M && !flag_N)
						{
							adddot2x2(padK, alpha, packedA + num_M1 * 4 * padK, packedB + num_N1 * 4 * padK, beta, C + num_M1 * 4 * ldc + num_N1 * 4, ldc);
							adddot2x1(padK, alpha, packedA + num_M1 * 4 * padK, packedB + (num_N1 * 4 + num_N2 * 2) * padK, beta, C + num_M1 * 4 * ldc + (num_N1 * 4 + num_N2 * 2), ldc);
						}
						else if (!flag_M && flag_N)
						{
							adddot2x2(padK, alpha, packedA + num_M1 * 4 * padK, packedB + num_N1 * 4 * padK, beta, C + num_M1 * 4 * ldc + num_N1 * 4, ldc);
							adddot1x2(padK, alpha, packedA + (num_M1 * 4 + num_M2 * 2) * padK, packedB + num_N1 * 4 * padK, beta, C + (num_M1 * 4 + num_M2 * 2) * ldc + num_N1 * 4, ldc);
						}
					}
				}
				else
				{
					LOG(FATAL) << "Impossible code!";
				}
				delete[] packedA;
				delete[] packedB;
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