#include "julius_gemm.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void packnoTransedA(const int M, const int K, const int padK, const float* A, const int lda, float* packedA)
			{
				memset(packedA, 0, M * padK * sizeof(float));
				for (int i = 0; i < M; i++)
				{
					memcpy(packedA + i * padK, A + i * lda, K * sizeof(float));
				}
			}

			void packTransedA(const int M, const int K, const int padK, const float* A, const int lda, float* packedA)
			{
				memset(packedA, 0, M * padK * sizeof(float));
				for (int j = 0; j < K; j++)
				{
					const int offsetA = j * lda;
					for (int i = 0; i < M; i++)
					{
						packedA[i * padK + j] = A[offsetA + i];
					}
				}
			}

			void packnoTransedB(const int N, const int K, const int padK, const float* B, const int ldb, float* packedB)
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

			void packTransedB(const int N, const int K, const int padK, const float* B, const int ldb, float* packedB)
			{
				memset(packedB, 0, N * padK * sizeof(float));
				for (int j = 0; j < N; j++)
				{
					memcpy(packedB + j * padK, B + j * ldb, K * sizeof(float));
				}
			}

#if SIMD_TYPE > SIMDTYPE_NONE
			inline void adddot1x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re = mm_setzero_ps();
				for (int j = 0; j < padK; j += mm_align_size)
				{
					const mm_type a = mm_load_ps(packedA_ptr + j);
					const mm_type b = mm_load_ps(packedB_ptr + j);
					re = mm_fmadd_ps(a, b, re);
				}
				*C_ptr = alpha * mm_sumall_ps(re) + beta * *C_ptr;
			}

			//adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr + padK, beta, C_ptr + 1, ldc);
			inline void adddot1x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re1 = mm_setzero_ps();
				mm_type re2 = mm_setzero_ps();
				for (int j = 0; j < padK; j += mm_align_size)
				{
					const mm_type a = mm_load_ps(packedA_ptr + j);
					const mm_type b1 = mm_load_ps(packedB_ptr + j);
					const mm_type b2 = mm_load_ps(packedB_ptr + padK + j);
					re1 = mm_fmadd_ps(a, b1, re1);
					re2 = mm_fmadd_ps(a, b2, re2);
				}
				*C_ptr = alpha * mm_sumall_ps(re1) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re2) + beta * *(C_ptr + 1);
			}

			//adddot1x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x1(padK, alpha,  packedA_ptr + padK, packedB_ptr, beta, C_ptr + ldc, ldc);
			inline void adddot2x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re1 = mm_setzero_ps();
				mm_type re2 = mm_setzero_ps();
				for (int j = 0; j < padK; j += mm_align_size)
				{
					const mm_type a1 = mm_load_ps(packedA_ptr + j);
					const mm_type a2 = mm_load_ps(packedA_ptr + padK + j);
					const mm_type b = mm_load_ps(packedB_ptr + j);
					re1 = mm_fmadd_ps(a1, b, re1);
					re2 = mm_fmadd_ps(a2, b, re2);
				}
				*C_ptr = alpha * mm_sumall_ps(re1) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re2) + beta * *(C_ptr + ldc);
			}

			//adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr + padK * 2, beta, C_ptr + 2, ldc);
			inline void adddot1x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[4] = { mm_setzero_ps() };
				mm_type a;
				mm_type b[4];
				const int padK_offset[4] = { padK * 0, padK * 1, padK * 2, padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a = mm_load_ps(packedA_ptr + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					re[0] = mm_fmadd_ps(a, b[0], re[0]);
					re[1] = mm_fmadd_ps(a, b[1], re[1]);
					re[2] = mm_fmadd_ps(a, b[2], re[2]);
					re[3] = mm_fmadd_ps(a, b[3], re[3]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + 2) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + 2);
				*(C_ptr + 3) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + 3);
			}

			//adddot1x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x2(padK, alpha,  packedA_ptr + padK, packedB_ptr, beta, C_ptr + ldc, ldc);
			inline void adddot2x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[4] = { mm_setzero_ps() };
				mm_type a[2];
				mm_type b[2];
				const int padK_offset[4] = { padK * 0, padK * 1, padK * 2, padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[0], b[1], re[1]);
					re[2] = mm_fmadd_ps(a[1], b[0], re[2]);
					re[3] = mm_fmadd_ps(a[1], b[1], re[3]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + ldc) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + ldc);
				*(C_ptr + ldc + 1) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + ldc + 1);
			}

			//adddot2x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot2x1(padK, alpha,  packedA_ptr + padK * 2, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			inline void adddot4x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[4] = { mm_setzero_ps() };
				mm_type a[4];
				mm_type b;
				const int padK_offset[4] = { padK * 0, padK * 1, padK * 2, padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					b = mm_load_ps(packedB_ptr + j);
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					re[0] = mm_fmadd_ps(a[0], b, re[0]);
					re[1] = mm_fmadd_ps(a[1], b, re[1]);
					re[2] = mm_fmadd_ps(a[2], b, re[2]);
					re[3] = mm_fmadd_ps(a[3], b, re[3]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1 * ldc) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1 * ldc);
				*(C_ptr + 2 * ldc) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + 2 * ldc);
				*(C_ptr + 3 * ldc) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + 3 * ldc);
			}

			//adddot1x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x4(padK, alpha,  packedA_ptr, packedB_ptr + padK * 4, beta, C_ptr + 4, ldc);*/
			inline void adddot1x8(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[8] = { mm_setzero_ps() };
				mm_type a;
				mm_type b[8];
				const int padK_offset[8] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a = mm_load_ps(packedA_ptr + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					b[4] = mm_load_ps(packedB_ptr + padK_offset[4] + j);
					b[5] = mm_load_ps(packedB_ptr + padK_offset[5] + j);
					b[6] = mm_load_ps(packedB_ptr + padK_offset[6] + j);
					b[7] = mm_load_ps(packedB_ptr + padK_offset[7] + j);
					re[0] = mm_fmadd_ps(a, b[0], re[0]);
					re[1] = mm_fmadd_ps(a, b[1], re[1]);
					re[2] = mm_fmadd_ps(a, b[2], re[2]);
					re[3] = mm_fmadd_ps(a, b[3], re[3]);
					re[4] = mm_fmadd_ps(a, b[4], re[4]);
					re[5] = mm_fmadd_ps(a, b[5], re[5]);
					re[6] = mm_fmadd_ps(a, b[6], re[6]);
					re[7] = mm_fmadd_ps(a, b[7], re[7]);
				}
				for (int i = 0; i < 8; i++)
				{
					*(C_ptr + i) = alpha * mm_sumall_ps(re[i]) + beta * *(C_ptr + i);
				}
			}

			//adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr + padK * 2, beta, C_ptr + 2, ldc);
			inline void adddot2x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[8] = { mm_setzero_ps() };
				mm_type a[2];
				mm_type b[4];
				const int padK_offset[4] = { padK * 0, padK * 1, padK * 2, padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[0], b[1], re[1]);
					re[2] = mm_fmadd_ps(a[0], b[2], re[2]);
					re[3] = mm_fmadd_ps(a[0], b[3], re[3]);
					re[4] = mm_fmadd_ps(a[1], b[0], re[4]);
					re[5] = mm_fmadd_ps(a[1], b[1], re[5]);
					re[6] = mm_fmadd_ps(a[1], b[2], re[6]);
					re[7] = mm_fmadd_ps(a[1], b[3], re[7]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + 2) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + 2);
				*(C_ptr + 3) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + 3);
				*(C_ptr + ldc) = alpha * mm_sumall_ps(re[4]) + beta * *(C_ptr + ldc);
				*(C_ptr + ldc + 1) = alpha * mm_sumall_ps(re[5]) + beta * *(C_ptr + ldc + 1);
				*(C_ptr + ldc + 2) = alpha * mm_sumall_ps(re[6]) + beta * *(C_ptr + ldc + 2);
				*(C_ptr + ldc + 3) = alpha * mm_sumall_ps(re[7]) + beta * *(C_ptr + ldc + 3);
			}

			//adddot2x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot2x2(padK, alpha,  packedA_ptr + padK * 2, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			inline void adddot4x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[8] = { mm_setzero_ps() };
				mm_type a[4];
				mm_type b[2];
				const int padK_offset[4] = { padK * 0, padK * 1, padK * 2, padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[1], b[0], re[1]);
					re[2] = mm_fmadd_ps(a[2], b[0], re[2]);
					re[3] = mm_fmadd_ps(a[3], b[0], re[3]);
					re[4] = mm_fmadd_ps(a[0], b[1], re[4]);
					re[5] = mm_fmadd_ps(a[1], b[1], re[5]);
					re[6] = mm_fmadd_ps(a[2], b[1], re[6]);
					re[7] = mm_fmadd_ps(a[3], b[1],  re[7]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + ldc * 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + ldc * 1);
				*(C_ptr + ldc * 2) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + ldc * 2);
				*(C_ptr + ldc * 3) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + ldc * 3);
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[4]) + beta * *(C_ptr + 1);
				*(C_ptr + ldc * 1 + 1) = alpha * mm_sumall_ps(re[5]) + beta * *(C_ptr + ldc * 1 + 1);
				*(C_ptr + ldc * 2 + 1) = alpha * mm_sumall_ps(re[6]) + beta * *(C_ptr + ldc * 2 + 1);
				*(C_ptr + ldc * 3 + 1) = alpha * mm_sumall_ps(re[7]) + beta * *(C_ptr + ldc * 3 + 1);
			}

			//adddot4x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot4x1(padK, alpha,  packedA_ptr + padK * 4, packedB_ptr, beta, C_ptr + 4 * ldc, ldc);
			inline void adddot8x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[8] = { mm_setzero_ps() };
				mm_type a[8];
				mm_type b;
				const int padK_offset[8] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					a[4] = mm_load_ps(packedA_ptr + padK_offset[4] + j);
					a[5] = mm_load_ps(packedA_ptr + padK_offset[5] + j);
					a[6] = mm_load_ps(packedA_ptr + padK_offset[6] + j);
					a[7] = mm_load_ps(packedA_ptr + padK_offset[7] + j);
					b = mm_load_ps(packedB_ptr + j);
					re[0] = mm_fmadd_ps(a[0], b, re[0]);
					re[1] = mm_fmadd_ps(a[1], b, re[1]);
					re[2] = mm_fmadd_ps(a[2], b, re[2]);
					re[3] = mm_fmadd_ps(a[3], b, re[3]);
					re[4] = mm_fmadd_ps(a[4], b, re[4]);
					re[5] = mm_fmadd_ps(a[5], b, re[5]);
					re[6] = mm_fmadd_ps(a[6], b, re[6]);
					re[7] = mm_fmadd_ps(a[7], b, re[7]);
				}
				for (int i = 0; i < 8; i++)
				{
					*(C_ptr + ldc * i) = alpha * mm_sumall_ps(re[i]) + beta * *(C_ptr + ldc * i);
				}
			}

			//adddot1x8(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot1x8(padK, alpha,  packedA_ptr, packedB_ptr + padK * 8, beta, C_ptr + 8, ldc);
			inline void adddot1x16(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[16] = { mm_setzero_ps() };
				mm_type a;
				mm_type b[16];
				const int padK_offset[16] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7,
				padK * 8, padK * 9, padK * 10, padK * 11, padK * 12, padK * 13, padK * 14, padK * 15 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a = mm_load_ps(packedA_ptr + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					b[4] = mm_load_ps(packedB_ptr + padK_offset[4] + j);
					b[5] = mm_load_ps(packedB_ptr + padK_offset[5] + j);
					b[6] = mm_load_ps(packedB_ptr + padK_offset[6] + j);
					b[7] = mm_load_ps(packedB_ptr + padK_offset[7] + j);
					b[8] = mm_load_ps(packedB_ptr + padK_offset[8] + j);
					b[9] = mm_load_ps(packedB_ptr + padK_offset[9] + j);
					b[10] = mm_load_ps(packedB_ptr + padK_offset[10] + j);
					b[11] = mm_load_ps(packedB_ptr + padK_offset[11] + j);
					b[12] = mm_load_ps(packedB_ptr + padK_offset[12] + j);
					b[13] = mm_load_ps(packedB_ptr + padK_offset[13] + j);
					b[14] = mm_load_ps(packedB_ptr + padK_offset[14] + j);
					b[15] = mm_load_ps(packedB_ptr + padK_offset[15] + j);
					re[0] = mm_fmadd_ps(a, b[0], re[0]);
					re[1] = mm_fmadd_ps(a, b[1], re[1]);
					re[2] = mm_fmadd_ps(a, b[2], re[2]);
					re[3] = mm_fmadd_ps(a, b[3], re[3]);
					re[4] = mm_fmadd_ps(a, b[4], re[4]);
					re[5] = mm_fmadd_ps(a, b[5], re[5]);
					re[6] = mm_fmadd_ps(a, b[6], re[6]);
					re[7] = mm_fmadd_ps(a, b[7], re[7]);
					re[8] = mm_fmadd_ps(a, b[8], re[8]);
					re[9] = mm_fmadd_ps(a, b[9], re[9]);
					re[10] = mm_fmadd_ps(a, b[10], re[10]);
					re[11] = mm_fmadd_ps(a, b[11], re[11]);
					re[12] = mm_fmadd_ps(a, b[12], re[12]);
					re[13] = mm_fmadd_ps(a, b[13], re[13]);
					re[14] = mm_fmadd_ps(a, b[14], re[14]);
					re[15] = mm_fmadd_ps(a, b[15], re[15]);
				}
				for (int i = 0; i < 16; i++)
				{
					*(C_ptr + i) = alpha * mm_sumall_ps(re[i]) + beta * *(C_ptr + i);
				}
			}

			//adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr + padK * 4, beta, C_ptr + 4, ldc);
			inline void adddot2x8(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[16] = { mm_setzero_ps() };
				mm_type a[2];
				mm_type b[8];
				const int padK_offset[8] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7};
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					b[4] = mm_load_ps(packedB_ptr + padK_offset[4] + j);
					b[5] = mm_load_ps(packedB_ptr + padK_offset[5] + j);
					b[6] = mm_load_ps(packedB_ptr + padK_offset[6] + j);
					b[7] = mm_load_ps(packedB_ptr + padK_offset[7] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[0], b[1], re[1]);
					re[2] = mm_fmadd_ps(a[0], b[2], re[2]);
					re[3] = mm_fmadd_ps(a[0], b[3], re[3]);
					re[4] = mm_fmadd_ps(a[0], b[4], re[4]);
					re[5] = mm_fmadd_ps(a[0], b[5], re[5]);
					re[6] = mm_fmadd_ps(a[0], b[6], re[6]);
					re[7] = mm_fmadd_ps(a[0], b[7], re[7]);
					re[8] = mm_fmadd_ps(a[1], b[0], re[8]);
					re[9] = mm_fmadd_ps(a[1], b[1], re[9]);
					re[10] = mm_fmadd_ps(a[1], b[2], re[10]);
					re[11] = mm_fmadd_ps(a[1], b[3], re[11]);
					re[12] = mm_fmadd_ps(a[1], b[4], re[12]);
					re[13] = mm_fmadd_ps(a[1], b[5], re[13]);
					re[14] = mm_fmadd_ps(a[1], b[6], re[14]);
					re[15] = mm_fmadd_ps(a[1], b[7], re[15]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + 2) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + 2);
				*(C_ptr + 3) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + 3);
				*(C_ptr + 4) = alpha * mm_sumall_ps(re[4]) + beta * *(C_ptr + 4);
				*(C_ptr + 5) = alpha * mm_sumall_ps(re[5]) + beta * *(C_ptr + 5);
				*(C_ptr + 6) = alpha * mm_sumall_ps(re[6]) + beta * *(C_ptr + 6);
				*(C_ptr + 7) = alpha * mm_sumall_ps(re[7]) + beta * *(C_ptr + 7);
				*(C_ptr + ldc) = alpha * mm_sumall_ps(re[8]) + beta * *(C_ptr + ldc);
				*(C_ptr + ldc + 1) = alpha * mm_sumall_ps(re[9]) + beta * *(C_ptr + ldc + 1);
				*(C_ptr + ldc + 2) = alpha * mm_sumall_ps(re[10]) + beta * *(C_ptr + ldc + 2);
				*(C_ptr + ldc + 3) = alpha * mm_sumall_ps(re[11]) + beta * *(C_ptr + ldc + 3);
				*(C_ptr + ldc + 4) = alpha * mm_sumall_ps(re[12]) + beta * *(C_ptr + ldc + 4);
				*(C_ptr + ldc + 5) = alpha * mm_sumall_ps(re[13]) + beta * *(C_ptr + ldc + 5);
				*(C_ptr + ldc + 6) = alpha * mm_sumall_ps(re[14]) + beta * *(C_ptr + ldc + 6);
				*(C_ptr + ldc + 7) = alpha * mm_sumall_ps(re[15]) + beta * *(C_ptr + ldc + 7);
			}

			//adddot2x4(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot2x4(padK, alpha,  packedA_ptr + 2 * padK, packedB_ptr, beta, C_ptr + 2 * ldc, ldc);
			inline void adddot4x4(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[16] = { mm_setzero_ps() };
				mm_type a[4];
				mm_type b[4];
				const int padK_offset[4] = { padK * 0, padK * 1 ,padK * 2,padK * 3 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					b[2] = mm_load_ps(packedB_ptr + padK_offset[2] + j);
					b[3] = mm_load_ps(packedB_ptr + padK_offset[3] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[0], b[1], re[1]);
					re[2] = mm_fmadd_ps(a[0], b[2], re[2]);
					re[3] = mm_fmadd_ps(a[0], b[3], re[3]);
					re[4] = mm_fmadd_ps(a[1], b[0], re[4]);
					re[5] = mm_fmadd_ps(a[1], b[1], re[5]);
					re[6] = mm_fmadd_ps(a[1], b[2], re[6]);
					re[7] = mm_fmadd_ps(a[1], b[3], re[7]);
					re[8] = mm_fmadd_ps(a[2], b[0], re[8]);
					re[9] = mm_fmadd_ps(a[2], b[1], re[9]);
					re[10] = mm_fmadd_ps(a[2], b[2], re[10]);
					re[11] = mm_fmadd_ps(a[2], b[3], re[11]);
					re[12] = mm_fmadd_ps(a[3], b[0], re[12]);
					re[13] = mm_fmadd_ps(a[3], b[1], re[13]);
					re[14] = mm_fmadd_ps(a[3], b[2], re[14]);
					re[15] = mm_fmadd_ps(a[3], b[3], re[15]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + 2) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + 2);
				*(C_ptr + 3) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + 3);
				*(C_ptr + ldc * 1) = alpha * mm_sumall_ps(re[4]) + beta * *(C_ptr + ldc * 1);
				*(C_ptr + ldc * 1 + 1) = alpha * mm_sumall_ps(re[5]) + beta * *(C_ptr + ldc * 1 + 1);
				*(C_ptr + ldc * 1 + 2) = alpha * mm_sumall_ps(re[6]) + beta * *(C_ptr + ldc * 1 + 2);
				*(C_ptr + ldc * 1 + 3) = alpha * mm_sumall_ps(re[7]) + beta * *(C_ptr + ldc * 1 + 3);
				*(C_ptr + ldc * 2) = alpha * mm_sumall_ps(re[8]) + beta * *(C_ptr + ldc * 2);
				*(C_ptr + ldc * 2 + 1) = alpha * mm_sumall_ps(re[9]) + beta * *(C_ptr + ldc * 2 + 1);
				*(C_ptr + ldc * 2 + 2) = alpha * mm_sumall_ps(re[10]) + beta * *(C_ptr + ldc * 2 + 2);
				*(C_ptr + ldc * 2 + 3) = alpha * mm_sumall_ps(re[11]) + beta * *(C_ptr + ldc * 2 + 3);
				*(C_ptr + ldc * 3) = alpha * mm_sumall_ps(re[12]) + beta * *(C_ptr + ldc * 3);
				*(C_ptr + ldc * 3 + 1) = alpha * mm_sumall_ps(re[13]) + beta * *(C_ptr + ldc * 3 + 1);
				*(C_ptr + ldc * 3 + 2) = alpha * mm_sumall_ps(re[14]) + beta * *(C_ptr + ldc * 3 + 2);
				*(C_ptr + ldc * 3 + 3) = alpha * mm_sumall_ps(re[15]) + beta * *(C_ptr + ldc * 3 + 3);
			}

			//adddot4x2(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot4x2(padK, alpha,  packedA_ptr + padK * 4, packedB_ptr, beta, C_ptr + 4 * ldc, ldc);
			inline void adddot8x2(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[16] = { mm_setzero_ps() };
				mm_type a[8];
				mm_type b[2];
				const int padK_offset[8] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					a[4] = mm_load_ps(packedA_ptr + padK_offset[4] + j);
					a[5] = mm_load_ps(packedA_ptr + padK_offset[5] + j);
					a[6] = mm_load_ps(packedA_ptr + padK_offset[6] + j);
					a[7] = mm_load_ps(packedA_ptr + padK_offset[7] + j);
					b[0] = mm_load_ps(packedB_ptr + padK_offset[0] + j);
					b[1] = mm_load_ps(packedB_ptr + padK_offset[1] + j);
					re[0] = mm_fmadd_ps(a[0], b[0], re[0]);
					re[1] = mm_fmadd_ps(a[0], b[1], re[1]);
					re[2] = mm_fmadd_ps(a[1], b[0], re[2]);
					re[3] = mm_fmadd_ps(a[1], b[1], re[3]);
					re[4] = mm_fmadd_ps(a[2], b[0], re[4]);
					re[5] = mm_fmadd_ps(a[2], b[1], re[5]);
					re[6] = mm_fmadd_ps(a[3], b[0], re[6]);
					re[7] = mm_fmadd_ps(a[3], b[1], re[7]);
					re[8] = mm_fmadd_ps(a[4], b[0], re[8]);
					re[9] = mm_fmadd_ps(a[4], b[1], re[9]);
					re[10] = mm_fmadd_ps(a[5], b[0], re[10]);
					re[11] = mm_fmadd_ps(a[5], b[1], re[11]);
					re[12] = mm_fmadd_ps(a[6], b[0], re[12]);
					re[13] = mm_fmadd_ps(a[6], b[1], re[13]);
					re[14] = mm_fmadd_ps(a[7], b[0], re[14]);
					re[15] = mm_fmadd_ps(a[7], b[1], re[15]);
				}
				*C_ptr = alpha * mm_sumall_ps(re[0]) + beta * *C_ptr;
				*(C_ptr + 1) = alpha * mm_sumall_ps(re[1]) + beta * *(C_ptr + 1);
				*(C_ptr + ldc) = alpha * mm_sumall_ps(re[2]) + beta * *(C_ptr + ldc);
				*(C_ptr + ldc + 1) = alpha * mm_sumall_ps(re[3]) + beta * *(C_ptr + ldc + 1);
				*(C_ptr + ldc * 2) = alpha * mm_sumall_ps(re[4]) + beta * *(C_ptr + ldc * 2);
				*(C_ptr + ldc * 2 + 1) = alpha * mm_sumall_ps(re[5]) + beta * *(C_ptr + ldc * 2 + 1);
				*(C_ptr + ldc * 3) = alpha * mm_sumall_ps(re[6]) + beta * *(C_ptr + ldc * 3);
				*(C_ptr + ldc * 3 + 1) = alpha * mm_sumall_ps(re[7]) + beta * *(C_ptr + ldc * 3 + 1);
				*(C_ptr + ldc * 4) = alpha * mm_sumall_ps(re[8]) + beta * *(C_ptr + ldc * 4);
				*(C_ptr + ldc * 4 + 1) = alpha * mm_sumall_ps(re[9]) + beta * *(C_ptr + ldc * 4 + 1);
				*(C_ptr + ldc * 5) = alpha * mm_sumall_ps(re[10]) + beta * *(C_ptr + ldc * 5);
				*(C_ptr + ldc * 5 + 1) = alpha * mm_sumall_ps(re[11]) + beta * *(C_ptr + ldc * 5 + 1);
				*(C_ptr + ldc * 6) = alpha * mm_sumall_ps(re[12]) + beta * *(C_ptr + ldc * 6);
				*(C_ptr + ldc * 6 + 1) = alpha * mm_sumall_ps(re[13]) + beta * *(C_ptr + ldc * 6 + 1);
				*(C_ptr + ldc * 7) = alpha * mm_sumall_ps(re[14]) + beta * *(C_ptr + ldc * 7);
				*(C_ptr + ldc * 7 + 1) = alpha * mm_sumall_ps(re[15]) + beta * *(C_ptr + ldc * 7 + 1);
			}

			//adddot8x1(padK, alpha,  packedA_ptr, packedB_ptr, beta, C_ptr, ldc);
			//adddot8x1(padK, alpha,  packedA_ptr + padK * 8, packedB_ptr, beta, C_ptr + 8 * ldc, ldc);
			inline void adddot16x1(const int padK, const float alpha, const float* packedA_ptr, const float* packedB_ptr, 
				const float beta, float* C_ptr, const int ldc)
			{
				mm_type re[16] = { mm_setzero_ps() };
				mm_type a[16];
				mm_type b;
				const int padK_offset[16] = { padK * 0, padK * 1, padK * 2, padK * 3, padK * 4, padK * 5, padK * 6, padK * 7,
				padK * 8, padK * 9, padK * 10, padK * 11, padK * 12, padK * 13, padK * 14, padK * 15 };
				for (int j = 0; j < padK; j += mm_align_size)
				{
					a[0] = mm_load_ps(packedA_ptr + padK_offset[0] + j);
					a[1] = mm_load_ps(packedA_ptr + padK_offset[1] + j);
					a[2] = mm_load_ps(packedA_ptr + padK_offset[2] + j);
					a[3] = mm_load_ps(packedA_ptr + padK_offset[3] + j);
					a[4] = mm_load_ps(packedA_ptr + padK_offset[4] + j);
					a[5] = mm_load_ps(packedA_ptr + padK_offset[5] + j);
					a[6] = mm_load_ps(packedA_ptr + padK_offset[6] + j);
					a[7] = mm_load_ps(packedA_ptr + padK_offset[7] + j);
					a[8] = mm_load_ps(packedA_ptr + padK_offset[8] + j);
					a[9] = mm_load_ps(packedA_ptr + padK_offset[9] + j);
					a[10] = mm_load_ps(packedA_ptr + padK_offset[10] + j);
					a[11] = mm_load_ps(packedA_ptr + padK_offset[11] + j);
					a[12] = mm_load_ps(packedA_ptr + padK_offset[12] + j);
					a[13] = mm_load_ps(packedA_ptr + padK_offset[13] + j);
					a[14] = mm_load_ps(packedA_ptr + padK_offset[14] + j);
					a[15] = mm_load_ps(packedA_ptr + padK_offset[15] + j);
					b = mm_load_ps(packedB_ptr + j);
					re[0] = mm_fmadd_ps(a[0], b, re[0]);
					re[1] = mm_fmadd_ps(a[1], b, re[1]);
					re[2] = mm_fmadd_ps(a[2], b, re[2]);
					re[3] = mm_fmadd_ps(a[3], b, re[3]);
					re[4] = mm_fmadd_ps(a[4], b, re[4]);
					re[5] = mm_fmadd_ps(a[5], b, re[5]);
					re[6] = mm_fmadd_ps(a[6], b, re[6]);
					re[7] = mm_fmadd_ps(a[7], b, re[7]);
					re[8] = mm_fmadd_ps(a[8], b, re[8]);
					re[9] = mm_fmadd_ps(a[9], b, re[9]);
					re[10] = mm_fmadd_ps(a[10], b, re[10]);
					re[11] = mm_fmadd_ps(a[11], b, re[11]);
					re[12] = mm_fmadd_ps(a[12], b, re[12]);
					re[13] = mm_fmadd_ps(a[13], b, re[13]);
					re[14] = mm_fmadd_ps(a[14], b, re[14]);
					re[15] = mm_fmadd_ps(a[15], b, re[15]);
				}
				for (int i = 0; i < 16; i++)
				{
					*(C_ptr + ldc * i) = alpha * mm_sumall_ps(re[i]) + beta * *(C_ptr + ldc * i);
				}
			}

			void execution_kernel_16register(const int M, const int N, const int padK, const float alpha, float* packedA, 
				float* packedB, const float beta,  float* C, const int ldc)
			{
				const int quotient_M = M / 4;
				const int quotient_N = N / 4;
				const int remain_M = M % 4;
				const int remain_N = N % 4;
				const int part_M = M - remain_M;
				const int part_N = N - remain_N;
				// deal with the most(left top) part of C in 4x4 block
				if (part_N >= part_M)
				{
					for (int i = 0; i < part_M; i += 4)
					{
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
						for (int j = 0; j < part_N; j += 4)
						{
							adddot4x4(padK, alpha, packedA + i * padK, packedB + j * padK, beta, C + i * ldc + j, ldc);
						}
					}
				}
				else
				{
					for (int j = 0; j < part_N; j += 4)
					{
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
						for (int i = 0; i < part_M; i += 4)
						{
							adddot4x4(padK, alpha, packedA + i * padK, packedB + j * padK, beta, C + i * ldc + j, ldc);
						}
					}
				}
				
				// deal with left bottom part of C in 2x8, 1x16, 1x8 or 1x4 block
				if (remain_M / 2)
				{
					int j = 0;
					const int total_8times = part_N / 8;
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
					for (int i = 0; i < total_8times; i++)
					{
						adddot2x8(padK, alpha, packedA + part_M * padK, packedB + i * 8 * padK, beta, C + part_M * ldc + i * 8, ldc);
					}
					j = 8 * total_8times;
					if (j + 4 == part_N)
					{
						adddot2x4(padK, alpha, packedA + part_M * padK, packedB + j * padK, beta, C + part_M * ldc + j, ldc);
					}
				}
				// the last row in C
				if (remain_M % 2)
				{
					int j = 0;
					int offset_M = remain_M / 2 ? 2 + part_M : part_M;
					const int total_16times = part_N / 16;
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
					for (int i = 0; i < total_16times; i++)
					{
						adddot1x16(padK, alpha, packedA + offset_M * padK, packedB + i * 16 * padK, beta, C + offset_M * ldc + i * 16, ldc);
					}
					j = total_16times * 16;
					for (; j + 8 <= part_N; j += 8)
					{
						adddot1x8(padK, alpha, packedA + offset_M * padK, packedB + j * padK, beta, C + offset_M * ldc + j, ldc);
					}
					if (j + 4 == part_N)
					{
						adddot1x4(padK, alpha, packedA + offset_M * padK, packedB + j * padK, beta, C + offset_M * ldc + j, ldc);
					}
				}

				// deal with right top part of C in 8x2, 16x1, 8x1 or 4x1 block
				if (remain_N / 2)
				{
					int i = 0;
					const int total_8times = part_M / 8;
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
					for (int j = 0; j < total_8times; j++)
					{
						adddot8x2(padK, alpha, packedA + j * 8 * padK, packedB + part_N * padK, beta, C + j * 8 * ldc + part_N, ldc);
					}
					i = total_8times * 8;
					if (i + 4 == part_M)
					{
						adddot4x2(padK, alpha, packedA + i * padK, packedB + part_N * padK, beta, C + i * ldc + part_N, ldc);
					}
				}
				// the last colum in C
				if (remain_N % 2)
				{
					int i = 0;
					int offset_M = remain_N / 2 ? 2 + part_N : part_N;
					const int total_16times = part_M / 16;
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
					for (int j = 0; j < total_16times; j++)
					{
						adddot16x1(padK, alpha, packedA + j * 16 * padK, packedB + offset_M * padK, beta, C + j * 16 * ldc + offset_M, ldc);
					}
					i = total_16times * 16;
					for (; i + 8 <= part_M; i += 8)
					{
						adddot8x1(padK, alpha, packedA + i * padK, packedB + offset_M * padK, beta, C + i * ldc + offset_M, ldc);
					}
					if (i + 4 == part_M)
					{
						adddot4x1(padK, alpha, packedA + i * padK, packedB + offset_M * padK, beta, C + i * ldc + offset_M, ldc);
					}
				}

				// deal with the last(right bottom) part of C in 1x1 block
				if (remain_M * remain_N > 0)
				{
					for (int row = 0; row < M - part_M; row++)
					{
						for (int col = 0; col < N - part_N; col++)
						{
							adddot1x1(padK, alpha, packedA + (part_M + row) * padK, packedB + (part_N + col) * padK, beta, C + (part_M + row) * ldc + (part_N + col), ldc);
						}
					}
				}
			}
#endif //!SIMD_TYPE > SIMDTYPE_NONE

			void cblas_sgemm_AnoTrans_BnoTrans(const int M, const int N, const int K, const float alpha, const float* A, const int lda,
				const float* B, const int ldb, const float beta, float* C, const int ldc)
			{
#if SIDM_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
				NATIVE_CODE_WARNING;
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = K % mm_align_size ? K - K % mm_align_size + mm_align_size: K;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				packnoTransedA(M, K, padK, A, lda, packedA);
				packnoTransedB(N, K, padK, B, ldb, packedB);
				execution_kernel_16register(M, N, padK, alpha, packedA, packedB, beta, C, ldc);
				delete[] packedA;
				delete[] packedB;
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
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = K % mm_align_size ? K - K % mm_align_size + mm_align_size : K;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				packTransedA(M, K, padK, A, lda, packedA);
				packnoTransedB(N, K, padK, B, ldb, packedB);
				execution_kernel_16register(M, N, padK, alpha, packedA, packedB, beta, C, ldc);
				delete[] packedA;
				delete[] packedB;
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
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = K % mm_align_size ? K - K % mm_align_size + mm_align_size : K;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				packnoTransedA(M, K, padK, A, lda, packedA);
				packTransedB(N, K, padK, B, ldb, packedB);
				execution_kernel_16register(M, N, padK, alpha, packedA, packedB, beta, C, ldc);
				delete[] packedA;
				delete[] packedB;
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
// AVX and SSE code follow the same logic, we merge them together.
#elif SIMD_TYPE >= SIMDTYPE_SSE
				const int padK = K % mm_align_size ? K - K % mm_align_size + mm_align_size : K;
				float* packedA = new float[M * padK];
				float* packedB = new float[N * padK];
				packTransedA(M, K, padK, A, lda, packedA);
				packTransedB(N, K, padK, B, ldb, packedB);
				execution_kernel_16register(M, N, padK, alpha, packedA, packedB, beta, C, ldc);
				delete[] packedA;
				delete[] packedB;
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