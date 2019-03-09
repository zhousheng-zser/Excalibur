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
					re[1] = mm_fmadd_ps(a[2], b, re[1]);
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
#endif //!SIMD_TYPE > SIMDTYPE_NONE

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
				packnoTransedA(M, K, padK, A, lda, packedA);
				packnoTransedB(N, K, padK, B, ldb, packedB);

				bool flag_M = true, flag_N = true;
				const int num_M1 = M / 4, num_N1 = N / 4, remain_M1 = M % 4, remain_N1 = N % 4;
				int num_M2, num_N2, remain_M2, remain_N2;
				if (remain_M1 != 0) { flag_M = false; }
				if (remain_N1 != 0) { flag_N = false; }
				if(flag_M && flag_N) 
				{
					for (int n_row = 0; n_row < num_M1; n_row ++)
					{
#ifdef _OPENMP
#pragma omp parallel for 
#endif // _OPENMP
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