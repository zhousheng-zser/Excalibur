#include "math_functions.hpp"
#include <filesystem>
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		math_functions::math_functions()
		{
		}


		math_functions::~math_functions()
		{
		}

		void math_functions::cpu_sgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C)
		{
			int lda = (TransA == CblasNoTrans) ? K : M;
			int ldb = (TransB == CblasNoTrans) ? N : K;
#if (USE_MKL || USE_OPENBLAS)
			cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
				ldb, beta, C, N);
#else
			julius::cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
				ldb, beta, C, N);
#endif
		}

		void math_functions::cpu_sgemv(const CBLAS_TRANSPOSE TransA, const int M,
			const int N, const float alpha, const float* A, const float* x,
			const float beta, float* y) 
		{
#if (USE_MKL || USE_OPENBLAS)
			cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
#else
			// ??! lda problem!
			//int lda = (TransA == CblasNoTrans) ? N : M;
			julius::cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
#endif
		}

#ifdef USE_MKL
		void math_functions::cpu_batch_sgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const float alpha, const float* A, const int A_offset, const float* B, const int B_offset, const float beta, float* C, const int C_offset, int num)
		{
			int lda = (TransA == CblasNoTrans) ? K : M;
			int ldb = (TransB == CblasNoTrans) ? N : K;
			//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
			std::vector<CBLAS_TRANSPOSE> transa_array(num,TransA);
			std::vector<CBLAS_TRANSPOSE> transb_array(num, TransB);
			std::vector<MKL_INT> m_array(num, M);
			std::vector<MKL_INT> n_array(num, N);
			std::vector<MKL_INT> k_array(num, K);
			std::vector<MKL_INT> lda_array(num, lda);
			std::vector<MKL_INT> ldb_array(num, ldb);
			std::vector<MKL_INT> ldc_array(num, N);
			std::vector<MKL_INT> group_size(num, 1);
			std::vector<float> alpha_array(num, alpha);
			std::vector<float> beta_array(num, beta);
			std::vector<const float*> a_array, b_array;
			std::vector<float*> c_array;

			for (size_t i = 0; i < num; i++)
			{
				a_array.push_back(A + i*A_offset);
				b_array.push_back(B + i*B_offset);
				c_array.push_back(C + i*C_offset);
			}

			cblas_sgemm_batch(CblasRowMajor, transa_array.data(), transb_array.data(), m_array.data(), n_array.data(), k_array.data(), 
				alpha_array.data(), a_array.data(), lda_array.data(), b_array.data(), ldb_array.data(), beta_array.data(), c_array.data(), ldc_array.data(), MKL_INT(num), group_size.data());
		}
#endif
		
		void math_functions::cpu_set(const int N, const float alpha, float* Y)
		{
			if (alpha == 0) {
				memset(Y, 0, sizeof(float) * N);
				return;
			}
			for (int i = 0; i < N; ++i) {
				Y[i] = alpha;
			}
		}

		void math_functions::cpu_sqr(const int N, const float* a, float* y)
		{
			vsSqr(N, a, y);
		}

		void math_functions::cpu_abs(const int N, const float* a, float* y)
		{
			vsAbs(N, a, y);
		}

	}
}



