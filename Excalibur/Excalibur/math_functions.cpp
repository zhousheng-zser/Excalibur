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
			//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
			cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
				ldb, beta, C, N);
			/*std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			std::cout << "total blas_gemm time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;*/
		}

		/*template <typename Dtype>
		void math_functions::excalibur_copy(const int N, const Dtype* X, Dtype* Y, int device)*/
		

		/*template void math_functions::excalibur_copy<float>(const int N, const float* X, float* Y, int device);
		template void math_functions::excalibur_copy<unsigned char>(const int N, const unsigned char* X, unsigned char* Y, int device);
		template void math_functions::excalibur_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y, int device);
		template void math_functions::excalibur_copy<int>(const int N, const int* X, int* Y, int device);
		template void math_functions::excalibur_copy<char>(const int N, const char* X, char* Y, int device);
		template void math_functions::excalibur_copy<double>(const int N, const double* X, double* Y, int device);
		template void math_functions::excalibur_copy<int const *>(const int N, int const* const *X, int const**Y, int device);*/

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



