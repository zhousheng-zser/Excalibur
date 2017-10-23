#include "math_functions.hpp"

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
		cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			ldb, beta, C, N);
	}

}


