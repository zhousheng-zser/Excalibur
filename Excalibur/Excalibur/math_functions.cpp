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

	void math_functions::excalibur_copy(const int N, const float* X, float* Y, int device)
	{
		if (X!=Y)
		{
			if (device>=0)
			{
#ifdef USE_CUDA
				cudaSetDevice(device);
				CUDA_CHECK(cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault));
#else
				NO_GPU
#endif
			}
			else
			{
				memcpy(Y, X, sizeof(float) * N);
			}
		}
	}

}


