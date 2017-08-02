#include "Accelerator.hpp"
#include "Excalibur_MathFunctions.hpp"

namespace Excalibur
{
	template<>
	void Excalibur_MathFunctions::excalibur_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C)
	{
		// Note that cublas follows fortran order.
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB =
			(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasSgemm(cublas_handle_, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
	}
}