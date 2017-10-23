#include "math_functions.hpp"
#ifdef USE_CUDA
namespace excalibur
{
	__global__ void set_kernel(const int n, const float alpha, float* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = alpha;
		}
	}

	void math_functions::gpu_set(const int N, const float alpha, float* Y)
	{
		if (alpha == 0) {
			CUDA_CHECK(cudaMemset(Y, 0, sizeof(float) * N));  // NOLINT(caffe/alt_fn)
			return;
		}
		// NOLINT_NEXT_LINE(whitespace/operators)
		set_kernel << <CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS >> >(
			N, alpha, Y);
	}

	void math_functions::gpu_sgemm(cublasHandle_t cublas_handle_, const CBLAS_TRANSPOSE TransA,
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
#endif