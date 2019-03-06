#ifdef USE_CUDA
#include "math_functions.hpp"

namespace glasssix
{
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
				CUDA_CHECK(cudaMemset(Y, 0, sizeof(float) * N));
				return;
			}
			set_kernel << <CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS >> >(
				N, alpha, Y);
		}

		__global__ void powx_kernel(const int n, const float* a,
			const float alpha, float* y) {
			CUDA_KERNEL_LOOP(index, n) {
				y[index] = pow(a[index], alpha);
			}
		}

		void math_functions::gpu_powx(const int N, const float* a, const float alpha, float* y)
		{
			powx_kernel << <CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS >> >(
				N, a, alpha, y);
		}

		__global__ void abs_kernel(const int n, const float* a, float* y) {
			CUDA_KERNEL_LOOP(index, n) {
				y[index] = abs(a[index]);
			}
		}

		void math_functions::gpu_abs(const int N, const float* a, float* y)
		{
			abs_kernel << <CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS >> >(
				N, a, y);
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

		void math_functions::gpu_saxpy(cublasHandle_t cublas_handle_, const int N, const float alpha, const float* X, float* Y)
		{
			CUBLAS_CHECK(cublasSaxpy(cublas_handle_, N, &alpha, X, 1, Y, 1));
		}
	}
}

#endif