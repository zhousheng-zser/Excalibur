#include "Accelerator.hpp"
#include "Excalibur_MathFunctions.hpp"

namespace Excalibur
{
#ifdef USE_CUDA
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

	template<>
	void Excalibur_MathFunctions::excalibur_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const double alpha, const double* A, const double* B, const double beta,
		double* C)
	{
		// Note that cublas follows fortran order.
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB =
			(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasDgemm(cublas_handle_, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
	}


	template <>
	void Excalibur_MathFunctions::excalibur_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const float alpha, const float* A, const float* x,
		const float beta, float* y) {
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK(cublasSgemv(cublas_handle_, cuTransA, N, M, &alpha,
			A, N, x, 1, &beta, y, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const double alpha, const double* A, const double* x,
		const double beta, double* y) {
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK(cublasDgemv(cublas_handle_, cuTransA, N, M, &alpha,
			A, N, x, 1, &beta, y, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_axpy<float>(const int N, const float alpha, const float* X,
		float* Y) {
		CUBLAS_CHECK(cublasSaxpy(cublas_handle_, N, &alpha, X, 1, Y, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_axpy<double>(const int N, const double alpha, const double* X,
		double* Y) {
		CUBLAS_CHECK(cublasDaxpy(cublas_handle_, N, &alpha, X, 1, Y, 1));
	}

	void Excalibur_MathFunctions::excalibur_gpu_memcpy(const size_t N, const void* X, void* Y) {
		if (X != Y) {
			CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_scal<float>(const int N, const float alpha, float *X) {
		CUBLAS_CHECK(cublasSscal(cublas_handle_, N, &alpha, X, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_scal<double>(const int N, const double alpha, double *X) {
		CUBLAS_CHECK(cublasDscal(cublas_handle_, N, &alpha, X, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_axpby<float>(const int N, const float alpha, const float* X,
		const float beta, float* Y) {
		excalibur_gpu_scal<float>(N, beta, Y);
		excalibur_gpu_axpy<float>(N, alpha, X, Y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_axpby<double>(const int N, const double alpha, const double* X,
		const double beta, double* Y) {
		excalibur_gpu_scal<double>(N, beta, Y);
		excalibur_gpu_axpy<double>(N, alpha, X, Y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_dot<float>(const int n, const float* x, const float* y,
		float* out) {
		CUBLAS_CHECK(cublasSdot(cublas_handle_, n, x, 1, y, 1, out));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_dot<double>(const int n, const double* x, const double* y,
		double * out) {
		CUBLAS_CHECK(cublasDdot(cublas_handle_, n, x, 1, y, 1, out));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_asum<float>(const int n, const float* x, float* y) {
		CUBLAS_CHECK(cublasSasum(cublas_handle_, n, x, 1, y));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_asum<double>(const int n, const double* x, double* y) {
		CUBLAS_CHECK(cublasDasum(cublas_handle_, n, x, 1, y));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_scale<float>(const int n, const float alpha, const float *x,
		float* y) {
		CUBLAS_CHECK(cublasScopy(cublas_handle_, n, x, 1, y, 1));
		CUBLAS_CHECK(cublasSscal(cublas_handle_, n, &alpha, y, 1));
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_scale<double>(const int n, const double alpha, const double *x,
		double* y) {
		CUBLAS_CHECK(cublasDcopy(cublas_handle_, n, x, 1, y, 1));
		CUBLAS_CHECK(cublasDscal(cublas_handle_, n, &alpha, y, 1));
	}
#endif
}