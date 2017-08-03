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

	template<>
	void Excalibur_MathFunctions::excalibur_gpu_gemm_batch<float>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float** A, const float** B, const float beta,
		float** C, const int batchCount)
	{
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB =
			(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasSgemmBatched(cublas_handle_, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N, batchCount));
	}

	template<>
	void Excalibur_MathFunctions::excalibur_gpu_gemm_batch<double>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const double alpha, const double** A, const double** B, const double beta,
		double** C, const int batchCount)
	{
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB =
			(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasDgemmBatched(cublas_handle_, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N, batchCount));
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

	template <typename Dtype>
	__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = alpha;
		}
	}

	template <typename Dtype>
	void Excalibur_MathFunctions::excalibur_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
		if (alpha == 0) 
		{
			CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));
			return;
		}
		// NOLINT_NEXT_LINE(whitespace/operators)
		set_kernel<Dtype> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, alpha, Y);
	}

	template void Excalibur_MathFunctions::excalibur_gpu_set<int>(const int N, const int alpha, int* Y);
	template void Excalibur_MathFunctions::excalibur_gpu_set<float>(const int N, const float alpha, float* Y);
	template void Excalibur_MathFunctions::excalibur_gpu_set<double>(const int N, const double alpha, double* Y);

	template <typename Dtype>
	__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] += alpha;
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_add_scalar(const int N, const float alpha, float* Y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		add_scalar_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, alpha, Y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_add_scalar(const int N, const double alpha, double* Y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		add_scalar_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, alpha, Y);
	}

	template <typename Dtype>
	__global__ void add_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = a[index] + b[index];
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_add<float>(const int N, const float* a, const float* b,
		float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		add_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_add<double>(const int N, const double* a, const double* b,
		double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		add_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <typename Dtype>
	__global__ void sub_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = a[index] - b[index];
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_sub<float>(const int N, const float* a, const float* b,
		float* y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		sub_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_sub<double>(const int N, const double* a, const double* b,
		double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		sub_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <typename Dtype>
	__global__ void mul_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = a[index] * b[index];
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_mul<float>(const int N, const float* a,
		const float* b, float* y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		mul_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_mul<double>(const int N, const double* a,
		const double* b, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		mul_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <typename Dtype>
	__global__ void div_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = a[index] / b[index];
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_div<float>(const int N, const float* a,
		const float* b, float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		div_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_div<double>(const int N, const double* a,
		const double* b, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		div_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, b, y);
	}

	template <typename Dtype>
	__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = abs(a[index]);
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_abs<float>(const int N, const float* a, float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		abs_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_abs<double>(const int N, const double* a, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		abs_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}


	template <typename Dtype>
	__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = exp(a[index]);
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_exp<float>(const int N, const float* a, float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		exp_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_exp<double>(const int N, const double* a, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		exp_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}

	template <typename Dtype>
	__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = log(a[index]);
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_log<float>(const int N, const float* a, float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		log_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_log<double>(const int N, const double* a, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		log_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, y);
	}

	template <typename Dtype>
	__global__ void powx_kernel(const int n, const Dtype* a,
		const Dtype alpha, Dtype* y) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			y[index] = pow(a[index], alpha);
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_powx<float>(const int N, const float* a,
		const float alpha, float* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		powx_kernel<float> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, alpha, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_gpu_powx<double>(const int N, const double* a,
		const double alpha, double* y) 
	{
		// NOLINT_NEXT_LINE(whitespace/operators)
		powx_kernel<double> << <EXCALIBUR_GET_BLOCKS(N), EXCALIBUR_CUDA_NUM_THREADS >> >(
			N, a, alpha, y);
	}

	DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
		- (x[index] < Dtype(0)));
	DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

#endif
}