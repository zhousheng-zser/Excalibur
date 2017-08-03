#pragma once
#ifndef _EXCALIBUR_MATHFUNCTIONS_HPP_
#define _EXCALIBUR_MATHFUNCTIONS_HPP_
#include "Mordred_BasicMath.hpp"

namespace Excalibur
{
	class Excalibur_MathFunctions
	{
		Avalon mode;
		int gpu_device_;
	public:
		Excalibur_MathFunctions();
//#ifdef USE_CUDA
		Excalibur_MathFunctions(int gpu_decvice, Avalon mode_);
//#endif
		~Excalibur_MathFunctions();

		//If there does not exist an instance of this class in this thread, create one and return.
		//static Excalibur_MathFunctions& Get();

		void set_Avalon(Avalon mode_);

#ifdef USE_CUDA
		cublasHandle_t cublas_handle_;
		inline cublasHandle_t cublas_handle()
		{
			return cublas_handle_;
		}

#ifdef USE_CUDNN
		cudnnHandle_t cudnn_handle_;
		inline static cudnnHandle_t cudnn_handle()
		{
			return Get().cudnn_handle_;
		}
#endif
#endif

		template <typename Dtype>
		void excalibur_cpu_gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
			Dtype* C);
#ifdef USE_MKL
		template <typename Dtype>
		void excalibur_cpu_gemm_batch(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const Dtype alpha, const Dtype** A, const Dtype** B, const Dtype beta,
			Dtype** C, const int group_count, const int group_size);
#endif
		template <typename Dtype>
		void excalibur_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
			const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
			Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_axpy(const int N, const Dtype alpha, const Dtype* X,
			Dtype* Y);

		template <typename Dtype>
		void excalibur_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
			const Dtype beta, Dtype* Y);

		template <typename Dtype>
		void excalibur_cpu_copy(const int N, const Dtype *X, Dtype *Y);

		template <typename Dtype>
		void excalibur_cpu_set(const int N, const Dtype alpha, Dtype *X);

		inline void excalibur_cpu_memset(const size_t N, const int alpha, void* X)
		{
			memset(X, alpha, N);
		}

		template <typename Dtype>
		void excalibur_cpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

		template <typename Dtype>
		void excalibur_cpu_scal(const int N, const Dtype alpha, Dtype *X);

		template <typename Dtype>
		void excalibur_cpu_sqr(const int N, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_exp(const int n, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_log(const int n, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_cpu_abs(const int n, const Dtype* a, Dtype* y);


		template <typename Dtype>
		Dtype excalibur_cpu_dot(const int n, const Dtype* x, const Dtype* y)
		{
			return excalibur_cpu_strided_dot(n, x, 1, y, 1);
		}

		template <typename Dtype>
		Dtype excalibur_cpu_strided_dot(const int n, const Dtype* x, const int incx,
			const Dtype* y, const int incy);

		// Returns the sum of the absolute values of the elements of vector x
		template <typename Dtype>
		Dtype excalibur_cpu_asum(const int n, const Dtype* x);

		// the branchless, type-safe version from
		// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
		template<typename Dtype>
		inline int8_t excalibur_cpu_sign(Dtype val)
		{
			return (Dtype(0) < val) - (val < Dtype(0));
		}

		template <typename Dtype>
		void excalibur_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

		template<typename Dtype> 
		void excalibur_cpu_sign(const int n, const Dtype* x, Dtype* y)
		{
			CHECK_GT(n, 0); CHECK(x); CHECK(y);
			for (int i = 0; i < n; ++i)
			{
				y[i] = excalibur_cpu_sign<Dtype>(x[i]);
			}
		}

		template<typename Dtype>
		void excalibur_cpu_sgnbit(const int n, const Dtype* x, Dtype* y)
		{
			CHECK_GT(n, 0); CHECK(x); CHECK(y);
			for (int i = 0; i < n; ++i)
			{
				y[i] = static_cast<bool>((std::signbit)(x[i]));
			}
		}

		template<typename Dtype>
		void excalibur_cpu_fabs(const int n, const Dtype* x, Dtype* y)
		{
			CHECK_GT(n, 0); CHECK(x); CHECK(y);
			for (int i = 0; i < n; ++i)
			{
				y[i] = std::fabs(x[i]);
			}
		}


#ifdef USE_CUDA  // GPU

		// Decaf gpu gemm provides an interface that is almost the same as the cpu
		// gemm function - following the c convention and calling the fortran-order
		// gpu code under the hood.
		template <typename Dtype>
		void excalibur_gpu_gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
			Dtype* C);

		template <typename Dtype>
		void excalibur_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
			const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
			Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
			Dtype* Y);

		template <typename Dtype>
		void excalibur_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
			const Dtype beta, Dtype* Y);

		void excalibur_gpu_memcpy(const size_t N, const void *X, void *Y);

		template <typename Dtype>
		void excalibur_gpu_set(const int N, const Dtype alpha, Dtype *X);

		inline void excalibur_gpu_memset(const size_t N, const int alpha, void* X) 
		{
			CUDA_CHECK(cudaMemset(X, alpha, N)); 
		}

		template <typename Dtype>
		void excalibur_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

		template <typename Dtype>
		void excalibur_gpu_scal(const int N, const Dtype alpha, Dtype *X);

		template <typename Dtype>
		void excalibur_gpu_scal(const int N, const Dtype alpha, Dtype* X, cudaStream_t str);

		template <typename Dtype>
		void excalibur_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_abs(const int n, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_exp(const int n, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_log(const int n, const Dtype* a, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

		template <typename Dtype>
		void excalibur_gpu_asum(const int n, const Dtype* x, Dtype* y);

		template<typename Dtype>
		void excalibur_gpu_sign(const int n, const Dtype* x, Dtype* y);

		template<typename Dtype>
		void excalibur_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_fabs(const int n, const Dtype* x, Dtype* y);

		template <typename Dtype>
		void excalibur_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);


#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void excalibur_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<EXCALIBUR_GET_BLOCKS(n), EXCALIBUR_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void excalibur_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<EXCALIBUR_GET_BLOCKS(n), EXCALIBUR_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif 
	};
}


#endif //_EXCALIBUR_MATHFUNCTIONS_HPP_