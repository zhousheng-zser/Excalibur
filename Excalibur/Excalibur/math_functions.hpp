#pragma once
#ifndef _MATH_FUNCTIONS_HPP_
#define _MATH_FUNCTIONS_HPP_
#include "mkl_alternate.hpp"

namespace excalibur
{
	class math_functions
	{
	public:
		math_functions();
		~math_functions();

		static void cpu_sgemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const float alpha, const float* A, const float* B, const float beta,
			float* C);

		static void excalibur_copy(const int N, const float *X, float *Y, int device);

		static void cpu_set(const int N, const float alpha, float* Y);

#ifdef  USE_CUDA
		static void gpu_sgemm(cublasHandle_t cublas_handle_, const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const float alpha, const float* A, const float* B, const float beta,
			float* C);

		static void gpu_set(const int N, const float alpha, float* Y);
#endif
	};
}

#endif // _MATH_FUNCTIONS_HPP_