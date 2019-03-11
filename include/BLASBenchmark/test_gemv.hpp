#pragma once
#ifndef _TEST_GEMV_HPP_
#define _TEST_GEMV_HPP_

#include "../../include/Julius/julius.hpp"
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cfloat>
#include <cblas.h>
#include <algorithm>

inline bool check_value(int M, const float* y1, int incy1, const float* y2, int incy2, float thresh = 1e-5, bool show = false)
{
	for (int i = 0; i < M; i++)
	{
		float v1 = y1[i * incy1];
		float v2 = y2[i * incy2];
		float scale = std::max(fabs(v1), fabs(v2));
		float real_thresh = std::max(thresh, thresh*scale);
		if (fabs(v1 - v2) > real_thresh)
		{
			if (show)
				printf("%d,%d = %f %f\n", i, 1, v1, v2);
			return false;
		}
	}
	return true;
}

inline void _test_gemv(int M, int N, int iters = 1000, bool trans_a = false, float thresh = 1e-4, bool show = false)
{
	float* A = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* x = (float*)_aligned_malloc((trans_a ? M : N) * sizeof(float), 32);
	float* y1 = (float*)_aligned_malloc((trans_a ? N : M) * sizeof(float), 32);
	float* y2 = (float*)_aligned_malloc((trans_a ? N : M) * sizeof(float), 32);
	int transa_label = trans_a ? 112 : 111;
	for (int i = 0; i < M*N; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < 1 * (trans_a ? M : N); i++)
		x[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < 1 * (trans_a ? N : M); i++)
	{
		y1[i] = rand() % 10001 / 5000.0f - 1.0f;
		y2[i] = y1[i];
	}
	double  mul_count, gflops;
	double t1 = omp_get_wtime(), t2;
	double time1 = FLT_MAX;
	{
		for (int i = 0; i < iters; i++)
		{
			glasssix::excalibur::cblas_sgemv(glasssix::excalibur::CblasRowMajor, (glasssix::excalibur::CBLAS_TRANSPOSE)transa_label,
				M, N, 1.1f, A, N, x, 1, 0.5f, y1, 1);
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N * 1 * iters;
		gflops = mul_count / (1 << 30) / (time1);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, juliusblas_gemv gflops = %.3f\n", M, N, 1, iters, mul_count, time1, gflops);
	}

	{
		t1 = omp_get_wtime();
		for (int i = 0; i < iters; i++)
		{
			cblas_sgemv(CblasRowMajor, (::CBLAS_TRANSPOSE)transa_label, M, N, 1.1f, A, N, x, 1, 0.5f, y2, 1);
		}
		t2 = omp_get_wtime();
		mul_count = (double)M*N * 1 * iters;
		double  time2 = t2 - t1;
		gflops = mul_count / (1 << 30) / (time2 / 1);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, openblas_gemv gflops = %.3f\n", M, N, 1, iters, mul_count, time2, gflops);
	}
	printf("check = %s\n", check_value(trans_a ? N : M, y1, 1, y2, 1, thresh, show) ? "True" : "False");
	_aligned_free(A);
	_aligned_free(x);
	_aligned_free(y1);
	_aligned_free(y2);
}

#endif // !_TEST_GEMV_HPP_

