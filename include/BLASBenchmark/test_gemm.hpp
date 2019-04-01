#pragma once
#ifndef _TEST_GEMM_HPP_
#define _TEST_GEMM_HPP_

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cfloat>
#include <algorithm>
#include <chrono>

inline bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh = 1e-5, bool show = false)
{
	int m, n;
	const float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float v1, v2;
	bool ret = true;
	for (m = 0, Cptr1 = C1, Cptr2 = C2; m < M; m++, Cptr1 += ldc1, Cptr2 += ldc2)
	{
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++)
		{
			v1 = *C_c_ptr1;
			v2 = *C_c_ptr2;
			float scale = std::max(fabs(v1), fabs(v2));
			float real_thresh = std::max(thresh, thresh*scale);
			if (fabs(v1 - v2) > real_thresh)
			{
				if (show)
					printf("%d,%d = %f %f\n", m, n, v1, v2);
				ret = false;
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}
	return ret;
}

inline double _test_gemm(int M, int N, int K, int iters = 1000, bool trans_a = false, bool trans_b = false, float thresh = 1e-4, bool show = false)
{
	int padK = K;// (K + 7) >> 3 << 3;
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padK*N * sizeof(float), 32);
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* C2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);

	int lda = trans_a ? M : padK;
	int ldb = trans_b ? padK: N;
	int transa_label = trans_a ? 112 : 111;
	int transb_label = trans_b ? 112 : 111;

	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < M*N; i++)
	{
		C1[i] = rand() % 10001 / 5000.0f - 1.0f;;
		C2[i] = C1[i];
	}
	//JuliusBLAS
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	double time1 = FLT_MAX;
#ifndef USE_MKL
	for (int i = 0; i < iters; i++)
	{
		glasssix::excalibur::cblas_sgemm(glasssix::excalibur::CblasRowMajor,
			(glasssix::excalibur::CBLAS_TRANSPOSE)transa_label, (glasssix::excalibur::CBLAS_TRANSPOSE)transb_label,
			M, N, K, 1.0, A, lda, B, ldb, 0.0f, C1, N);
	}
	t2 = omp_get_wtime();
	std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
	time1 = t2 - t1;
	time1 = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(p2 - p1).count() / 1000 / 1000 / 1000;
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / time1;
	//printf("C1[0] = %f\n", C1[0]);
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, juliusblas_gemm gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
#endif
	//OpenBLAS
	t1 = omp_get_wtime();
	std::chrono::time_point<std::chrono::system_clock> p3 = std::chrono::system_clock::now();
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(::CblasRowMajor, (::CBLAS_TRANSPOSE)transa_label, (::CBLAS_TRANSPOSE)transb_label,
			M, N, K, 1.0, A, lda, B, ldb, 0.0f, C2, N);
	}
	//printf("C2[0] = %f\n", C2[0]);
	t2 = omp_get_wtime();
	std::chrono::time_point<std::chrono::system_clock> p4 = std::chrono::system_clock::now();
	double time2 = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(p4 - p3).count() / 1000 / 1000 / 1000;
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / time2;
	//double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, openblas_gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);
#ifndef USE_MKL
	printf("check = %s\n", check_value(M, N, C1, N, C2, N, thresh, show) ? "True" : "False");
#endif
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C1);
	_aligned_free(C2);
	return std::min(time1, time2) / iters;
}

inline double _test_fgemm(int M, int N, int K, int iters = 1000, bool trans_a = false, bool trans_b = false, float thresh = 1e-4, bool show = false)
{
	int padK = K;// (K + 7) >> 3 << 3;
	signed char* A = (signed char*)_aligned_malloc(M*padK * sizeof(signed char), 32);
	signed char* B = (signed char*)_aligned_malloc(padK*N * sizeof(signed char), 32);
	int* C1 = (int*)_aligned_malloc(M*N * sizeof(int), 32);
	int* C2 = (int*)_aligned_malloc(M*N * sizeof(int), 32);

	int lda = trans_a ? M : padK;
	int ldb = trans_b ? padK : N;
	int transa_label = trans_a ? 112 : 111;
	int transb_label = trans_b ? 112 : 111;

	for (int i = 0; i < M*padK; i++)
		A[i] = (signed char)(rand() % 10001 / 5000.0f - 1.0f);
	for (int i = 0; i < padK*N; i++)
		B[i] = (signed char)(rand() % 10001 / 5000.0f - 1.0f);
	for (int i = 0; i < M*N; i++)
	{
		C1[i] = (int)(rand() % 10001 / 5000.0f - 1.0f);
		C2[i] = C1[i];
	}
	//JuliusBLAS
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	double time1 = FLT_MAX;
	for (int i = 0; i < iters; i++)
	{
		glasssix::excalibur::cblas_fgemm(glasssix::excalibur::CblasRowMajor, (glasssix::excalibur::CBLAS_TRANSPOSE)transa_label, (glasssix::excalibur::CBLAS_TRANSPOSE)transb_label,
			M, N, K, 1.0, A, lda, B, ldb, 0.0f, C2, N);
	}
	t2 = omp_get_wtime();
	std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
	time1 = t2 - t1;
	time1 = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(p2 - p1).count() / 1000 / 1000 / 1000;
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / time1;
	//printf("C1[0] = %f\n", C1[0]);
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, juliusblas_fgemm gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C1);
	_aligned_free(C2);
	return std::min(time1, time1) / iters;
}

#endif // !_TEST_GEMM_HPP_
