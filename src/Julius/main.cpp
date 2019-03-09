#include "julius.hpp"
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cfloat>
#include <cblas.h>
#include <glasssix\accelerator.hpp>
#include <glasssix\timer.hpp>


bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh = 1e-5, bool show = false)
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
			float scale = __max(fabs(v1), fabs(v2));
			float real_thresh = __max(thresh, thresh*scale);
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

bool check_value(int M,  const float* y1, int incy1, const float* y2, int incy2, float thresh = 1e-5, bool show = false)
{
	for (int i = 0; i < M; i++)
	{
		float v1 = y1[i * incy1];
		float v2 = y2[i * incy2];
		float scale = __max(fabs(v1), fabs(v2));
		float real_thresh = __max(thresh, thresh*scale);
		if (fabs(v1 - v2) > real_thresh)
		{
			if (show)
				printf("%d,%d = %f %f\n", i, 1, v1, v2);
			return false;
		}
	}
	return true;
}

double _test_gemm(int M, int N, int K, int iters = 1000, float thresh = 1e-4, bool show = false)
{
	int padK = (K + 7) >> 3 << 3;
	float* A = (float*)_aligned_malloc(M*padK * sizeof(float), 32);
	float* B = (float*)_aligned_malloc(padK*N * sizeof(float), 32);
	float* C1 = (float*)_aligned_malloc(M*N * sizeof(float), 32);	
   	float* C2 = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* q = (float*)_aligned_malloc(32, 32);


	for (int i = 0; i < M*padK; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < padK*N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < M*N; i++)
	{
		C1[i] = rand() % 10001 / 5000.0f - 1.0f;;
		C2[i] = C1[i];
	}
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	double time1 = FLT_MAX;
	{
		for (int i = 0; i < iters; i++)
		{
			//glasssix::gemm_32f_AnoTrans_Btrans_auto(M, N, K, A, padK, B, padK, C1, N);
			glasssix::excalibur::cblas_sgemm(glasssix::excalibur::CblasRowMajor, glasssix::excalibur::CblasNoTrans, glasssix::excalibur::CblasNoTrans,
				M, N, K, 1.0, A, padK, B, N, 0.5f, C1, N);
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*K*iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
		//printf("C1[0] = %f\n", C1[0]);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, juliusblas_gemm gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
	}



	t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(::CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, padK, B, N, 0.5f, C2, N);
	}
	//printf("C2[0] = %f\n", C2[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M*N*K*iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);
	double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, openblas_gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);

	printf("check = %s\n", check_value(M, N, C1, N, C2, N, thresh, show) ? "True" : "False");
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C1);
	_aligned_free(C2);
	_aligned_free(q);


	return __min(time1, time2) / iters;
}

void _test_gemv(int M, int N, int iters = 1000, float thresh = 1e-4, bool show = false)
{
	float* A = (float*)_aligned_malloc(M*N * sizeof(float), 32);
	float* x = (float*)_aligned_malloc(N * sizeof(float), 32);
	float* y1 = (float*)_aligned_malloc(M * sizeof(float), 32);
	float* y2 = (float*)_aligned_malloc(M * sizeof(float), 32);

	for (int i = 0; i < M*N; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < 1*N; i++)
		x[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < 1 * M; i++)
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
			glasssix::excalibur::cblas_sgemv(glasssix::excalibur::CblasRowMajor, glasssix::excalibur::CblasNoTrans,
				M, N, 1.1f, A, N, x, 1, 0.5f, y1, 1);
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M*N*1*iters;
		gflops = mul_count / (1 << 30) / (time1);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, juliusblas_gemv gflops = %.3f\n", M, N, 1, iters, mul_count, time1, gflops);
	}

	{
		t1 = omp_get_wtime();
		for (int i = 0; i < iters; i++)
		{
			cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.1f, A, N, x, 1, 0.5f, y2, 1);
		}
		t2 = omp_get_wtime();
		mul_count = (double)M*N*1*iters;
		double  time2 = t2 - t1;
		gflops = mul_count / (1 << 30) / (time2 / 1);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, openblas_gemv gflops = %.3f\n", M, N, 1, iters, mul_count, time2, gflops);
	}
	printf("check = %s\n", check_value(M, y1, 1, y2, 1, thresh, show) ? "True" : "False");
	_aligned_free(A);
	_aligned_free(x);
	_aligned_free(y1);
	_aligned_free(y2);
}

int main()
{
	for (int i = 0; i < 100000; i++)
	{
		int M = rand() % 1000 + 1;
		int N = rand() % 1000 + 1;
		int K = rand() % 1000 + 1;
		if (M%4!=0||N%4!=0)
		{
			continue;
		}
		_test_gemm(M, N, K, 1, 1e-4, true);
		//_test_gemv(M, N, 1, 1e-4, true);
	}
	return 0;
}