#include <malloc.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include <fstream>
#include "julius.hpp"


bool check_value(int M, int N, const float* C1, int ldc1, const float* C2, int ldc2, float thresh, bool show)
{
	int m, n;
	const float* Cptr1, * Cptr2, * C_c_ptr1, * C_c_ptr2;
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
			float real_thresh = __max(thresh, thresh * scale);
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

double _test_gemm(int M, int N, int K, int iters, float thresh, bool show)
{
	//int padK = K;// (K + 7) >> 3 << 3;
	float* A = (float*)malloc(M * K * sizeof(float));
	float* B = (float*)malloc(K * N * sizeof(float));
	float* C1 = (float*)malloc(M * N * sizeof(float));
	float* C2 = (float*)malloc(M * N * sizeof(float));
	float* q = (float*)malloc(32);
	const CBLAS_TRANSPOSE TransA = CblasTrans;
	const CBLAS_TRANSPOSE TransB = CblasNoTrans;
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;

	for (int i = 0; i < M * K; i++)
		A[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < K * N; i++)
		B[i] = rand() % 10001 / 5000.0f - 1.0f;
	for (int i = 0; i < M * N; i++)
	{
		C1[i] = rand() % 10001 / 5000.0f - 1.0f;
		C2[i] = C1[i];// rand() % 10001 / 5000.0f - 1.0f;
	}
		
	double t1 = omp_get_wtime(), t2, mul_count, gflops;
	double time1 = FLT_MAX;
	{
		for (int i = 0; i < iters; i++)
		{
			glasssix::excalibur::cblas_sgemm(glasssix::excalibur::CblasRowMajor, 
				glasssix::excalibur::CBLAS_TRANSPOSE(TransA),
				glasssix::excalibur::CBLAS_TRANSPOSE(TransB),
				M, N, K, 1.5, A, lda, B, ldb, 0.2f, C1, N);
		}
		t2 = omp_get_wtime();
		time1 = t2 - t1;
		mul_count = (double)M * N * K * iters;
		gflops = mul_count / (1 << 30) / (t2 - t1);
	    //printf("C1[0] = %f\n", C1[0]);
		//std::ofstream outfile("julius_blas.txt", std::ios::app);
		printf("%d x %d x %d * %d = %.3e, time = %.3f s, Julius gflops = %.3f\n", M, N, K, iters, mul_count, time1, gflops);
		/*outfile << gflops << std::endl;
		outfile.close();*/
	}



	t1 = omp_get_wtime();
	openblas_set_num_threads(1);
	for (int i = 0; i < iters; i++)
	{
		cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, 1.5, A, lda, B, ldb, 0.2, C2, N);
	}
	//printf("C2[0] = %f\n", C2[0]);
	t2 = omp_get_wtime();
	mul_count = (double)M * N * K * iters;
	gflops = mul_count / (1 << 30) / (t2 - t1);
	double  time2 = t2 - t1;
	printf("%d x %d x %d * %d = %.3e, time = %.3f s, gemm gflops = %.3f\n", M, N, K, iters, mul_count, time2, gflops);
	/*std::ofstream outfile("open_blas.txt", std::ios::app);
	outfile << gflops << std::endl;
	outfile.close();*/
	printf("check = %s\n", check_value(M, N, C1, N, C2, N, thresh, show) ? "True" : "False");
	free(A);
	free(B);
	free(C1);
	free(C2);
	free(q);


	return __min(time1, time2) / iters;
}

int main()
{
	for (int i = 1; i < 1000; i++)
	{
		int M = rand() % 1000 + 1;
		int N = rand() % 1000 + 1;
		int K = rand() % 1000 + 1;
		_test_gemm(i, i, i, 1, 1e-4, true);
	}
	return 0;
}