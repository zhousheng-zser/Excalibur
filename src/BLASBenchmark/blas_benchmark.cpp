#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cfloat>
#include <glasssix\accelerator.hpp>
#ifndef USE_MKL
#include "../../include/Julius/julius.hpp"
#include <cblas.h>
#endif // !USE_MKL
#include "test_gemv.hpp"
#include "test_gemm.hpp"
#include "gemm_problems.hpp"

int main()
{
	//for (int i = 0; i < 100000; i++)
	//{
	//	int M = rand() % 4000 + 1;
	//	int N = rand() % 4000 + 1;
	//	int K = rand() % 4000 + 1;
	//	_test_gemm(M, N, K, 1, false, true, 1e-4, true);
	//	//_test_gemv(M, N, 1, true, 1e-4, true);
	//}
	for (int i = 0; i < inference_cassius_set.size(); i++)
	{
		int M, N, K;
		bool transa, transb;
		std::tie(M, N, K, transa, transb) = inference_cassius_set[i];
		_test_gemm(M, N, K, 100, transa, transb, 1e-4, true);
	}

	for (int i = 1; i < 4000; i++)
	{
		_test_gemm(i, i, i, 1, false, false, 1e-4, true);
	}
	return 0;
}