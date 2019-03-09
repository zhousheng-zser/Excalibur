#include "../../include/Julius/julius.hpp"
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cfloat>
#include <cblas.h>
#include <glasssix\accelerator.hpp>
#include "test_gemv.hpp"
#include "test_gemm.hpp"

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