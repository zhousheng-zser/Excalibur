#pragma once
#ifndef _JULIUS_HPP_
#define _JULIUS_HPP_

namespace glasssix
{
	namespace excalibur
	{
		enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 };
		enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
		enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
		enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
		enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
		enum CBLAS_STORAGE { CblasPacked = 151 };
		enum CBLAS_IDENTIFIER { CblasAMatrix = 161, CblasBMatrix = 162 };
		enum CBLAS_OFFSET { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 };

		class julius
		{
			
		public:
			julius();
			~julius();
			//Level 1 functions
			static void cblas_saxpby(const int N, const float alpha, const float* X,
				const int incX, const float beta, float* Y, const int incY);

			static void cblas_daxpby(const int N, const double alpha, const double* X,
				const int incX, const double beta, double* Y, const int incY);

			//Level 2 functions
			static void cblas_sgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int m, const int n,
				const float alpha, const float  *a, const int lda, const float  *x, const int incx, const float beta, float  *y, const int incy);

			static void cblas_dgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int m, const int n,
				const double alpha, const double  *a, const int lda, const double  *x, const int incx, const double beta, double  *y, const int incy);

			//Level 3 functions
			static void cblas_sgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
				const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb,
				const float beta, float* C, const int ldc);

			static void cblas_dgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
				const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb,
				const double beta, double* C, const int ldc);
		};
	}
}
#endif // !_JULIUS_HPP_



