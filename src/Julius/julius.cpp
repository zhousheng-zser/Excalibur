#include "julius_dot.hpp"
#include "julius_sdot.hpp"
#include "julius_axpby.hpp"
#include "julius_gemv.hpp"
#include "julius_gemm.hpp"
#include "julius.hpp"

namespace glasssix
{
	namespace excalibur
	{
		float  cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy)
		{
			if (n <= 0)
			{
				return alpha;
			}
			else
			{
				CHECK_GT(incx, 1);
				CHECK_GT(incy, 1);
				return juliusblas::cblas_sdsdot(n, alpha, x, incx, y, incy);
			}
		}

		double cblas_dsdot(const int n, const float *x, const int incx, const float *y, const int incy)
		{
			if (n <= 0)
			{
				return 0.0;
			}
			else
			{
				CHECK_GT(incx, 1);
				CHECK_GT(incy, 1);
				return juliusblas::cblas_dsdot(n, x, incx, y, incy);
			}
		}

		float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy)
		{
			if (n <= 0)
			{
				return 0.0f;
			}
			else
			{
				return juliusblas::cblas_sdot(n, x, incx, y, incy);
			}
		}

		double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy)
		{
			if (n <= 0)
			{
				return 0.0;
			}
			else
			{
				return juliusblas::cblas_ddot(n, x, incx, y, incy);
			}
		}

		void cblas_saxpby(const int N, const float alpha, const float* X,
			const int incX, const float beta, float* Y, const int incY)
		{
			juliusblas::cblas_saxpby(N, alpha, X, incX, beta, Y, incY);
		}

		void cblas_daxpby(const int N, const double alpha, const double* X,
			const int incX, const double beta, double* Y, const int incY)
		{
			NOT_IMPLEMENTED;
		}

		void cblas_sgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int M, const int N,
			const float alpha, const float  *A, const int lda, const float  *x, const int incx, const float beta, float  *y, const int incy)
		{
			switch (order)
			{
			case CblasRowMajor:
				if (trans == CblasNoTrans)
				{
					juliusblas::cblas_sgemv_AnoTrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else if (trans == CblasTrans)
				{
					juliusblas::cblas_sgemv_ATrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemv.";
				}
				break;
			case CblasColMajor:
				NOT_IMPLEMENTED << " with CblasColMajor in Julius sgemv.";
				break;
			default:
				if (trans == CblasNoTrans)
				{
					juliusblas::cblas_sgemv_AnoTrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else if (trans == CblasTrans)
				{
					juliusblas::cblas_sgemv_ATrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemv.";
				}
				break;
			}
		}

		void cblas_dgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int m, const int n,
			const double alpha, const double  *a, const int lda, const double  *x, const int incx, const double beta, double  *y, const int incy)
		{
			NOT_IMPLEMENTED;
		}

		void cblas_sgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb,
			const float beta, float* C, const int ldc)
		{
			switch (Order)
			{
			case CblasRowMajor:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			case CblasColMajor:
				NOT_IMPLEMENTED << " with CblasColMajor in Julius sgemm.";
				break;
			default:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			}
		}

		void cblas_dgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb,
			const double beta, double* C, const int ldc)
		{
			NOT_IMPLEMENTED;
		}
	}
}